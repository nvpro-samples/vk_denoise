/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.glsl"

#include "nvvkhl/shaders/constants.glsl"
#include "nvvkhl/shaders/ggx.glsl"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/dh_hdr.h"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/random.glsl"
#include "nvvkhl/shaders/bsdf_functions.h"


hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(buffer_reference, scalar) readonly buffer Vertices  { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices   { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer PrimMeshInfos { PrimMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer Materials { GltfShadeMaterial m[]; };

#include "get_hit.glsl"

layout(set = 0, binding = eTlas ) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures
layout(set = 2, binding = eImpSamples,  scalar)	buffer _EnvAccel { EnvAccel envSamplingData[]; };
layout(set = 2, binding = eHdr) uniform sampler2D hdrTexture;

layout(push_constant) uniform RtxPushConstant_ { PushConstant pc; };
// clang-format on

// Includes depending on layout description
#include "nvvkhl/shaders/pbr_mat_eval.glsl"  // texturesMap
#include "nvvkhl/shaders/hdr_env_sampling.glsl"


void stopPath()
{
  payload.hitT = INFINITE;
}

struct ShadingResult
{
  vec3 weight;
  vec3 radiance;
  vec3 rayOrigin;
  vec3 rayDirection;
};

// --------------------------------------------------------------------
// Sampling the Sun or the HDR
// - Returns
//      The contribution divided by PDF
//      The direction to the light source
//      The PDF
//
vec3 sampleLights(in HitState state, inout uint seed, out vec3 dirToLight, out float lightPdf)
{
  vec3 rand_val     = vec3(rand(seed), rand(seed), rand(seed));
  vec4 radiance_pdf = environmentSample(hdrTexture, rand_val, dirToLight);
  vec3 radiance     = radiance_pdf.xyz;
  lightPdf          = radiance_pdf.w;

  // Apply rotation and environment intensity
  dirToLight = rotate(dirToLight, vec3(0, 1, 0), frameInfo.envRotation);
  radiance *= frameInfo.clearColor.xyz;

  // Return radiance over pdf
  return radiance / lightPdf;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
ShadingResult shading(in PbrMaterial pbrMat, in HitState hit)
{
  ShadingResult result;

  vec3 to_eye = -gl_WorldRayDirectionEXT;

  result.radiance = pbrMat.emissive;  // Emissive material


  // Light contribution; can be environment or punctual lights
  vec3  contribution         = vec3(0);
  vec3  dirToLight           = vec3(0);
  float lightPdf             = 0.F;
  vec3  lightRadianceOverPdf = sampleLights(hit, payload.seed, dirToLight, lightPdf);

  const bool nextEventValid = (dot(dirToLight, hit.geonrm) > 0.0f) && lightPdf != 0.0f;

  // Evaluate BSDF
  if(nextEventValid)
  {
    BsdfEvaluateData evalData;
    evalData.k1   = -gl_WorldRayDirectionEXT;
    evalData.k2   = dirToLight;

    bsdfEvaluate(evalData, pbrMat);

    if(evalData.pdf > 0.0)
    {
      const float mis_weight = lightPdf / (lightPdf + evalData.pdf);

      // sample weight
      const vec3 w = lightRadianceOverPdf * mis_weight;
      contribution += w * evalData.bsdf_diffuse;
      contribution += w * evalData.bsdf_glossy;
    }
  }

  // Sample BSDF
  {
    BsdfSampleData sampleData;
    sampleData.k1   = -gl_WorldRayDirectionEXT;  // outgoing direction
    sampleData.xi   = vec4(rand(payload.seed), rand(payload.seed), rand(payload.seed), rand(payload.seed));
    bsdfSample(sampleData, pbrMat);

    result.weight       = sampleData.bsdf_over_pdf;
    result.rayDirection = sampleData.k2;
    vec3 offsetDir      = dot(result.rayDirection, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
    result.rayOrigin    = offsetRay(hit.pos, offsetDir);

    if(sampleData.event_type == BSDF_EVENT_ABSORB)
    {
      stopPath();
      return result;  // Need to add the contribution ?
    }
  }

  if(nextEventValid)
  {
    // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
    uint ray_flag = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
    payload.hitT = 0.0F;
    traceRayEXT(topLevelAS, ray_flag, 0xFF, 0, 0, 0, result.rayOrigin, 0.001, dirToLight, INFINITE, 0);
    // If hitting nothing, add light contribution
    if(payload.hitT == INFINITE)
      result.radiance += contribution;
    payload.hitT = gl_HitTEXT;
  }

  return result;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  PrimMeshInfos pInfo_ = PrimMeshInfos(sceneDesc.primInfoAddress);
  PrimMeshInfo  pinfo  = pInfo_.i[gl_InstanceCustomIndexEXT];

  HitState hit = getHitState(pinfo.vertexAddress, pinfo.indexAddress);

  // Scene materials
  uint      matIndex  = max(0, pinfo.materialIndex);  // material of primitive mesh
  Materials materials = Materials(sceneDesc.materialAddress);

  // Material of the object and evaluated material (includes textures)
  GltfShadeMaterial mat    = materials.m[matIndex];
  PbrMaterial       pbrMat = evaluateMaterial(mat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

  payload.hitT         = gl_HitTEXT;
  ShadingResult result = shading(pbrMat, hit);

  payload.weight       = result.weight;
  payload.contrib      = result.radiance;
  payload.rayOrigin    = result.rayOrigin;
  payload.rayDirection = result.rayDirection;

  // -- Debug --
  //  payload.contrib = hit.nrm * .5 + .5;
  //  payload.contrib = matEval.albedo.xyz;
  //  payload.contrib = mat.pbrBaseColorFactor.xyz;
  //  payload.contrib = matEval.tangent * .5 + .5;
  //  payload.contrib = vec3(matEval.metallic);
  //  payload.contrib = vec3(matEval.roughness);
  //  stopRay();
}
