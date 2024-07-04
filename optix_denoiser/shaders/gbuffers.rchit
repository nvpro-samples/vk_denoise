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
#include "payload.glsl"
#include "dh_bindings.h"

#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/ggx.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/dh_hdr.h"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/pbr_mat_struct.h"
#include "nvvkhl/shaders/vertex_accessor.h"


hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 1) rayPayloadInEXT GbufferPayload payloadGbuf;

layout(buffer_reference, scalar) readonly buffer Materials { GltfShadeMaterial m[]; };

layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures
// clang-format on


#include "nvvkhl/shaders/pbr_mat_eval.h"
#include "compress.glsl"

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3  nrm;
  vec2  uv;
  vec3  tangent;
  vec3  bitangent;
  float bitangentSign;
};


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
HitState GetHitState(RenderPrimitive renderPrim)
{
  HitState hit;

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = getTriangleIndices(renderPrim, gl_PrimitiveID);

  // Position
  //  const vec3 pos0     = v0.position.xyz;
  //  const vec3 pos1     = v1.position.xyz;
  //  const vec3 pos2     = v2.position.xyz;
  //  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  //  hit.pos             = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0        = getVertexNormal(renderPrim, triangleIndex.x);
  const vec3 nrm1        = getVertexNormal(renderPrim, triangleIndex.y);
  const vec3 nrm2        = getVertexNormal(renderPrim, triangleIndex.z);
  const vec3 normal      = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 worldNormal = normalize(vec3(normal * gl_WorldToObjectEXT));
  //  const vec3 geomNormal  = normalize(cross(pos1 - pos0, pos2 - pos0));
  hit.nrm = dot(worldNormal, gl_WorldRayDirectionEXT) <= 0.0 ? worldNormal : -worldNormal;  // Front-face

  // TexCoord
  const vec2 uv0 = getVertexTexCoord0(renderPrim, triangleIndex.x);
  const vec2 uv1 = getVertexTexCoord0(renderPrim, triangleIndex.y);
  const vec2 uv2 = getVertexTexCoord0(renderPrim, triangleIndex.z);
  hit.uv         = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

  // Tangent - Bitangent
  const vec4 tng0    = getVertexTangent(renderPrim, triangleIndex.x);
  const vec4 tng1    = getVertexTangent(renderPrim, triangleIndex.y);
  const vec4 tng2    = getVertexTangent(renderPrim, triangleIndex.z);
  vec3       tangent = normalize(tng0.xyz * barycentrics.x + tng1.xyz * barycentrics.y + tng2.xyz * barycentrics.z);
  vec3       world_tangent  = normalize(vec3(tangent * gl_WorldToObjectEXT));
  vec3       world_binormal = cross(worldNormal, world_tangent) * tng0.w;
  hit.tangent               = world_tangent;
  hit.bitangent             = world_binormal;
  hit.bitangentSign         = tng0.w;

  return hit;
}


//-----------------------------------------------------------------------
// Returning the G-Buffer informations
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  RenderNode renderNode       = RenderNodeBuf(sceneDesc.renderNodeAddress)._[gl_InstanceID];
  RenderPrimitive renderPrim  = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[gl_InstanceCustomIndexEXT];

  HitState hit = GetHitState(renderPrim);

  // Scene materials
  uint      matIndex  = max(0, renderNode.materialID);  // material of primitive mesh
  Materials materials = Materials(sceneDesc.materialAddress);

  // Material of the object and evaluated material (includes textures)
  GltfShadeMaterial mat    = materials.m[matIndex];
  PbrMaterial       pbrMat = evaluateMaterial(mat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

  payloadGbuf.packAlbedo = packUnorm4x8(vec4(pbrMat.baseColor, pbrMat.opacity));
  //payloadGbuf.packAlbedo = packUnorm4x8(mat.pbrBaseColorFactor);
  payloadGbuf.packNormal = compress_unit_vec(pbrMat.N);
}
