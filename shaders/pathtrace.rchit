#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
//#extension GL_EXT_nonuniform_qualifier : enable

#include "../config.hpp"
#include "raycommon.glsl"
#include "sampling.glsl"

// Payload information of the ray returning: 0 hit, 2 shadow
layout(location = 0) rayPayloadInEXT PerRayData_pathtrace prd;
layout(location = 2) rayPayloadEXT bool payloadShadow;

// Raytracing hit attributes: barycentrics
hitAttributeEXT vec2 attribs;

// clang-format off
layout(binding = B_BVH, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = B_SCENE, set = 0) uniform _ubo { Scene SceneInfo; };
layout(binding = B_PRIM_INFO, set = 0) readonly buffer _OffsetIndices { primInfo InstanceInfo[]; } ;
layout(binding = B_VERTEX, set = 0) readonly buffer _VertexBuf { float VertexBuf[]; } ;
layout(binding = B_INDEX, set = 0) readonly buffer _Indices { uint IndexBuf[]; } ;
layout(binding = B_NORMAL, set = 0) readonly buffer _NormalBuf { float NormalBuf[]; } ;
layout(binding = B_MATERIAL, set = 0) readonly buffer _MaterialBuffer { Material m[]; } MaterialBuffer;
// clang-format on

// Return the vertex position
vec3 getVertex(uint index)
{
  vec3 vp;
  vp.x = VertexBuf[3 * index + 0];
  vp.y = VertexBuf[3 * index + 1];
  vp.z = VertexBuf[3 * index + 2];
  return vp;
}

vec3 getNormal(uint index)
{
  vec3 vp;
  vp.x = NormalBuf[3 * index + 0];
  vp.y = NormalBuf[3 * index + 1];
  vp.z = NormalBuf[3 * index + 2];
  return vp;
}

// Structure of what a vertex is
struct Vertex
{
  vec3 pos;
  vec3 nrm;
};


// Getting the interpolated vertex
Vertex getVertex(ivec3 trianglIndex, uint vertexOffset, vec3 barycentrics)
{
  Vertex v0, v1, v2;
  v0.pos = getVertex(trianglIndex.x + vertexOffset);
  v1.pos = getVertex(trianglIndex.y + vertexOffset);
  v2.pos = getVertex(trianglIndex.z + vertexOffset);
  v0.nrm = getNormal(trianglIndex.x + vertexOffset);
  v1.nrm = getNormal(trianglIndex.y + vertexOffset);
  v2.nrm = getNormal(trianglIndex.z + vertexOffset);

  Vertex vtx;
  vtx.pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  vtx.nrm = normalize(v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z);

  // World space
  vtx.pos = vec3(gl_ObjectToWorldEXT * vec4(vtx.pos, 1.0));
  vtx.nrm = normalize(vec3(vtx.nrm * gl_WorldToObjectEXT));

  return vtx;
}


void main()
{
  // gl_InstanceID gives the Instance Info
  // gl_PrimitiveID gives the triangle for this instance

  // Retrieve the vertex information of the triangle
  //------------------------------------------------
  // Getting the 'first index' for this instance (offset of the instance + offset of the triangle)
  uint indexOffset = InstanceInfo[gl_InstanceID].indexOffset + (3 * gl_PrimitiveID);
  // Getting the 3 indices of the triangle
  ivec3 ind = ivec3(IndexBuf[indexOffset + 0], IndexBuf[indexOffset + 1], IndexBuf[indexOffset + 2]);
  // The barycentric of the hit point
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  // Vertex offset as defined in glTF
  uint vertexOffset = InstanceInfo[gl_InstanceID].vertexOffset;
  // Get all interpolated vertex information
  Vertex v = getVertex(ind, vertexOffset, barycentrics);
  //------------------------------------------------

  vec3 origin;  // = v.pos;
  //origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;;
  origin = offsetRay(v.pos, v.nrm);
  //origin = offsetRay(origin, v.nrm);

  Material m = MaterialBuffer.m[InstanceInfo[gl_InstanceID].materialIndex];

  vec3 base_color = m.pbrBaseColorFactor.rgb;
  prd.origin      = origin;
  prd.attenuation = prd.attenuation * base_color / (M_PIf);
  prd.albedo      = base_color;
  prd.normal      = v.nrm;

  if(m.emissiveFactor.r >= 1 || m.emissiveFactor.g >= 1 || m.emissiveFactor.b >= 1)
  {
    prd.radiance = m.emissiveFactor;
    prd.done     = 1;
    return;
  }


  // Sampling the hemisphere (diffuse)
  vec3        tangent, binormal;
  const float z1 = rnd(prd.seed);
  const float z2 = rnd(prd.seed);
  computeOrthonormalBasis(v.nrm, tangent, binormal);
  vec3 p;
  cosine_sample_hemisphere(z1, z2, p);
  inverse_transform(p, v.nrm, tangent, binormal);
  prd.direction = p;  // New sampling direction


  // Shadow trace for lights
  vec3 result = vec3(0, 0, 0);
  for(int i = 0; i < SceneInfo.nbLights; ++i)
  {
    vec3  lightDir       = SceneInfo.lights[i].position.xyz - v.pos;
    float lightDist      = length(lightDir);
    lightDir             = normalize(lightDir);
    float lightIntencity = SceneInfo.lights[i].color.a * 1.f / (lightDist * lightDist);

    float dotNL = max(0.0, dot(v.nrm, lightDir));

    payloadShadow = true;
    float tmin    = 0.0;
    float tmax    = lightDist;
    if(dotNL > 0)
    {
      traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                  0xFF, 1 /* sbtRecordOffset */, 0 /* sbtRecordStride */, 1 /* missIndex */, origin, tmin, lightDir,
                  tmax, 2 /*payload location*/);
    }

    if(payloadShadow)
      lightIntencity = 0.0;

    result += SceneInfo.lights[i].color.rgb * dotNL * lightIntencity;
  }

  prd.radiance = result;
}
