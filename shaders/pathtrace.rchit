#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
//#extension GL_EXT_nonuniform_qualifier : enable

#include "sampling.h"
#include "share.h"

// Payload information of the ray returning: 0 hit, 2 shadow
layout(location = 0) rayPayloadInNV PerRayData_pathtrace prd;
layout(location = 2) rayPayloadNV bool payloadShadow;

// Raytracing hit attributes: barycentrics
hitAttributeNV vec3 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureNV topLevelAS;

layout(binding = 2, set = 0) uniform _ubo
{
  Scene s;
}
ubo;

// Per Instance information
layout(binding = 3, set = 0) readonly buffer _OffsetIndices
{
  primInfo i[];
}
instanceInfo;
// Vertex position buffer
layout(binding = 4, set = 0) readonly buffer _VertexBuf
{
  float v[];
}
VertexBuf;
// Index buffer
layout(binding = 5, set = 0) readonly buffer _Indices
{
  uint i[];
}
indices;
// Normal buffer
layout(binding = 6, set = 0) readonly buffer _NormalBuf
{
  float v[];
}
NormalBuf;
// Matrices buffer for all instances
layout(binding = 7, set = 0) readonly buffer _MatrixBuffer
{
  InstancesMatrices m[];
}
MatrixBuffer;

// Materials
layout(binding = 8, set = 0) readonly buffer _MaterialBuffer
{
  Material m[];
}
MaterialBuffer;



// Return the vertex position
vec3 getVertex(uint index)
{
  vec3 vp;
  vp.x = VertexBuf.v[3 * index + 0];
  vp.y = VertexBuf.v[3 * index + 1];
  vp.z = VertexBuf.v[3 * index + 2];
  return vp;
}

vec3 getNormal(uint index)
{
  vec3 vp;
  vp.x = NormalBuf.v[3 * index + 0];
  vp.y = NormalBuf.v[3 * index + 1];
  vp.z = NormalBuf.v[3 * index + 2];
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

  // Properly transforming the normal
  vtx.nrm = normalize(vec3(MatrixBuffer.m[gl_InstanceID].worldIT * vec4(vtx.nrm, 0.0)));
  // Properly transforming the vertex
  vtx.pos = vec3(MatrixBuffer.m[gl_InstanceID].world * vec4(vtx.pos, 1.0));

  return vtx;
}


void main()
{
  // gl_InstanceID gives the Instance Info
  // gl_PrimitiveID gives the triangle for this instance

  // Retrieve the vertex information of the triangle
  //------------------------------------------------
  // Getting the 'first index' for this instance (offset of the instance + offset of the triangle)
  uint indexOffset = instanceInfo.i[gl_InstanceID].indexOffset + (3 * gl_PrimitiveID);
  // Getting the 3 indices of the triangle
  ivec3 ind = ivec3(indices.i[indexOffset + 0], indices.i[indexOffset + 1], indices.i[indexOffset + 2]);
  // The barycentric of the hit point
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  // Vertex offset as defined in glTF
  uint vertexOffset = instanceInfo.i[gl_InstanceID].vertexOffset;
  // Get all interpolated vertex information
  Vertex v = getVertex(ind, vertexOffset, barycentrics);
  //------------------------------------------------

  vec3 origin;  // = v.pos;
  //origin = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV;;
  origin = offsetRay(v.pos, v.nrm);
  //origin = offsetRay(origin, v.nrm);

  Material m = MaterialBuffer.m[instanceInfo.i[gl_InstanceID].materialIndex];

  vec3 diffuse_color = m.baseColorFactor.rgb;
  prd.origin         = origin;
  prd.attenuation    = prd.attenuation * diffuse_color / (M_PIf);
  prd.countEmitted   = 0;


  if (m.emissiveFactor.r >= 1 || m.emissiveFactor.g >= 1 || m.emissiveFactor.b >= 1)
  {
    prd.radiance = m.emissiveFactor;
    prd.done = 1;
	return;
  }


  // Sampling the hemisphere (diffuse)
  const float z1 = radinv_fl(prd.seed, 5 + 3 * prd.depth);
  const float z2 = radinv_fl(prd.seed, 6 + 3 * prd.depth);
  vec3        tangent, binormal;
  computeOrthonormalBasis(v.nrm, tangent, binormal);
  vec3 p;
  cosine_sample_hemisphere(z1, z2, p);
  inverse_transform(p, v.nrm, tangent, binormal);
  prd.direction = p; // New sampling direction

  
//  {
//      prd.radiance = abs(p);
//    prd.done = 1;
//	return;
//  }
//

  // Shadow trace for lights
  vec3 result = vec3(0, 0, 0);
  for(int i = 0; i < ubo.s.nbLights; ++i)
  {
    vec3  lightDir       = ubo.s.lights[i].position.xyz - v.pos;
    float lightDist      = length(lightDir);
    lightDir             = normalize(lightDir);
    float lightIntencity = ubo.s.lights[i].color.a * 1.f / (lightDist * lightDist);

    float dotNL = max(0.0, dot(v.nrm, lightDir));

    payloadShadow = true;
    float tmin    = 0.0;
    float tmax    = lightDist;
    //if(dotNL > 0)
    {
      traceNV(topLevelAS, gl_RayFlagsTerminateOnFirstHitNV | gl_RayFlagsOpaqueNV | gl_RayFlagsSkipClosestHitShaderNV,
              0xFF, 1 /* sbtRecordOffset */, 0 /* sbtRecordStride */, 1 /* missIndex */, origin, tmin, lightDir, tmax,
              2 /*payload location*/);
    }

    if(payloadShadow)
      lightIntencity = 0.0;

    result += ubo.s.lights[i].color.rgb * dotNL * lightIntencity;
  }

  prd.radiance = vec3(1, 1, 1) * result;
}
