// This function returns the geometric information at hit point
// Note: depends on the buffer layout PrimMeshInfo

#ifndef GETHIT_GLSL
#define GETHIT_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "nvvkhl/shaders/ray_util.glsl"
#include "nvvkhl/shaders/vertex_accessor.h"

precision highp float;

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3  pos;
  vec3  nrm;
  vec3  geonrm;
  vec2  uv;
  vec3  tangent;
  vec3  bitangent;
  float bitangentSign;
};


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
HitState getHitState(RenderPrimitive renderPrim)
{
  HitState hit;

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = getTriangleIndices(renderPrim,gl_PrimitiveID);


  // Triangle info
  const vec3 pos0 = getVertexPosition(renderPrim, triangleIndex.x);
  const vec3 pos1 = getVertexPosition(renderPrim, triangleIndex.y);
  const vec3 pos2 = getVertexPosition(renderPrim, triangleIndex.z);
  const vec3 nrm0 = getVertexNormal(renderPrim, triangleIndex.x);
  const vec3 nrm1 = getVertexNormal(renderPrim, triangleIndex.y);
  const vec3 nrm2 = getVertexNormal(renderPrim, triangleIndex.z);
  const vec2 uv0  = getVertexTexCoord0(renderPrim, triangleIndex.x);
  const vec2 uv1  = getVertexTexCoord0(renderPrim, triangleIndex.y);
  const vec2 uv2  = getVertexTexCoord0(renderPrim, triangleIndex.z);
  const vec4 tng0 = getVertexTangent(renderPrim, triangleIndex.x);
  const vec4 tng1 = getVertexTangent(renderPrim, triangleIndex.y);
  const vec4 tng2 = getVertexTangent(renderPrim, triangleIndex.z);

  // Position
  hit.pos = mixBary(pos0, pos1, pos2, barycentrics);
  hit.pos = pointOffset(hit.pos, pos0, pos1, pos2, nrm0, nrm1, nrm2, barycentrics);  // Shadow offset position - hacking shadow terminator
  hit.pos = vec3(gl_ObjectToWorldEXT * vec4(hit.pos, 1.0));

  // Normal
  hit.nrm    = normalize(mixBary(nrm0, nrm1, nrm2, barycentrics));
  hit.nrm    = normalize(vec3(hit.nrm * gl_WorldToObjectEXT));
  hit.geonrm = normalize(cross(pos1 - pos0, pos2 - pos0));
  hit.geonrm = normalize(vec3(hit.geonrm * gl_WorldToObjectEXT));

  // TexCoord
  hit.uv = mixBary(uv0, uv1, uv2, barycentrics);

  // Tangent - Bitangent
  hit.tangent       = normalize(mixBary(tng0.xyz, tng1.xyz, tng2.xyz, barycentrics));
  hit.tangent       = vec3(gl_ObjectToWorldEXT * vec4(hit.tangent, 0.0));
  hit.tangent       = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
  hit.bitangent     = cross(hit.nrm, hit.tangent) * tng0.w;
  hit.bitangentSign = tng0.w;

  // Adjusting normal
  const vec3 V = -gl_WorldRayDirectionEXT;
  if(dot(hit.geonrm, V) < 0)  // Flip if back facing
    hit.geonrm = -hit.geonrm;

  // If backface
  if(dot(hit.geonrm, hit.nrm) < 0)  // Make Normal and GeoNormal on the same side
  {
    hit.nrm       = -hit.nrm;
    hit.tangent   = -hit.tangent;
    hit.bitangent = -hit.bitangent;
  }

  return hit;
}


#endif
