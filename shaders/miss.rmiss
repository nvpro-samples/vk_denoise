#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"
layout(location = 0) rayPayloadInNV PerRayData_pathtrace prd;

void main()
{
  prd.radiance = vec3(0, 0, 0);
  prd.done     = 1;
}
