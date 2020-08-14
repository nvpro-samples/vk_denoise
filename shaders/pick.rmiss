#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout(location = 0) rayPayloadInNV PerRayData_pick prd;

void main()
{
  prd.instanceID = ~0;
}
