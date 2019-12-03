#version 450
#extension GL_GOOGLE_include_directive : enable

// Tonemapping functions
#include "tonemapping.h"

layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;

layout(push_constant) uniform shaderInformation
{
  int   tonemapper;
  float gamma;
  float exposure;
}
pushc;

void main()
{
  vec2 uv    = outUV;
  vec3 color = texture(noisyTxt, uv).rgb;

  fragColor = vec4(toneMap(color, pushc.tonemapper, pushc.gamma, pushc.exposure), 1.0f);
}
