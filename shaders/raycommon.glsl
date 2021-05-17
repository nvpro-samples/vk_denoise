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

struct PerRayData_pathtrace
{
  vec3 result;
  vec3 radiance;
  vec3 attenuation;
  vec3 origin;
  vec3 direction;
  vec3 normal;
  vec3 albedo;
  uint seed;
  int  depth;
  int  done;
};

struct PerRayData_pick
{
  vec4 worldPos;
  vec4 barycentrics;
  uint instanceID;
  uint primitiveID;
};


struct Light
{
  vec4 position;
  vec4 color;
  //  float intensity;
  //  float _pad;
};

// Per Instance information
struct primInfo
{
  uint indexOffset;
  uint vertexOffset;
  uint materialIndex;
};

// Matrices buffer for all instances
struct InstancesMatrices
{
  mat4 world;
  mat4 worldIT;
};

struct Scene
{
  mat4  projection;
  mat4  model;
  vec4  camPos;
  int   nbLights;  // w = lightRadiance
  int   _pad1;
  int   _pad2;
  int   _pad3;
  Light lights[10];
};

struct Material
{
  vec4 pbrBaseColorFactor;
  vec3 emissiveFactor;
  int  _pad;
};
