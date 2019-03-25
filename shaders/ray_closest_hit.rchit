#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) rayPayloadInNV vec3 hitValue;

void main()
{
    hitValue = vec3(0.2, 0.5, 0.5);
}
