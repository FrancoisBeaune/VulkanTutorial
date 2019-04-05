#version 460

// TODO: necessary?
// #extension GL_ARB_separate_shader_objects : enable

layout(binding = 2) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoords;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(texture(texSampler, fragTexCoords).rgb * fragColor, 1.0);
}
