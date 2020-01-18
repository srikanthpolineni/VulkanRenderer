#version 450
#extension GL_ARB_separate_shader_objects : enable

// Vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inColor;
layout (location = 3) in vec4 inBoneWeights;
layout (location = 4) in ivec4 inBoneIDs;

// Instanced attributes
layout (location = 5) in mat4 model;

struct BoneDataInstance {
	mat4 bones[65];
};


layout(binding = 0) uniform UBO {
    mat4 proj;
	mat4 view;
	BoneDataInstance boneDataInstance[2];
} ubo;


layout (location = 0) out vec3 fragColor;
layout (location = 1) out vec2 fragTexCoord;


out gl_PerVertex 
{
	vec4 gl_Position;   
};

void main() {
	

	mat4 boneTransform = ubo.boneDataInstance[gl_InstanceIndex].bones[inBoneIDs[0]] * inBoneWeights[0];
	boneTransform     += ubo.boneDataInstance[gl_InstanceIndex].bones[inBoneIDs[1]] * inBoneWeights[1];
	boneTransform     += ubo.boneDataInstance[gl_InstanceIndex].bones[inBoneIDs[2]] * inBoneWeights[2];
	boneTransform     += ubo.boneDataInstance[gl_InstanceIndex].bones[inBoneIDs[3]] * inBoneWeights[3];	
	
	
    gl_Position = ubo.proj * ubo.view * model * boneTransform * vec4(inPosition.xyz , 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;	
}