#version 450

layout( location = 0 ) in vec3 inPosition;

layout( set = 0, binding = 0 ) uniform UniformBuffer {
    mat4 proj;
    mat4 view;
};

layout( location = 0 ) out vec3 vert_texcoord;

void main() {
  gl_Position = proj * view * vec4(inPosition.xyz, 1.0);
  vert_texcoord = inPosition;
  //vert_texcoord.x *= -1.0;
}