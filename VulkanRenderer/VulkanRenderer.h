#pragma once

#include "Camera.hpp"

#define SIZEOF(vect) (static_cast<uint32_t>(vect.size())) 
#define FPTR(vect) (vect.size() > 0 ? vect.data() : nullptr)

#define WIDTH 1200
#define HEIGHT 800
#define M_PI       3.14159265358979323846   // pi
#define MAX_BONES 65
#define MAX_BONES_PER_VERTEX 4
#define KEY_W 0x57
#define KEY_A 0x41
#define KEY_S 0x53
#define KEY_D 0x44
struct WindowParameters {
	HINSTANCE HInstance;
	HWND HWnd;
};
struct QueueInfo {
	uint32_t FamilyIndex;
	std::vector<float> Priorities;
};
struct InstanceData {
	glm::vec4 model_0;
	glm::vec4 model_1;
	glm::vec4 model_2;
	glm::vec4 model_3;
};
struct VertexBoneData
{
	std::array<uint32_t, MAX_BONES_PER_VERTEX> IDs;
	std::array<float, MAX_BONES_PER_VERTEX> weights;

	// Ad bone weighting to vertex info
	void add(uint32_t boneID, float weight)
	{
		for (uint32_t i = 0; i < MAX_BONES_PER_VERTEX; i++)
		{
			if (weights[i] == 0.0f)
			{
				IDs[i] = boneID;
				weights[i] = weight;
				return;
			}
		}
		//assert(0);
	}
};
// Stores information on a single bone
struct BoneInfo
{
	aiMatrix4x4 offset;
	aiMatrix4x4 finalTransformation;

	BoneInfo()
	{
		offset = aiMatrix4x4();
		finalTransformation = aiMatrix4x4();
	};
};
struct VertexMesh {
	glm::vec3 pos;
	glm::vec2 uv;
	glm::vec3 color;
	float boneWeights[MAX_BONES_PER_VERTEX];
	uint32_t boneIDs[MAX_BONES_PER_VERTEX];

	static std::array<VkVertexInputBindingDescription, 2> getBindingDescriptions() {
		std::array < VkVertexInputBindingDescription, 2> bindingDescriptions = {};
		bindingDescriptions[0].binding = 0;
		bindingDescriptions[0].stride = sizeof(VertexMesh);
		bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		bindingDescriptions[1].binding = 1;
		bindingDescriptions[1].stride = sizeof(InstanceData);
		bindingDescriptions[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		return bindingDescriptions;
	}
	static std::array<VkVertexInputAttributeDescription, 9> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 9> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = 0;

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[1].offset = sizeof(float) * 3;

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = sizeof(float) * 5;

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[3].offset = sizeof(float) * 8;

		attributeDescriptions[4].binding = 0;
		attributeDescriptions[4].location = 4;
		attributeDescriptions[4].format = VK_FORMAT_R32G32B32A32_SINT;
		attributeDescriptions[4].offset = sizeof(float) * 12;

		attributeDescriptions[5].binding = 1;
		attributeDescriptions[5].location = 5;
		attributeDescriptions[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[5].offset = 0;

		attributeDescriptions[6].binding = 1;
		attributeDescriptions[6].location = 6;
		attributeDescriptions[6].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[6].offset = sizeof(float) * 4;

		attributeDescriptions[7].binding = 1;
		attributeDescriptions[7].location = 7;
		attributeDescriptions[7].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[7].offset = sizeof(float) * 8;

		attributeDescriptions[8].binding = 1;
		attributeDescriptions[8].location = 8;
		attributeDescriptions[8].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[8].offset = sizeof(float) * 12;

		return attributeDescriptions;
	}

};

struct VertexSky {
	glm::vec3 pos;

	static std::array<VkVertexInputBindingDescription, 1> getBindingDescriptions() {
		std::array < VkVertexInputBindingDescription, 1> bindingDescriptions = {};
		bindingDescriptions[0].binding = 0;
		bindingDescriptions[0].stride = sizeof(VertexSky);
		bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;


		return bindingDescriptions;
	}
	static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = 0;

		return attributeDescriptions;
	}

};


struct BoneDataInstance {
	glm::mat4 bones[MAX_BONES];
};

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> desiredInstanceExtensions =
{
	VK_KHR_SURFACE_EXTENSION_NAME,
	VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
	VK_EXT_DEBUG_UTILS_EXTENSION_NAME
};

const std::vector<const char*> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation",
	"VK_LAYER_RENDERDOC_Capture"
};
std::vector<char const *> desiredDeviceExtensions =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME

};

const std::string MESH_ONE_PATH = "Data\\Models\\Put Back Rifle Behind Shoulder.dae";
const std::string MESH_TWO_PATH = "Data\\Models\\Great Sword Slash.dae";
const std::string MESH_TEXTURE_PATH = "Data\\Textures\\maria_diffuse.png";
const std::string SKY_ONE_PATH = "Data\\Models\\sky.dae";


class MouseStateParameters {
public:
	struct ButtonsState {
		bool IsPressed;
		bool WasClicked;
		bool WasRelease;
	} Buttons[2];
	struct Position {
		int X;
		int Y;
		struct Delta {
			int X;
			int Y;
		} Delta;
	} Position;
	struct WheelState {
		bool  WasMoved;
		float Distance;
	} Wheel;

public:
	MouseStateParameters() {
		Buttons[0].IsPressed = false;
		Buttons[0].WasClicked = false;
		Buttons[0].WasRelease = false;
		Buttons[1].IsPressed = false;
		Buttons[1].WasClicked = false;
		Buttons[1].WasRelease = false;
		Position.X = 0;
		Position.Y = 0;
		Position.Delta.X = 0;
		Position.Delta.Y = 0;
		Wheel.WasMoved = false;
		Wheel.Distance = 0.0f;
	}
};

class SkinnedMesh
{
public:
	// Bone related stuff
	// Maps bone name with index
	std::map<std::string, uint32_t> boneMapping;
	// Bone details
	std::vector<BoneInfo> boneInfo;
	// Number of bones present
	uint32_t numBones = 0;
	// Root inverse transform matrix
	aiMatrix4x4 globalInverseTransform;
	// Per-vertex bone info
	std::vector<VertexBoneData> bones;
	// Bone transformations
	std::vector<aiMatrix4x4> boneTransforms;

	// Currently active animation
	aiAnimation* pAnimation;

	// Store reference to the ASSIMP scene for accessing properties of it during animation
	Assimp::Importer Importer;
	const aiScene* scene;

	// Set active animation by index
	void setAnimation(uint32_t animationIndex)
	{
		assert(animationIndex < scene->mNumAnimations);
		pAnimation = scene->mAnimations[animationIndex];
	}

	// Load bone information from ASSIMP mesh
	void loadBones(const aiMesh* pMesh, uint32_t vertexOffset, std::vector<VertexBoneData>& Bones)
	{
		for (uint32_t i = 0; i < pMesh->mNumBones; i++)
		{
			uint32_t index = 0;

			assert(pMesh->mNumBones <= MAX_BONES);

			std::string name(pMesh->mBones[i]->mName.data);

			if (boneMapping.find(name) == boneMapping.end())
			{
				// Bone not present, add new one
				index = numBones;
				numBones++;
				BoneInfo bone;
				boneInfo.push_back(bone);
				boneInfo[index].offset = pMesh->mBones[i]->mOffsetMatrix;
				boneMapping[name] = index;
			}
			else
			{
				index = boneMapping[name];
			}

			for (uint32_t j = 0; j < pMesh->mBones[i]->mNumWeights; j++)
			{
				uint32_t vertexID = vertexOffset + pMesh->mBones[i]->mWeights[j].mVertexId;
				Bones[vertexID].add(index, pMesh->mBones[i]->mWeights[j].mWeight);
			}
		}
		boneTransforms.resize(numBones);
	}

	// Recursive bone transformation for given animation time
	void update(float time)
	{
		float TicksPerSecond = (float)(scene->mAnimations[0]->mTicksPerSecond != 0 ? scene->mAnimations[0]->mTicksPerSecond : 25.0f);
		float TimeInTicks = time * TicksPerSecond;
		float AnimationTime = fmod(TimeInTicks, (float)scene->mAnimations[0]->mDuration);

		aiMatrix4x4 identity = aiMatrix4x4();
		readNodeHierarchy(AnimationTime, scene->mRootNode, identity);

		for (uint32_t i = 0; i < boneTransforms.size(); i++)
		{
			boneTransforms[i] = boneInfo[i].finalTransformation;
		}
	}


private:
	// Find animation for a given node
	const aiNodeAnim* findNodeAnim(const aiAnimation* animation, const std::string nodeName)
	{
		for (uint32_t i = 0; i < animation->mNumChannels; i++)
		{
			const aiNodeAnim* nodeAnim = animation->mChannels[i];
			if (std::string(nodeAnim->mNodeName.data) == nodeName)
			{
				return nodeAnim;
			}
		}
		return nullptr;
	}

	// Returns a 4x4 matrix with interpolated translation between current and next frame
	aiMatrix4x4 interpolateTranslation(float time, const aiNodeAnim* pNodeAnim)
	{
		aiVector3D translation;

		if (pNodeAnim->mNumPositionKeys == 1)
		{
			translation = pNodeAnim->mPositionKeys[0].mValue;
		}
		else
		{
			uint32_t frameIndex = 0;
			for (uint32_t i = 0; i < pNodeAnim->mNumPositionKeys - 1; i++)
			{
				if (time < (float)pNodeAnim->mPositionKeys[i + 1].mTime)
				{
					frameIndex = i;
					break;
				}
			}

			aiVectorKey currentFrame = pNodeAnim->mPositionKeys[frameIndex];
			aiVectorKey nextFrame = pNodeAnim->mPositionKeys[(frameIndex + 1) % pNodeAnim->mNumPositionKeys];

			float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

			const aiVector3D& start = currentFrame.mValue;
			const aiVector3D& end = nextFrame.mValue;

			translation = (start + delta * (end - start));
		}

		aiMatrix4x4 mat;
		aiMatrix4x4::Translation(translation, mat);
		return mat;
	}

	// Returns a 4x4 matrix with interpolated rotation between current and next frame
	aiMatrix4x4 interpolateRotation(float time, const aiNodeAnim* pNodeAnim)
	{
		aiQuaternion rotation;

		if (pNodeAnim->mNumRotationKeys == 1)
		{
			rotation = pNodeAnim->mRotationKeys[0].mValue;
		}
		else
		{
			uint32_t frameIndex = 0;
			for (uint32_t i = 0; i < pNodeAnim->mNumRotationKeys - 1; i++)
			{
				if (time < (float)pNodeAnim->mRotationKeys[i + 1].mTime)
				{
					frameIndex = i;
					break;
				}
			}

			aiQuatKey currentFrame = pNodeAnim->mRotationKeys[frameIndex];
			aiQuatKey nextFrame = pNodeAnim->mRotationKeys[(frameIndex + 1) % pNodeAnim->mNumRotationKeys];

			float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

			const aiQuaternion& start = currentFrame.mValue;
			const aiQuaternion& end = nextFrame.mValue;

			aiQuaternion::Interpolate(rotation, start, end, delta);
			rotation.Normalize();
		}

		aiMatrix4x4 mat(rotation.GetMatrix());
		return mat;
	}


	// Returns a 4x4 matrix with interpolated scaling between current and next frame
	aiMatrix4x4 interpolateScale(float time, const aiNodeAnim* pNodeAnim)
	{
		aiVector3D scale;

		if (pNodeAnim->mNumScalingKeys == 1)
		{
			scale = pNodeAnim->mScalingKeys[0].mValue;
		}
		else
		{
			uint32_t frameIndex = 0;
			for (uint32_t i = 0; i < pNodeAnim->mNumScalingKeys - 1; i++)
			{
				if (time < (float)pNodeAnim->mScalingKeys[i + 1].mTime)
				{
					frameIndex = i;
					break;
				}
			}

			aiVectorKey currentFrame = pNodeAnim->mScalingKeys[frameIndex];
			aiVectorKey nextFrame = pNodeAnim->mScalingKeys[(frameIndex + 1) % pNodeAnim->mNumScalingKeys];

			float delta = (time - (float)currentFrame.mTime) / (float)(nextFrame.mTime - currentFrame.mTime);

			const aiVector3D& start = currentFrame.mValue;
			const aiVector3D& end = nextFrame.mValue;

			scale = (start + delta * (end - start));
		}

		aiMatrix4x4 mat;
		aiMatrix4x4::Scaling(scale, mat);
		return mat;
	}

	// Get node hierarchy for current animation time
	void readNodeHierarchy(float AnimationTime, const aiNode* pNode, const aiMatrix4x4& ParentTransform)
	{
		std::string NodeName(pNode->mName.data);

		aiMatrix4x4 NodeTransformation(pNode->mTransformation);

		const aiNodeAnim* pNodeAnim = findNodeAnim(pAnimation, NodeName);

		if (pNodeAnim)
		{
			// Get interpolated matrices between current and next frame
			aiMatrix4x4 matScale = interpolateScale(AnimationTime, pNodeAnim);
			aiMatrix4x4 matRotation = interpolateRotation(AnimationTime, pNodeAnim);
			aiMatrix4x4 matTranslation = interpolateTranslation(AnimationTime, pNodeAnim);

			NodeTransformation = matTranslation * matRotation * matScale;
		}

		aiMatrix4x4 GlobalTransformation = ParentTransform * NodeTransformation;

		if (boneMapping.find(NodeName) != boneMapping.end())
		{
			uint32_t BoneIndex = boneMapping[NodeName];
			boneInfo[BoneIndex].finalTransformation = globalInverseTransform * GlobalTransformation * boneInfo[BoneIndex].offset;
		}

		for (uint32_t i = 0; i < pNode->mNumChildren; i++)
		{
			readNodeHierarchy(AnimationTime, pNode->mChildren[i], GlobalTransformation);
		}
	}
};

class VulkanRenderer {

private:
	WindowParameters windowParams;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR presentationSurface;
	VkPhysicalDevice physicalDevice;
	VkPhysicalDeviceFeatures deviceFeatures;
	VkPresentModeKHR desiredPresentationMode = VK_PRESENT_MODE_MAILBOX_KHR;
	uint32_t genericQueueFamilyIndex = -1;
	uint32_t genericQueueIndex = -1;
	VkDevice device;
	VkQueue genericQueue;
	VkSwapchainKHR swapChain;
	uint32_t numberOfSwapChainImages = 0;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkRenderPass renderPass;
	VkCommandPool commandPool;
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	std::vector<VkFence> imagesInFlight;
	size_t currentFrame = 0;
	bool framebufferResized = false;
	MouseStateParameters  mouseState;
	float horizontalAngle = 0.0f;
	std::vector<SkinnedMesh*> skinnedMeshs;
	float runningTime = 0.0f;
	float frameTimer = 1.0f;
	float animationSpeed = 0.75f;
	glm::vec3 rotation = glm::vec3();
	glm::vec3 cameraPos = glm::vec3();
	bool paused = false;
	float rotationSpeed = 1.0f;
	float zoomSpeed = 1.0f;
	float timer = 0.0f;
	float zoom = 0;
	Camera camera;
	glm::vec2 mousePos;
	std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp;
	uint32_t frameCounter = 0;

	struct {
		std::vector<VertexMesh> mesh;
		std::vector<VertexSky> sky;
	} vertices;
	struct {
		std::vector<uint32_t> mesh;
		std::vector<uint32_t> sky;
	} indices;
	struct {
		VkImage mesh;
		VkImage sky;
	} textureImage;
	struct {
		VkDeviceMemory sky;
		VkDeviceMemory mesh;
	} textureImageMemory;
	struct {
		VkImageView mesh;
		VkImageView sky;
	} textureImageView;
	struct {
		VkSampler mesh;
		VkSampler sky;
	} textureSampler;
	struct {
		VkDescriptorSetLayout  mesh;
		VkDescriptorSetLayout sky;
	} descriptorSetLayout;
	struct {
		VkPipelineLayout mesh;
		VkPipelineLayout sky;
	} pipelineLayout;
	struct {
		VkPipeline mesh;
		VkPipeline sky;
	} graphicsPipeline;
	struct {
		VkBuffer mesh;
		VkBuffer sky;
	} vertexBuffer;
	struct {
		VkDeviceMemory mesh;
		VkDeviceMemory sky;
	} vertexBufferMemory;
	struct {
		VkBuffer mesh;
		VkBuffer sky;
	} indexBuffer;
	struct {
		VkDeviceMemory mesh;
		VkDeviceMemory sky;
	} indexBufferMemory;
	struct {
		VkBuffer mesh;
	} instanceBuffer;
	struct {
		VkDeviceMemory mesh;
	} instanceBufferMemory;
	struct
	{
		std::vector<VkBuffer> mesh;
		std::vector<VkBuffer> sky;
	} uniformBuffers;
	struct
	{
		std::vector<VkDeviceMemory> mesh;
		std::vector<VkDeviceMemory> sky;
	}	uniformBuffersMemory;
	struct {
		VkDescriptorPool mesh;
		VkDescriptorPool sky;
	} descriptorPool;
	struct {
		std::vector<VkDescriptorSet> mesh;
		std::vector<VkDescriptorSet> sky;
	} descriptorSets;


public:
	struct {
		bool left = false;
		bool right = false;
		bool middle = false;
	} mouseButtons;

	struct UniformBufferObjectMesh {
		glm::mat4 proj;
		glm::mat4 view;
		//glm::mat4 model[2];
		BoneDataInstance boneDataInstance[2];
	} uboMesh;

	struct UniformBufferObjectSky {
		glm::mat4 proj;
		glm::mat4 view;
	} uboSky;

public:
	VulkanRenderer();

	~VulkanRenderer();

	void HandleMessages(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	void HandleMouseMove(int32_t x, int32_t y);

	void MouseMoved(double x, double y, bool & handled);
	void Run();

private:
	void InitializeWindow(char className[], char title[]);

	void Render();

	void DrawFrame();

	void DestroyVulkan();

	void UnregisterWindow();

	void InitializeVulkan(WindowParameters windowParams, char title[]);

	void PickQueue();

	void CreateLogicalDevice();

	void PickQueueFamilyIndex();

	void PickPhysicalDevice();

	void CreateSurfaceKHR(WindowParameters &windowParams);

	void SetupDebugUtil();

	void CreateInstance(char title[]);

	bool IsExtensionSupported(std::vector<VkExtensionProperties> const & available_extensions, char const * const extension);

	void CreateSwapChain();

	void CreateSwapChainImages();

	void CreateSwapChainImageViews();

	void CreateCommandPool();

	VkCommandBuffer CreateCommandBuffer();

	void BeginCommandBufferRecording(VkCommandBuffer commandBuffer);

	void EndCommandBufferRecording(VkCommandBuffer commandBuffer, bool submitQueue);

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

	void CreateImage(uint32_t width, uint32_t height, VkFormat format, uint32_t layers, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

	void CopyBufferToBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t baseArrayLayer, uint32_t layerCount);

	void LoadMesh(const std::string modelPath, std::vector<VertexMesh>& vertices, std::vector<uint32_t>& indices);

	void LoadSky(const std::string modelPath, std::vector<VertexSky>& vertices, std::vector<uint32_t>& indices);
	void LoadAnimation(const std::string modelPath);

	void CreateTextureImage();

	void CreateTextureImageView();

	VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VkImageViewType viewType, uint32_t layerCount);

	void SetupImageMemoryBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags generatingStages, VkPipelineStageFlags consumingStages, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageAspectFlags imageAspect, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex, VkImage image, uint32_t layerCount);

	void SetupBufferMemoryBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags generatingStages, VkPipelineStageFlags consumingStages, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size);

	uint32_t FindMemoryType(VkMemoryRequirements memory_requirements, VkMemoryPropertyFlags properties);

	void CreateRenderPass();

	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

	void CreateDescriptorSetLayout();

	void CreateGraphicsPipeline();

	VkShaderModule CreateShaderModule(const std::vector<char>& code);

	void CreateDepthResources();

	VkFormat FindDepthFormat();

	void CreateFramebuffers();

	void CreateTextureSampler();

	void CreateVertexBuffer();

	void CreateIndexBuffer();

	void CreateUniformBuffers();

	void CreateDescriptorPool();

	void CreateDescriptorSets();

	void CreateCommandBuffers();

	void CreateSynchronizationObjects();

	void UpdateUniformBuffer(uint32_t currentImage);

	void RecreateSwapChain();

	void CleanupSwapChain();

	void PrepareInstanceData();

	void MouseClick(size_t button_index, bool state);

	void MouseMove(int x, int y);

	void MouseWheel(float distance);

	static VKAPI_ATTR VkBool32 VKAPI_CALL DriverDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
#ifdef _DEBUG
		std::cerr << "validation layer: " << pCallbackData->pMessageIdName << ":" << pCallbackData->pMessage << std::endl;
#endif // _DEBUG
		return VK_FALSE;
	}

	void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
		if (func != nullptr) {
			func(instance, debugMessenger, pAllocator);
		}
	}

	static std::vector<char> ReadFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}
};
