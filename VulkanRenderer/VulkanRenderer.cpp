#include <vulkan.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <Windows.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <time.h> 
#include <random>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>
#include <vulkan.h>
#include <assimp/Importer.hpp> 
#include <assimp/scene.h>     
#include <assimp/postprocess.h>
#include <assimp/cimport.h>
#include "VulkanRenderer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

VulkanRenderer* app;

int main()
{
	uint32_t instance_version;
	vkEnumerateInstanceVersion(&instance_version);
	std::cout << "Version. Major:" << VK_VERSION_MAJOR(instance_version) << " Minor:" << VK_VERSION_MINOR(instance_version) << " Patch:" << VK_VERSION_PATCH(instance_version) << std::endl;
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const VkLayerProperties& layerProperties : availableLayers)
	{
		std::cout << layerProperties.layerName << std::endl;
	}

	try {

		app = new VulkanRenderer();
		app->Run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		int input;
		std::cin >> input;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (app != NULL)																		\
	{
		app->HandleMessages(hWnd, uMsg, wParam, lParam);
	}
	return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}

VulkanRenderer::VulkanRenderer()
{
	camera.type = Camera::CameraType::firstperson;
	camera.setPerspective(60.0f, (float)WIDTH / (float)HEIGHT, 0.1f, 1024.0f);
	//camera.setRotation(glm::vec3(-12.0f, 159.0f, 0.0f));
	camera.setTranslation(glm::vec3(0.4f, 1.25f, 0.0f));
	camera.movementSpeed = 5.0f;
}

void VulkanRenderer::Run()
{

	char className[] = "VulkanRenderer";
	char title[] = "Vulkan Renderer";
	InitializeWindow(className, title);
	InitializeVulkan(windowParams, title);
	Render();
	vkDeviceWaitIdle(device);
	DestroyVulkan();
}

VulkanRenderer::~VulkanRenderer()
{
	vkDeviceWaitIdle(device);
	DestroyVulkan();
	UnregisterWindow();
}



void VulkanRenderer::HandleMessages(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_CLOSE:
		PostQuitMessage(0);
		break;
	case WM_PAINT:
		ValidateRect(windowParams.HWnd, NULL);
		break;
	case WM_SIZE:
	case WM_EXITSIZEMOVE:
		framebufferResized = true;
		break;
	case WM_KEYDOWN:
		if (VK_ESCAPE == wParam) {
			PostQuitMessage(0);
			break;
		}
		if (camera.firstperson)
		{
			switch (wParam)
			{
			case KEY_W:
				camera.keys.up = true;
				break;
			case KEY_S:
				camera.keys.down = true;
				break;
			case KEY_A:
				camera.keys.left = true;
				break;
			case KEY_D:
				camera.keys.right = true;
				break;
			}
		}
		break;
	case WM_KEYUP:
		if (camera.firstperson)
		{
			switch (wParam)
			{
			case KEY_W:
				camera.keys.up = false;
				break;
			case KEY_S:
				camera.keys.down = false;
				break;
			case KEY_A:
				camera.keys.left = false;
				break;
			case KEY_D:
				camera.keys.right = false;
				break;
			}
		}
		break;
	case WM_LBUTTONDOWN:
		mousePos = glm::vec2((float)LOWORD(lParam), (float)HIWORD(lParam));
		mouseButtons.left = true;
		break;
	case WM_LBUTTONUP:
		mouseButtons.left = false;
		break;
	case WM_RBUTTONDOWN:
		mousePos = glm::vec2((float)LOWORD(lParam), (float)HIWORD(lParam));
		mouseButtons.right = true;
		break;
	case WM_RBUTTONUP:
		mouseButtons.right = false;
		break;
	case WM_MBUTTONDOWN:
		mousePos = glm::vec2((float)LOWORD(lParam), (float)HIWORD(lParam));
		mouseButtons.middle = true;
		break;
	case WM_MBUTTONUP:
		mouseButtons.middle = false;
		break;
	case WM_MOUSEMOVE:
		HandleMouseMove(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_MOUSEWHEEL:
	{
		short wheelDelta = GET_WHEEL_DELTA_WPARAM(wParam);
		zoom += (float)wheelDelta * 0.005f * zoomSpeed;
		camera.translate(glm::vec3(0.0f, 0.0f, (float)wheelDelta * 0.005f * zoomSpeed));
		break;
	}
	}
}

void VulkanRenderer::HandleMouseMove(int32_t x, int32_t y)
{
	int32_t dx = (int32_t)mousePos.x - x;
	int32_t dy = (int32_t)mousePos.y - y;

	bool handled = false;


	MouseMoved((float)x, (float)y, handled);

	if (handled) {
		mousePos = glm::vec2((float)x, (float)y);
		return;
	}

	if (mouseButtons.left) {
		rotation.x += dy * 1.25f * rotationSpeed;
		rotation.y -= dx * 1.25f * rotationSpeed;
		camera.rotate(glm::vec3(dy * camera.rotationSpeed, -dx * camera.rotationSpeed, 0.0f));
	}
	if (mouseButtons.right) {
		zoom += dy * .005f * zoomSpeed;
		camera.translate(glm::vec3(-0.0f, 0.0f, dy * .005f * zoomSpeed));
	}
	if (mouseButtons.middle) {
		cameraPos.x -= dx * 0.01f;
		cameraPos.y -= dy * 0.01f;
		camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
	}
	mousePos = glm::vec2((float)x, (float)y);
}

void VulkanRenderer::MouseMoved(double x, double y, bool & handled) {}

void VulkanRenderer::InitializeWindow(char className[], char title[])
{
	windowParams.HInstance = GetModuleHandle(nullptr);
	WNDCLASSEX window_class = {
		sizeof(WNDCLASSEX),
		CS_HREDRAW | CS_VREDRAW,
		WndProc,
		0,
		0,
		windowParams.HInstance,
		LoadIcon(NULL, IDI_APPLICATION),
		LoadCursor(NULL, IDC_ARROW),
		(HBRUSH)GetStockObject(BLACK_BRUSH),
		NULL,
		className,
		LoadIcon(NULL, IDI_WINLOGO)
	};

	if (!RegisterClassEx(&window_class)) {
		std::cerr << "Window RegisterClassEX failed." << std::endl;
		throw std::runtime_error("Window RegisterClassEX failed.");
	}
	windowParams.HWnd = CreateWindowEx(WS_EX_CLIENTEDGE, className, title, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, WIDTH, HEIGHT, nullptr, nullptr, windowParams.HInstance, nullptr);
	if (windowParams.HWnd == nullptr) {
		std::cerr << "Window CreateWindowEx failed." << std::endl;
		throw std::runtime_error("Window CreateWindowEx failed.");
	}

	ShowWindow(windowParams.HWnd, SW_SHOWNORMAL);
	UpdateWindow(windowParams.HWnd);
}

void VulkanRenderer::Render()
{
	bool loop = true;
	MSG msg;
	while (loop) {
		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
			if (msg.message == WM_QUIT) {
				loop = false;
				break;
			}
		}
		if (!IsIconic(windowParams.HWnd)) {
			auto tStart = std::chrono::high_resolution_clock::now();
			DrawFrame();
			runningTime += frameTimer * animationSpeed;
			frameCounter++;
			auto tEnd = std::chrono::high_resolution_clock::now();
			auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
			frameTimer = (float)tDiff / 1000.0f;
			camera.update(frameTimer);
			float fpsTimer = (float)(std::chrono::duration<double, std::milli>(tEnd - lastTimestamp).count());
			if (fpsTimer > 1000.0f)
			{
				frameCounter = 0;
				lastTimestamp = tEnd;
			}
		}
	}
}

void VulkanRenderer::DrawFrame()
{
	vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		RecreateSwapChain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	UpdateUniformBuffer(imageIndex);

	if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
		vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
	}
	imagesInFlight[imageIndex] = inFlightFences[currentFrame];

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

	VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	vkResetFences(device, 1, &inFlightFences[currentFrame]);

	result = vkQueueSubmit(genericQueue, 1, &submitInfo, inFlightFences[currentFrame]);
	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to submit draw command buffer!");
	}

	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { swapChain };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;

	presentInfo.pImageIndices = &imageIndex;

	result = vkQueuePresentKHR(genericQueue, &presentInfo);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
		framebufferResized = false;
		RecreateSwapChain();
	}
	else if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to present swap chain image!");
	}

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanRenderer::DestroyVulkan()
{
	CleanupSwapChain();

	vkDestroySampler(device, textureSampler.mesh, nullptr);
	vkDestroySampler(device, textureSampler.sky, nullptr);
	vkDestroyImageView(device, textureImageView.mesh, nullptr);
	vkDestroyImageView(device, textureImageView.sky, nullptr);

	vkDestroyImage(device, textureImage.mesh, nullptr);
	vkDestroyImage(device, textureImage.sky, nullptr);
	vkFreeMemory(device, textureImageMemory.mesh, nullptr);
	vkFreeMemory(device, textureImageMemory.sky, nullptr);

	vkDestroyDescriptorSetLayout(device, descriptorSetLayout.mesh, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout.sky, nullptr);

	vkDestroyBuffer(device, indexBuffer.mesh, nullptr);
	vkDestroyBuffer(device, indexBuffer.sky, nullptr);
	vkFreeMemory(device, indexBufferMemory.mesh, nullptr);
	vkFreeMemory(device, indexBufferMemory.sky, nullptr);
	vkDestroyBuffer(device, vertexBuffer.mesh, nullptr);
	vkDestroyBuffer(device, vertexBuffer.sky, nullptr);
	vkFreeMemory(device, vertexBufferMemory.mesh, nullptr);
	vkFreeMemory(device, vertexBufferMemory.sky, nullptr);
	vkDestroyBuffer(device, instanceBuffer.mesh, nullptr);
	vkFreeMemory(device, instanceBufferMemory.mesh, nullptr);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
		vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
		vkDestroyFence(device, inFlightFences[i], nullptr);
	}

	vkDestroyCommandPool(device, commandPool, nullptr);

	vkDestroyDevice(device, nullptr);

	DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);


	vkDestroySurfaceKHR(instance, presentationSurface, nullptr);
	vkDestroyInstance(instance, nullptr);
}

void VulkanRenderer::UnregisterWindow()
{
	if (windowParams.HWnd) {
		DestroyWindow(windowParams.HWnd);
	}

	if (windowParams.HInstance) {
		UnregisterClass("VulkanRenderer", windowParams.HInstance);
	}
}

void VulkanRenderer::InitializeVulkan(WindowParameters windowParams, char title[])
{
	CreateInstance(title);
	SetupDebugUtil();
	CreateSurfaceKHR(windowParams);
	PickPhysicalDevice();
	PickQueueFamilyIndex();
	CreateLogicalDevice();
	PickQueue();
	CreateSwapChain();
	CreateRenderPass();
	CreateDepthResources();
	CreateFramebuffers();
	CreateCommandPool();

	CreateDescriptorSetLayout();
	CreateGraphicsPipeline();
	CreateTextureImage();
	CreateTextureImageView();
	CreateTextureSampler();
	LoadAnimation(MESH_ONE_PATH);
	LoadAnimation(MESH_TWO_PATH);
	LoadMesh(MESH_ONE_PATH, vertices.mesh, indices.mesh);
	LoadSky(SKY_ONE_PATH, vertices.sky, indices.sky);
	CreateVertexBuffer();
	CreateIndexBuffer();
	PrepareInstanceData();
	CreateUniformBuffers();
	CreateDescriptorPool();
	CreateDescriptorSets();

	CreateCommandBuffers();
	CreateSynchronizationObjects();
}

void VulkanRenderer::PickQueue()
{
	//Get GRAPHIC[TRNASFER|PRESENTATION] queue. 
	vkGetDeviceQueue(device, genericQueueFamilyIndex, genericQueueIndex, &genericQueue);
	if (VK_NULL_HANDLE == genericQueue)
	{
		std::cerr << "Could not get the Queue." << std::endl;
		throw std::runtime_error("Could not get the Queue.");
	}
}

void VulkanRenderer::CreateLogicalDevice()
{
	std::vector<QueueInfo> queue_infos = { { genericQueueFamilyIndex,{ 1.0f } } };
	std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
	for (auto & info : queue_infos) {
		queue_create_infos.push_back(
			{
				VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				nullptr,
				0,
				info.FamilyIndex,
				SIZEOF(info.Priorities),
				FPTR(info.Priorities)
			}
		);
	};

	VkDeviceCreateInfo device_create_info = {
		VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		nullptr,
		0,
		static_cast<uint32_t>(queue_create_infos.size()),
		queue_create_infos.data(),
		static_cast<uint32_t>(validationLayers.size()),
		validationLayers.data(),
		SIZEOF(desiredDeviceExtensions),
		FPTR(desiredDeviceExtensions),
		&deviceFeatures
	};

	if (VK_SUCCESS != vkCreateDevice(physicalDevice, &device_create_info, nullptr, &device) || (device == VK_NULL_HANDLE))
	{
		std::cerr << "Could not create logical device." << std::endl;
		throw std::runtime_error("Could not create logical device.");
	}
}

void VulkanRenderer::PickQueueFamilyIndex()
{
	//Get GRAPHIC[TRNASFER|PRESENTATION] queue family index.
	uint32_t queue_family_count = 0;
	std::vector< VkQueueFamilyProperties> queue_families;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_count, nullptr);
	queue_families.resize(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_count, &queue_families[0]);

	for (uint32_t index = 0; index < static_cast<uint32_t>(queue_families.size()); ++index) {
		VkBool32 presentation_supported = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, index, presentationSurface, &presentation_supported);
		if ((queue_families[index].queueCount > 0) && presentation_supported && (queue_families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			genericQueueFamilyIndex = index;
			genericQueueIndex = 0;
			break;
		}

	}
	if (genericQueueFamilyIndex == -1 || genericQueueIndex == -1) {
		std::cerr << "Could not get required queue family." << std::endl;
		throw std::runtime_error("Could not get required queue family.");
	}
}

void VulkanRenderer::PickPhysicalDevice()
{
	uint32_t devices_count = 0;
	std::vector<VkPhysicalDevice> physical_devices;
	vkEnumeratePhysicalDevices(instance, &devices_count, nullptr);
	physical_devices.resize(devices_count);
	vkEnumeratePhysicalDevices(instance, &devices_count, &physical_devices[0]);
	if (devices_count == 0) {
		std::cerr << "Could not enumerate physical devices." << std::endl;
		throw std::runtime_error("Could not enumerate physical devices.");
	}

	for (auto& physical_device : physical_devices)
	{
		VkPhysicalDeviceFeatures device_features;
		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceFeatures(physical_device, &device_features);
		vkGetPhysicalDeviceProperties(physical_device, &device_properties);

		if (!device_features.geometryShader || device_properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			continue;
		}

		//Whether physical device supports required extensions
		uint32_t extensions_count = 0;
		std::vector< VkExtensionProperties> available_extensions;
		vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, nullptr);
		available_extensions.resize(extensions_count);
		vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, &available_extensions[0]);
		for (auto& desired_extension : desiredDeviceExtensions)
		{
			if (!IsExtensionSupported(available_extensions, desired_extension)) {
				//std::cout << "Extension named '" << desired_extension << "' is not supported." << std::endl;
				continue;
			}
		}

		// Whether physical device queues supports presentation or not.
		uint32_t queue_family_count = 0;
		std::vector< VkQueueFamilyProperties> queue_families;
		vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
		queue_families.resize(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, &queue_families[0]);
		for (uint32_t index = 0; index < static_cast<uint32_t>(queue_families.size()); ++index) {
			VkBool32 presentation_supported = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, index, presentationSurface, &presentation_supported);
			if (!presentation_supported)
				continue;
		}

		//Whether physical device supports Mailbox presentation mode or not.
		uint32_t present_modes_count = 0;
		std::vector<VkPresentModeKHR> present_modes;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, presentationSurface, &present_modes_count, nullptr);
		present_modes.resize(present_modes_count);
		vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, presentationSurface, &present_modes_count, &present_modes[0]);
		if (std::find(present_modes.begin(), present_modes.end(), desiredPresentationMode) == present_modes.end())
		{
			//std::cout << "Mailbox presentation mode not supported" << std::endl;
			continue;
		}

		physicalDevice = physical_device;
		deviceFeatures = {};
		deviceFeatures.geometryShader = VK_TRUE;
		deviceFeatures.samplerAnisotropy = VK_TRUE;
	}
	if (physicalDevice == VK_NULL_HANDLE) {
		std::cerr << "Couldn't find compatible physical device." << std::endl;
		throw std::runtime_error("Couldn't find compatible physical device.");
	}
}

void VulkanRenderer::CreateSurfaceKHR(WindowParameters &windowParams)
{
	VkWin32SurfaceCreateInfoKHR surface_create_info = {
		VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
		nullptr,
		0,
		windowParams.HInstance,
		windowParams.HWnd
	};
	if (VK_SUCCESS != vkCreateWin32SurfaceKHR(instance, &surface_create_info, nullptr, &presentationSurface))
	{
		std::cerr << "Could not create presentation surface." << std::endl;
		throw std::runtime_error("Could not create presentation surface.");
	}
}

void VulkanRenderer::SetupDebugUtil()
{
	VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {};
	debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	debug_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	debug_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	debug_create_info.pfnUserCallback = DriverDebugCallback;
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		if (func(instance, &debug_create_info, nullptr, &debugMessenger) != VK_SUCCESS) {
			std::cerr << "failed to set up debug messenger!" << std::endl;
			throw std::runtime_error("Failed to set up debug messenger");
		}
	}
	else {
		std::cerr << "failed to set up debug messenger!" << std::endl;
		throw std::runtime_error("Failed to set up debug messenger");
	}
}

void VulkanRenderer::CreateInstance(char title[])
{
	uint32_t extension_count = 0;
	std::vector<VkExtensionProperties> available_extensions;
	vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
	available_extensions.resize(extension_count);
	vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, &available_extensions[0]);
	for (auto& desired_extension : desiredInstanceExtensions) {
		if (!IsExtensionSupported(available_extensions, desired_extension)) {
			std::cerr << "Extension named '" << desired_extension << "' is not supported.";
			throw std::runtime_error("Extension not supported.");
		}
	}

	VkApplicationInfo application_info = {
		VK_STRUCTURE_TYPE_APPLICATION_INFO,
		nullptr,
		title,
		VK_MAKE_VERSION(1,0,0),
		title,
		VK_MAKE_VERSION(1,0,0),
		VK_MAKE_VERSION(1,0,0)
	};

	VkInstanceCreateInfo instance_create_info = {
		VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		nullptr,
		0,
		&application_info,
		SIZEOF(validationLayers),
		FPTR(validationLayers),
		SIZEOF(desiredInstanceExtensions),
		FPTR(desiredInstanceExtensions)
	};
	VkResult result = vkCreateInstance(&instance_create_info, nullptr, &instance);
	if (VK_SUCCESS != result) {
		throw std::runtime_error("Could not create Vulkan Instance.");
	}
}

bool VulkanRenderer::IsExtensionSupported(std::vector<VkExtensionProperties> const & available_extensions, char const * const extension)
{
	for (auto & available_extension : available_extensions) {
		if (strstr(available_extension.extensionName, extension)) {
			return true;
		}
	}
	return false;
}

void VulkanRenderer::CreateSwapChain()
{
	VkSurfaceCapabilitiesKHR surface_capabilities;
	if (VK_SUCCESS != vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, presentationSurface, &surface_capabilities))
	{
		std::cerr << "Couldn't get capabilities for the presentation surface." << std::endl;
		throw std::runtime_error("Couldn't get capabilities for the presentation surface.");
	}

	numberOfSwapChainImages = surface_capabilities.minImageCount + 1;
	if (surface_capabilities.minImageCount > 0 && surface_capabilities.maxImageCount < numberOfSwapChainImages) {
		numberOfSwapChainImages = surface_capabilities.maxImageCount;
	}
	VkExtent2D size_of_images;
	if (surface_capabilities.currentExtent.width == 0xFFFFFFFF)
	{
		size_of_images = { WIDTH,HEIGHT };
		if (size_of_images.width < surface_capabilities.minImageExtent.width) {
			size_of_images.width = surface_capabilities.minImageExtent.width;
		}
		else if (size_of_images.width > surface_capabilities.maxImageExtent.width) {
			size_of_images.width = surface_capabilities.maxImageExtent.width;
		}

		if (size_of_images.height < surface_capabilities.minImageExtent.height) {
			size_of_images.height = surface_capabilities.minImageExtent.height;
		}
		else if (size_of_images.height > surface_capabilities.maxImageExtent.height) {
			size_of_images.height = surface_capabilities.maxImageExtent.height;
		}
	}
	else {
		size_of_images = surface_capabilities.currentExtent;
	}
	VkImageUsageFlags desired_usages = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	VkImageUsageFlags image_usage = desired_usages & surface_capabilities.supportedUsageFlags;

	VkSurfaceTransformFlagBitsKHR desired_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, surface_transform;
	if (surface_capabilities.supportedTransforms & desired_transform) {
		surface_transform = desired_transform;
	}
	else {
		surface_transform = surface_capabilities.currentTransform;
	}

	VkSurfaceFormatKHR desired_surface_format = { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	uint32_t format_count = 0;
	std::vector<VkSurfaceFormatKHR> surface_formats;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, presentationSurface, &format_count, nullptr);
	surface_formats.resize(format_count);
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, presentationSurface, &format_count, &surface_formats[0]);
	VkFormat image_format;
	VkColorSpaceKHR image_color_space;
	bool is_format_color_set = false;
	if (1 == surface_formats.size() && VK_FORMAT_UNDEFINED == surface_formats[0].format) {
		image_format = desired_surface_format.format;
		image_color_space = desired_surface_format.colorSpace;
	}
	else {
		for (auto & surface_format : surface_formats) {
			if (desired_surface_format.format == surface_format.format && desired_surface_format.colorSpace == surface_format.colorSpace) {
				image_format = desired_surface_format.format;
				image_color_space = desired_surface_format.colorSpace;
				is_format_color_set = true;
			}
		}
	}
	if (!is_format_color_set) {
		for (auto & surface_format : surface_formats) {
			if (desired_surface_format.format == surface_format.format) {
				image_format = desired_surface_format.format;
				image_color_space = surface_format.colorSpace;
				is_format_color_set = true;
			}
		}
	}
	if (!is_format_color_set) {
		image_format = surface_formats[0].format;
		image_color_space = surface_formats[0].colorSpace;
	}
	VkSwapchainKHR oldSwapchain = {};
	VkSwapchainCreateInfoKHR swapchain_create_info = {
	VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
	nullptr,
	0,
	presentationSurface,
	numberOfSwapChainImages,
	image_format,
	image_color_space,
	size_of_images,
	1,
	image_usage,
	VK_SHARING_MODE_EXCLUSIVE,
	genericQueueFamilyIndex,
	nullptr,
	surface_transform,
	VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
	desiredPresentationMode,
	VK_TRUE,
	oldSwapchain
	};
	if (VK_SUCCESS != vkCreateSwapchainKHR(device, &swapchain_create_info, nullptr, &swapChain)) {
		std::cerr << "Could not create a swapchain." << std::endl;
		throw std::runtime_error("Could not create a swapchain.");
	}
	CreateSwapChainImages();
	swapChainImageFormat = image_format;
	swapChainExtent = size_of_images;
	CreateSwapChainImageViews();
}

void VulkanRenderer::CreateSwapChainImages()
{
	uint32_t swapchain_image_count = 0;
	vkGetSwapchainImagesKHR(device, swapChain, &swapchain_image_count, nullptr);
	swapChainImages.resize(swapchain_image_count);
	if (VK_SUCCESS != vkGetSwapchainImagesKHR(device, swapChain, &swapchain_image_count, &swapChainImages[0]) || swapchain_image_count == 0)
	{
		std::cerr << "Could not get the swap chain images" << std::endl;
		throw std::runtime_error("Could not get the swap chain images");
	}
}

void VulkanRenderer::CreateSwapChainImageViews()
{
	swapChainImageViews.resize(swapChainImages.size());

	for (uint32_t i = 0; i < swapChainImages.size(); i++) {
		swapChainImageViews[i] = CreateImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 1);
	}
}

void VulkanRenderer::CreateCommandPool()
{
	VkCommandPoolCreateInfo commandpool_create_info = {
	VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	nullptr,
	VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
	genericQueueFamilyIndex
	};
	if (VK_SUCCESS != vkCreateCommandPool(device, &commandpool_create_info, nullptr, &commandPool))
	{
		std::cerr << "Could not create command pool." << std::endl;
		throw std::runtime_error("Could not create command pool.");
	}
}

VkCommandBuffer VulkanRenderer::CreateCommandBuffer()
{
	VkCommandBufferAllocateInfo command_buffer_allocation_info = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	nullptr,
	commandPool,
	VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	1
	};
	VkCommandBuffer commandBuffer;
	if (VK_SUCCESS != vkAllocateCommandBuffers(device, &command_buffer_allocation_info, &commandBuffer))
	{
		std::cerr << "Could not create command buffer(s)." << std::endl;
		throw std::runtime_error("Could not create command buffer(s)");
	}
	return commandBuffer;
}

void VulkanRenderer::BeginCommandBufferRecording(VkCommandBuffer commandBuffer) {

	VkCommandBufferUsageFlags usage = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VkCommandBufferBeginInfo command_buffer_begin_info = {
	VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	nullptr,
	usage,
	nullptr
	};
	if (VK_SUCCESS != vkBeginCommandBuffer(commandBuffer, &command_buffer_begin_info))
	{
		std::cerr << "Could not begin command buffer recording operation." << std::endl;
		throw std::runtime_error("Could not begin command buffer recording operation.");
	}
}

void VulkanRenderer::EndCommandBufferRecording(VkCommandBuffer commandBuffer, bool submitQueue)
{
	if (VK_SUCCESS != vkEndCommandBuffer(commandBuffer)) {
		std::cerr << "Error occurred during command buffer recording." << std::endl;
		throw std::runtime_error("Error occurred during command buffer recording.");
	}
	if (submitQueue) {
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(genericQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(genericQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

	}
}

void VulkanRenderer::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
	VkBufferCreateInfo buffer_create_info = {
		VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		nullptr,
		0,
		size,
		usage,
		VK_SHARING_MODE_EXCLUSIVE,
		genericQueueFamilyIndex,
		nullptr
	};
	if (VK_SUCCESS != vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer)) {
		std::cerr << "Could not create a buffer." << std::endl;
		throw std::runtime_error("Could not create a buffer.");
	}

	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

	VkMemoryAllocateInfo buffer_memory_allocate_info = {
	VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	nullptr,
	memory_requirements.size,
	FindMemoryType(memory_requirements, properties)
	};
	if (VK_SUCCESS != vkAllocateMemory(device, &buffer_memory_allocate_info, nullptr, &bufferMemory)) {
		throw std::runtime_error("failed to allocate buffer memory!");
	}
	vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void VulkanRenderer::CreateImage(uint32_t width, uint32_t height, VkFormat format, uint32_t layers, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = layers;
	imageInfo.format = format;
	imageInfo.tiling = tiling;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usage;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (layers == 6)
	{
		imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	}
	if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image!");
	}

	VkMemoryRequirements memory_requirements;
	vkGetImageMemoryRequirements(device, image, &memory_requirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memory_requirements.size;
	allocInfo.memoryTypeIndex = FindMemoryType(memory_requirements, properties);

	if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate image memory!");
	}
	vkBindImageMemory(device, image, imageMemory, 0);
}

void VulkanRenderer::CopyBufferToBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
	VkCommandBuffer commandBuffer = CreateCommandBuffer();
	BeginCommandBufferRecording(commandBuffer);

	SetupBufferMemoryBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, dstBuffer, 0, VK_WHOLE_SIZE);

	VkBufferCopy copyRegion = {};
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	SetupBufferMemoryBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, dstBuffer, 0, VK_WHOLE_SIZE);

	EndCommandBufferRecording(commandBuffer, true);
}

void VulkanRenderer::CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t baseArrayLayer, uint32_t layerCount)
{
	VkCommandBuffer commandBuffer = CreateCommandBuffer();
	BeginCommandBufferRecording(commandBuffer);

	SetupImageMemoryBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, layerCount);

	VkBufferImageCopy region = {};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = baseArrayLayer;
	region.imageSubresource.layerCount = layerCount;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = { width, height, 1 };
	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	SetupImageMemoryBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, layerCount);

	EndCommandBufferRecording(commandBuffer, true);
}

void VulkanRenderer::LoadMesh(const std::string modelPath, std::vector<VertexMesh>& vertices, std::vector<uint32_t>& indices)
{
	const int defaultFlags = aiProcess_Triangulate | aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals | aiProcess_FlipUVs;
	auto* skinnedMesh = new SkinnedMesh();
	skinnedMesh->scene = skinnedMesh->Importer.ReadFile(modelPath.c_str(), defaultFlags);
	uint32_t vertexBase = 0;
	std::vector<VertexBoneData> boneData = skinnedMeshs[0]->bones;
	for (unsigned int i = 0; i < skinnedMesh->scene->mNumMeshes; i++)
	{
		for (unsigned int v = 0; v < skinnedMesh->scene->mMeshes[i]->mNumVertices; v++)
		{
			VertexMesh vertex;
			vertex.pos = glm::make_vec3(&skinnedMesh->scene->mMeshes[i]->mVertices[v].x);
			vertex.uv = glm::make_vec2(&skinnedMesh->scene->mMeshes[i]->mTextureCoords[0][v].x);
			vertex.color = (skinnedMesh->scene->mMeshes[i]->HasVertexColors(0)) ? glm::make_vec3(&skinnedMesh->scene->mMeshes[i]->mColors[0][v].r) : glm::vec3(1.0f);

			for (uint32_t j = 0; j < MAX_BONES_PER_VERTEX; j++) {
				vertex.boneWeights[j] = boneData[vertexBase + v].weights[j];
				vertex.boneIDs[j] = boneData[vertexBase + v].IDs[j];

			}
			vertices.push_back(vertex);
		}
		vertexBase += skinnedMesh->scene->mMeshes[i]->mNumVertices;
		uint32_t indexBase = static_cast<uint32_t>(indices.size());
		for (unsigned int j = 0; j < skinnedMesh->scene->mMeshes[i]->mNumFaces; j++)
		{
			const aiFace& Face = skinnedMesh->scene->mMeshes[i]->mFaces[j];
			if (Face.mNumIndices != 3)
				continue;
			indices.push_back(Face.mIndices[0] + indexBase);
			indices.push_back(Face.mIndices[1] + indexBase);
			indices.push_back(Face.mIndices[2] + indexBase);
		}
	}
}


void VulkanRenderer::LoadSky(const std::string modelPath, std::vector<VertexSky>& vertices, std::vector<uint32_t>& indices)
{
	const int defaultFlags = aiProcess_Triangulate | aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals | aiProcess_FlipUVs;
	auto* skinnedMesh = new SkinnedMesh();
	skinnedMesh->scene = skinnedMesh->Importer.ReadFile(modelPath.c_str(), defaultFlags);
	uint32_t vertexBase = 0;
	for (unsigned int i = 0; i < skinnedMesh->scene->mNumMeshes; i++)
	{
		for (unsigned int v = 0; v < skinnedMesh->scene->mMeshes[i]->mNumVertices; v++)
		{
			VertexSky vertex;
			vertex.pos = glm::vec3(
				skinnedMesh->scene->mMeshes[i]->mVertices[v].x * 512.0f,
				skinnedMesh->scene->mMeshes[i]->mVertices[v].y * 512.0f,
				skinnedMesh->scene->mMeshes[i]->mVertices[v].z * 512.0f
			);
			vertices.push_back(vertex);
		}
		vertexBase += skinnedMesh->scene->mMeshes[i]->mNumVertices;
		uint32_t indexBase = static_cast<uint32_t>(indices.size());
		for (unsigned int j = 0; j < skinnedMesh->scene->mMeshes[i]->mNumFaces; j++)
		{
			const aiFace& Face = skinnedMesh->scene->mMeshes[i]->mFaces[j];
			if (Face.mNumIndices != 3)
				continue;
			indices.push_back(Face.mIndices[0] + indexBase);
			indices.push_back(Face.mIndices[1] + indexBase);
			indices.push_back(Face.mIndices[2] + indexBase);
		}
	}
}



void VulkanRenderer::LoadAnimation(const std::string modelPath)
{
	const int defaultFlags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs;
	auto* skinnedMesh = new SkinnedMesh();
	skinnedMesh->scene = skinnedMesh->Importer.ReadFile(modelPath.c_str(), defaultFlags);

	skinnedMesh->setAnimation(0);

	uint32_t vertexCount(0);
	for (uint32_t m = 0; m < skinnedMesh->scene->mNumMeshes; m++) {
		vertexCount += skinnedMesh->scene->mMeshes[m]->mNumVertices;
	};
	skinnedMesh->bones.resize(vertexCount);
	// Store global inverse transform matrix of root node 
	skinnedMesh->globalInverseTransform = skinnedMesh->scene->mRootNode->mTransformation;
	skinnedMesh->globalInverseTransform.Inverse();
	// Load bones (weights and IDs)
	uint32_t vertexBase(0);
	for (uint32_t m = 0; m < skinnedMesh->scene->mNumMeshes; m++) {
		aiMesh *paiMesh = skinnedMesh->scene->mMeshes[m];
		if (paiMesh->mNumBones > 0) {
			skinnedMesh->loadBones(paiMesh, vertexBase, skinnedMesh->bones);
		}
		vertexBase += skinnedMesh->scene->mMeshes[m]->mNumVertices;
	}

	skinnedMeshs.push_back(skinnedMesh);
}

void VulkanRenderer::CreateTextureImage()
{
	{
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(MESH_TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
		}
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels);

		CreateImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage.mesh, textureImageMemory.mesh);

		CopyBufferToImage(stagingBuffer, textureImage.mesh, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 0, 1);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	{

		CreateImage(1024, 1024, VK_FORMAT_R8G8B8A8_UNORM, 6, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage.sky, textureImageMemory.sky);

		std::vector<std::string> cubemap_images = {
			  "Data\\Textures\\miramar_ft.tga",
			  "Data\\Textures\\miramar_bk.tga",
			  "Data\\Textures\\miramar_up.tga",
			  "Data\\Textures\\miramar_dn.tga",
			  "Data\\Textures\\miramar_rt.tga",
			  "Data\\Textures\\miramar_lf.tga"


		};
		unsigned char *textureData[6];
		int texWidth, texHeight, texChannels;
		for (size_t i = 0; i < cubemap_images.size(); i++)
		{

			unsigned char *image = stbi_load(cubemap_images[i].c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
			if (!image) {
				throw std::runtime_error("failed to load texture image!");
			}
			textureData[i] = image;
		}
		VkDeviceSize bufferSize = texWidth * texHeight * 4 * 6;
		VkDeviceSize layerSize = bufferSize / 6;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, stagingBuffer, stagingBufferMemory);
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		for (size_t i = 0; i < 6; ++i)
		{
			memcpy(static_cast<byte*>(data) + (layerSize * i), textureData[i], static_cast<size_t>(layerSize));
		}

		vkUnmapMemory(device, stagingBufferMemory);
		CopyBufferToImage(stagingBuffer, textureImage.sky, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 0, 6);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

	}

}

void VulkanRenderer::CreateTextureImageView()
{
	textureImageView.mesh = CreateImageView(textureImage.mesh, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 1);
	textureImageView.sky = CreateImageView(textureImage.sky, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_CUBE, 6);
}

VkImageView VulkanRenderer::CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VkImageViewType viewType, uint32_t layerCount) {
	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = viewType;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = aspectFlags;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = layerCount;
	viewInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };

	VkImageView imageView;
	if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture image view!");
	}

	return imageView;
}

void VulkanRenderer::SetupImageMemoryBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags generatingStages, VkPipelineStageFlags consumingStages, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageAspectFlags imageAspect, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex, VkImage image, uint32_t layerCount)
{
	VkImageMemoryBarrier imageMemoryBarrier = {
		VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		nullptr,
		srcAccessMask,
		dstAccessMask,
		oldLayout,
		newLayout,
		srcQueueFamilyIndex,
		dstQueueFamilyIndex,
		image,
		{
			imageAspect,
			0,
			1,
			0,
			layerCount
		}
	};

	vkCmdPipelineBarrier(commandBuffer, generatingStages, consumingStages, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
}

void VulkanRenderer::SetupBufferMemoryBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags generatingStages, VkPipelineStageFlags consumingStages, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
	VkBufferMemoryBarrier bufferMemoryBarrier = {
	VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
	nullptr,
	srcAccessMask,
	dstAccessMask,
	srcQueueFamilyIndex,
	dstQueueFamilyIndex,
	buffer,
	offset,
	size
	};
	vkCmdPipelineBarrier(commandBuffer, generatingStages, consumingStages, 0, 0, nullptr, 1, &bufferMemoryBarrier, 0, nullptr);
}

uint32_t VulkanRenderer::FindMemoryType(VkMemoryRequirements memory_requirements, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physical_device_memory_properties);
	for (uint32_t type = 0; type < physical_device_memory_properties.memoryTypeCount; ++type)
	{
		if ((memory_requirements.memoryTypeBits & (1 << type)) && ((physical_device_memory_properties.memoryTypes[type].propertyFlags & properties) == properties)) {
			return type;
		}
	}
	throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanRenderer::CreateRenderPass()
{
	VkAttachmentDescription colorAttachment = {
		0,
		swapChainImageFormat,
		VK_SAMPLE_COUNT_1_BIT,
		VK_ATTACHMENT_LOAD_OP_CLEAR,
		VK_ATTACHMENT_STORE_OP_STORE,
		VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		VK_ATTACHMENT_STORE_OP_DONT_CARE,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
	};

	VkFormat depthFormat = FindSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

	VkAttachmentDescription depthAttachment = {
		0,
		depthFormat,
		VK_SAMPLE_COUNT_1_BIT,
		VK_ATTACHMENT_LOAD_OP_CLEAR,
		VK_ATTACHMENT_STORE_OP_DONT_CARE,
		VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		VK_ATTACHMENT_STORE_OP_DONT_CARE,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
	};

	VkAttachmentReference colorAttachmentRef = {
		0,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	};

	VkAttachmentReference depthAttachmentRef = {
		1,
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
	};

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;
	subpass.pDepthStencilAttachment = &depthAttachmentRef;

	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = SIZEOF(attachments);
	renderPassInfo.pAttachments = FPTR(attachments);
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;

	if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass!");
	}
}

VkFormat VulkanRenderer::FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
	for (VkFormat format : candidates) {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
			return format;
		}
	}

	throw std::runtime_error("failed to find supported format!");
}

void VulkanRenderer::CreateDescriptorSetLayout()
{
	{
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout.mesh) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	{
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout.sky) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}


}

void VulkanRenderer::CreateGraphicsPipeline()
{
	{
		auto vertShaderCode = ReadFile("Shaders/mesh.vert.spv");
		auto fragShaderCode = ReadFile("Shaders/mesh.frag.spv");

		VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = VertexMesh::getBindingDescriptions();
		auto attributeDescriptions = VertexMesh::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescription.size());;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.flags = 0;

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout.mesh;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout.mesh) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.layout = pipelineLayout.mesh;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline.mesh) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	{
		auto vertShaderCode = ReadFile("Shaders/sky.vert.spv");
		auto fragShaderCode = ReadFile("Shaders/sky.frag.spv");

		VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = VertexSky::getBindingDescriptions();
		auto attributeDescriptions = VertexSky::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescription.size());;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.flags = 0;

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout.sky;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout.sky) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.layout = pipelineLayout.sky;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline.sky) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}
}

VkShaderModule VulkanRenderer::CreateShaderModule(const std::vector<char>& code) {
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module!");
	}

	return shaderModule;
}

void VulkanRenderer::CreateDepthResources()
{
	VkFormat depthFormat = FindDepthFormat();
	CreateImage(swapChainExtent.width, swapChainExtent.height, depthFormat, 1, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
	depthImageView = CreateImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_VIEW_TYPE_2D, 1);
}

VkFormat VulkanRenderer::FindDepthFormat()
{
	return FindSupportedFormat(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);
}

void VulkanRenderer::CreateFramebuffers()
{
	swapChainFramebuffers.resize(swapChainImageViews.size());
	for (size_t i = 0; i < swapChainImageViews.size(); i++) {
		std::array<VkImageView, 2> attachments = {
			swapChainImageViews[i],
			depthImageView
		};

		VkFramebufferCreateInfo framebufferInfo = {};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = swapChainExtent.width;
		framebufferInfo.height = swapChainExtent.height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}
}

void VulkanRenderer::CreateTextureSampler()
{
	{
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler.mesh) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}
	{
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16;
		samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler.sky) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}
}

void VulkanRenderer::CreateVertexBuffer()
{
	{
		VkDeviceSize bufferSize = vertices.mesh.size() * sizeof(VertexMesh);

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.mesh.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer.mesh, vertexBufferMemory.mesh);

		CopyBufferToBuffer(stagingBuffer, vertexBuffer.mesh, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
	{
		VkDeviceSize bufferSize = vertices.sky.size() * sizeof(VertexSky);

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.sky.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer.sky, vertexBufferMemory.sky);

		CopyBufferToBuffer(stagingBuffer, vertexBuffer.sky, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
}

void VulkanRenderer::CreateIndexBuffer()
{
	{
		VkDeviceSize bufferSize = static_cast<uint64_t>(indices.mesh.size()) * sizeof(uint32_t);

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.mesh.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer.mesh, indexBufferMemory.mesh);

		CopyBufferToBuffer(stagingBuffer, indexBuffer.mesh, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
	{
		VkDeviceSize bufferSize = static_cast<uint64_t>(indices.sky.size()) * sizeof(uint32_t);

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.sky.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer.sky, indexBufferMemory.sky);

		CopyBufferToBuffer(stagingBuffer, indexBuffer.sky, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
}

void VulkanRenderer::PrepareInstanceData()
{
	std::vector<InstanceData> instanceData;
	instanceData.resize(2);

	std::default_random_engine rndEngine((unsigned)time(nullptr));
	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (uint32_t i = 0; i < 2; i++) {
		float theta = 2 * float(M_PI) * uniformDist(rndEngine);
		float phi = acos(1 - 2 * uniformDist(rndEngine));

		glm::vec3 instancePos = glm::vec3(sin(phi) * cos(theta), 0.0f, cos(phi)) * 150.0f;
		glm::vec3 instanceRot = glm::vec3(0.0f, float(M_PI) * uniformDist(rndEngine), 0.0f);
		float scale = 0.1f;

		glm::mat4 model = glm::translate(glm::mat4(1.0f), instancePos);
		model = glm::rotate(model, glm::radians(instanceRot.x), glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, glm::radians(instanceRot.y), glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(model, glm::radians(instanceRot.z), glm::vec3(0.0f, 0.0f, 1.0f));
		model = glm::scale(model, glm::vec3(scale));

		instanceData[i].model_0 = model[0];
		instanceData[i].model_1 = model[1];
		instanceData[i].model_2 = model[2];
		instanceData[i].model_3 = model[3];
	}


	VkDeviceSize bufferSize = static_cast<uint64_t>(instanceData.size() * sizeof(InstanceData));
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
	void* data;
	vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, instanceData.data(), (size_t)bufferSize);
	vkUnmapMemory(device, stagingBufferMemory);

	CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, instanceBuffer.mesh, instanceBufferMemory.mesh);

	CopyBufferToBuffer(stagingBuffer, instanceBuffer.mesh, bufferSize);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);
}


void VulkanRenderer::CreateUniformBuffers()
{
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObjectMesh);

		uniformBuffers.mesh.resize(swapChainImages.size());
		uniformBuffersMemory.mesh.resize(swapChainImages.size());


		for (size_t i = 0; i < swapChainImages.size(); i++) {
			CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers.mesh[i], uniformBuffersMemory.mesh[i]);
		}
	}
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObjectSky);

		uniformBuffers.sky.resize(swapChainImages.size());
		uniformBuffersMemory.sky.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers.sky[i], uniformBuffersMemory.sky[i]);
		}
	}
}

void VulkanRenderer::CreateDescriptorPool()
{
	{
		std::array<VkDescriptorPoolSize, 2> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool.mesh) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}
	{
		std::array<VkDescriptorPoolSize, 2> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool.sky) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}
}

void VulkanRenderer::CreateDescriptorSets()
{
	{
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout.mesh);
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool.mesh;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.mesh.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.mesh.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = uniformBuffers.mesh[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObjectMesh);

			VkDescriptorImageInfo imageInfo = {};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageView.mesh;
			imageInfo.sampler = textureSampler.mesh;

			std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets.mesh[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets.mesh[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}
	{
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout.sky);
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool.sky;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.sky.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.sky.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = uniformBuffers.sky[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObjectSky);

			VkDescriptorImageInfo imageInfo = {};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageView.sky;
			imageInfo.sampler = textureSampler.sky;

			std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets.sky[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets.sky[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}
}

void VulkanRenderer::CreateCommandBuffers()
{
	commandBuffers.resize(swapChainFramebuffers.size());

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

	if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffers!");
	}

	for (size_t i = 0; i < commandBuffers.size(); i++)
	{
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[i];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues = {};
		clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clearValues[1].depthStencil = { 1.0f, 0 };

		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport = {};
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);

		VkRect2D scissor = {};
		scissor.extent.width = swapChainExtent.width;
		scissor.extent.height = swapChainExtent.height;
		scissor.offset.x = 0;
		scissor.offset.y = 0;
		vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);

		VkDeviceSize offsets[1] = { 0 };

		//Sky
		vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout.sky, 0, 1, &descriptorSets.sky[i], 0, nullptr);
		vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline.sky);
		vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &vertexBuffer.sky, offsets);
		vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer.sky, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.sky.size()), 1, 0, 0, 0);

		//Mesh
		vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout.mesh, 0, 1, &descriptorSets.mesh[i], 0, nullptr);
		vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline.mesh);
		vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &vertexBuffer.mesh, offsets);
		vkCmdBindVertexBuffers(commandBuffers[i], 1, 1, &instanceBuffer.mesh, offsets);
		vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer.mesh, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.mesh.size()), 2, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffers[i]);

		if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}
}

void VulkanRenderer::CreateSynchronizationObjects() {
	imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
	imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

	VkSemaphoreCreateInfo semaphoreInfo = {};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
			vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create synchronization objects for a frame!");
		}
	}
}

void VulkanRenderer::UpdateUniformBuffer(uint32_t currentImage)
{
	{
		uboMesh.proj = camera.matrices.perspective;//glm::perspective(glm::radians(60.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 1024.0f);
		uboMesh.view = camera.matrices.view;
		/*{
			uboMesh.model[0] = glm::translate(glm::mat4(1.0f), glm::vec3(-150.f, 0.0f, 0.0f));
			uboMesh.model[1] = glm::translate(glm::mat4(1.0f), glm::vec3(150.f, 0.0f, 0.0f));
			for (uint32_t i = 0; i < 2; i++)
			{
				uboMesh.model[i] = glm::rotate(uboMesh.model[i], glm::radians(0.f), glm::vec3(1.0f, 0.0f, 0.0f));
				uboMesh.model[i] = glm::rotate(uboMesh.model[i], glm::radians(0.f), glm::vec3(0.0f, 1.0f, 0.0f));
				uboMesh.model[i] = glm::rotate(uboMesh.model[i], glm::radians(0.f), glm::vec3(0.0f, 0.0f, 1.0f));
				uboMesh.model[i] = glm::scale(uboMesh.model[i], glm::vec3(1.0f));
			}
		}

		{
			uboMesh.view = glm::translate(glm::mat4(1.0f), glm::vec3(cameraPos.x, -cameraPos.z, -zoom));
			uboMesh.view = glm::rotate(uboMesh.view, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
			uboMesh.view = glm::rotate(uboMesh.view, glm::radians(rotation.z), glm::vec3(0.0f, 1.0f, 0.0f));
			uboMesh.view = glm::rotate(uboMesh.view, glm::radians(rotation.y), glm::vec3(0.0f, 0.0f, 1.0f));
		}
		*/

		for (uint32_t i = 0; i < skinnedMeshs.size(); i++)
		{
			skinnedMeshs[i]->update(runningTime);
			for (uint32_t j = 0; j < skinnedMeshs[i]->boneTransforms.size(); j++)
			{
				uboMesh.boneDataInstance[i].bones[j] = glm::transpose(glm::make_mat4(&skinnedMeshs[i]->boneTransforms[j].a1));
			}
		}


		void* data;
		vkMapMemory(device, uniformBuffersMemory.mesh[currentImage], 0, sizeof(uboMesh), 0, &data);
		memcpy(data, &uboMesh, sizeof(uboMesh));
		vkUnmapMemory(device, uniformBuffersMemory.mesh[currentImage]);
	}
	{
		uboSky.proj = camera.matrices.perspective;
		//uboSky.proj = glm::perspective(glm::radians(60.0f), (float)swapChainExtent.width / (float)swapChainExtent.height, 0.001f, 256.0f);

		uboSky.view = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0));
		uboSky.view = glm::rotate(uboSky.view, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboSky.view = glm::rotate(uboSky.view, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboSky.view = glm::rotate(uboSky.view, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		void* data;
		vkMapMemory(device, uniformBuffersMemory.sky[currentImage], 0, sizeof(uboSky), 0, &data);
		memcpy(data, &uboSky, sizeof(uboSky));
		vkUnmapMemory(device, uniformBuffersMemory.sky[currentImage]);
	}
}

void VulkanRenderer::RecreateSwapChain()
{
	int width = 0, height = 0;
	while (width == 0 || height == 0) {
		RECT rect;
		if (GetWindowRect(windowParams.HWnd, &rect))
		{
			width = rect.right - rect.left;
			height = rect.bottom - rect.top;
		}
	}

	vkDeviceWaitIdle(device);

	CleanupSwapChain();

	CreateSwapChain();
	CreateSwapChainImageViews();
	CreateRenderPass();
	CreateGraphicsPipeline();
	CreateDepthResources();
	CreateFramebuffers();
	CreateUniformBuffers();
	CreateDescriptorPool();
	CreateDescriptorSets();
	CreateCommandBuffers();
}

void VulkanRenderer::CleanupSwapChain()
{
	vkDestroyImageView(device, depthImageView, nullptr);
	vkDestroyImage(device, depthImage, nullptr);
	vkFreeMemory(device, depthImageMemory, nullptr);

	for (auto framebuffer : swapChainFramebuffers) {
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	}

	vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

	vkDestroyPipeline(device, graphicsPipeline.mesh, nullptr);
	vkDestroyPipelineLayout(device, pipelineLayout.mesh, nullptr);
	vkDestroyPipeline(device, graphicsPipeline.sky, nullptr);
	vkDestroyPipelineLayout(device, pipelineLayout.sky, nullptr);
	vkDestroyRenderPass(device, renderPass, nullptr);

	for (auto imageView : swapChainImageViews) {
		vkDestroyImageView(device, imageView, nullptr);
	}

	vkDestroySwapchainKHR(device, swapChain, nullptr);

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		vkDestroyBuffer(device, uniformBuffers.mesh[i], nullptr);
		vkFreeMemory(device, uniformBuffersMemory.mesh[i], nullptr);
		vkDestroyBuffer(device, uniformBuffers.sky[i], nullptr);
		vkFreeMemory(device, uniformBuffersMemory.sky[i], nullptr);
	}

	vkDestroyDescriptorPool(device, descriptorPool.mesh, nullptr);
	vkDestroyDescriptorPool(device, descriptorPool.sky, nullptr);
}

void VulkanRenderer::MouseClick(size_t button_index, bool state)
{
	if (2 > button_index) {
		mouseState.Buttons[button_index].IsPressed = state;
		mouseState.Buttons[button_index].WasClicked = state;
		mouseState.Buttons[button_index].WasRelease = !state;
	}
}

void VulkanRenderer::MouseMove(int x, int y)
{
	mouseState.Position.Delta.X = x - mouseState.Position.X;
	mouseState.Position.Delta.Y = y - mouseState.Position.Y;
	mouseState.Position.X = x;
	mouseState.Position.Y = y;
}

void VulkanRenderer::MouseWheel(float distance)
{
	mouseState.Wheel.WasMoved = true;
	mouseState.Wheel.Distance = distance;
}