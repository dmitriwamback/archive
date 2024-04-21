#if defined(DEBUG_VULKAN_TEST)

#include <vulkan/vulkan.h>

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <optional>
#include <set>
#include <array>
#include <fstream>

#if defined(__APPLE__)
    #define APPLE_DEBUG_TEST
#endif

int currentFrame = 0;
const int maxFrames = 1;

typedef struct context {

    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue, presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkFormat format;
    VkExtent2D extent;
    GLFWwindow* window;
    VkDebugUtilsMessengerEXT debug;
} context;

std::vector<VkImage> swapchainImages;
std::vector<VkImageView> swapchainImageViews;

typedef struct swapchainDetails {

    VkSurfaceCapabilitiesKHR capabilites;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
} swapchainDetails;

typedef struct queuefamily {
    std::optional<uint32_t> graphicsFamily, presentQueue;
} queuefamily;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
}, deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#if defined(APPLE_DEBUG_TEST)
    "VK_KHR_portability_subset"
#endif
};

#if defined(NDEBUG)
    const bool validation = false;
#else
    const bool validation = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT callbackData,
                                                    void* userData) {

    std::cerr << callbackData.pMessage << '\n';
    return VK_FALSE;
}

static context* vkcontext;
queuefamily family;

bool CheckValidationLayerSupport() {

    uint32_t layerCount;
    std::vector<VkLayerProperties> properties;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    properties.resize(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, properties.data());

    for (const char* layername : validationLayers) {
        bool layerFound = false;
        for (const auto& layerp : properties) { if (strcmp(layername, layerp.layerName) == 0) { layerFound = true; break; } }

        if (!layerFound) return false;
    }
    return true;
}

queuefamily FindQueueFamily(VkPhysicalDevice device) {
    queuefamily family;

    uint32_t familyCount;
    std::vector<VkQueueFamilyProperties> families;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);

    families.resize(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());

    int i = 0;
    for (const auto& queueFamily : families) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) family.graphicsFamily = i;

        VkBool32 present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, vkcontext->surface, &present);

        if (present) family.presentQueue = i;
        if (family.graphicsFamily.has_value() && family.presentQueue.has_value()) return family;
        i++;
    }

    return family;
}

bool CheckDeviceExtensions(VkPhysicalDevice device) {

    uint32_t extCount;
    std::vector<VkExtensionProperties> availableProperties;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    availableProperties.resize(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, availableProperties.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& ext : availableProperties) {
        requiredExtensions.erase(ext.extensionName);
    }

    return requiredExtensions.empty();
}

swapchainDetails QuerySwapchainDetails(VkPhysicalDevice device) {
    swapchainDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, vkcontext->surface, &details.capabilites);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, vkcontext->surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, vkcontext->surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, vkcontext->surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, vkcontext->surface,&presentModeCount, details.presentModes.data());
    }

    return details;
}

bool IsDeviceSuitable(VkPhysicalDevice device) {

    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceProperties(device, &properties);
    vkGetPhysicalDeviceFeatures(device, &features);

    swapchainDetails details = QuerySwapchainDetails(device);
    bool adequate = false;
    adequate = !details.formats.empty() && !details.presentModes.empty();

    family = FindQueueFamily(device);
    return family.graphicsFamily.has_value() && CheckDeviceExtensions(device) && adequate;
}

VkSurfaceFormatKHR ChooseSwapchainSurface(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

    for (const auto& format : availableFormats) {
        if (format.format == VK_FORMAT_B8G8R8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR ChooseSwapchainPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {

    for (const auto& presentMode : availablePresentModes) {
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) return presentMode;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {

    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) return capabilities.currentExtent;

    int width, height;
    glfwGetFramebufferSize(vkcontext->window, &width, &height);
    VkExtent2D extent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return extent;
}

VkPhysicalDevice FindPhysicalDevice(std::vector<VkPhysicalDevice> devices) {

    for (VkPhysicalDevice device : devices) {
        if (IsDeviceSuitable(device)) return device;
    }

    throw std::runtime_error("Could not find a suitable device!");
    return VK_NULL_HANDLE;
}

std::vector<const char*> GetRequiredExtensions() {
    uint32_t glfwExtCount;
    const char** glfwExts;

    glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

    if (validation) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
}

void CreateDevice() {

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    for (uint32_t qFamily : std::set<uint32_t>{family.graphicsFamily.value(), family.presentQueue.value()}) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = qFamily;
        queueCreateInfo.queueCount = 1;
        float priority = 1.0f;
        queueCreateInfo.pQueuePriorities = &priority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures features{};
    VkDeviceCreateInfo deviceCreateInfo{};

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &features;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    deviceCreateInfo.enabledExtensionCount = deviceExtensions.size();

    if (vkCreateDevice(vkcontext->physicalDevice, &deviceCreateInfo, nullptr, &vkcontext->device) != VK_SUCCESS) throw std::runtime_error("Couldn't create logical device!");
    vkGetDeviceQueue(vkcontext->device, family.graphicsFamily.value(), 0, &vkcontext->graphicsQueue);
    vkGetDeviceQueue(vkcontext->device, family.presentQueue.value(), 0, &vkcontext->presentQueue);
}

std::vector<VkFramebuffer> framebuffers = std::vector<VkFramebuffer>();
std::vector<VkCommandBuffer> commandbuffers = std::vector<VkCommandBuffer>();
VkRenderPass renderpass;

void CreateSwapchain() {
    swapchainDetails details = QuerySwapchainDetails(vkcontext->physicalDevice);
    VkSurfaceFormatKHR sformat = ChooseSwapchainSurface(details.formats);
    VkPresentModeKHR pMode = ChooseSwapchainPresentMode(details.presentModes);
    VkExtent2D extent = ChooseSwapExtent(details.capabilites);

    vkcontext->extent = extent;
    vkcontext->format = sformat.format;
    uint32_t imageCount = details.capabilites.minImageCount + 1;
    if (details.capabilites.maxImageCount > 0 && imageCount > details.capabilites.maxImageCount) imageCount = details.capabilites.maxImageCount;

    VkSwapchainCreateInfoKHR swapchainCreateInfo{};
    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.surface = vkcontext->surface;
    swapchainCreateInfo.minImageCount = imageCount;
    swapchainCreateInfo.imageFormat = sformat.format;
    swapchainCreateInfo.imageColorSpace = sformat.colorSpace;
    swapchainCreateInfo.imageExtent = extent;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    queuefamily indices = FindQueueFamily(vkcontext->physicalDevice);
    if (indices.graphicsFamily != indices.presentQueue) {
        uint32_t q[] = {indices.graphicsFamily.value(), indices.presentQueue.value()};
        swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainCreateInfo.queueFamilyIndexCount = 2;
        swapchainCreateInfo.pQueueFamilyIndices = q;
    }
    else swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

    swapchainCreateInfo.pQueueFamilyIndices = nullptr;
    swapchainCreateInfo.preTransform = details.capabilites.currentTransform;
    swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCreateInfo.presentMode = pMode;
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(vkcontext->device, &swapchainCreateInfo, nullptr, &vkcontext->swapchain) != VK_SUCCESS) throw std::runtime_error("Couldn't create Swapchain!");

    vkGetSwapchainImagesKHR(vkcontext->device, vkcontext->swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(vkcontext->device, vkcontext->swapchain, &imageCount, swapchainImages.data());

    swapchainImageViews.resize(swapchainImages.size());
    framebuffers.resize(swapchainImageViews.size());
    for (size_t i = 0; i < swapchainImages.size(); i++) {

        VkImageViewCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageCreateInfo.image = swapchainImages[i];
        imageCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageCreateInfo.format = vkcontext->format;

        imageCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        imageCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCreateInfo.subresourceRange.baseMipLevel = 0;
        imageCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageCreateInfo.subresourceRange.levelCount = 1;
        imageCreateInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(vkcontext->device, &imageCreateInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) throw std::runtime_error("Couldn't create image views!");
    }
}

typedef struct Vertex {

    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;

    static VkVertexInputBindingDescription GetBindingDescription() {

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding;
    }

    static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescription() {

        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, uv);

        return attributeDescriptions;
    }
};

uint32_t FindMemoryType(uint32_t filter, VkMemoryPropertyFlags properties) {

    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(vkcontext->physicalDevice, &memory);

    for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
        if (filter & (1 << 0) && (memory.memoryTypes[i].propertyFlags & properties) == properties) return i;
    }
    throw std::runtime_error("Couldn't find memory type!");
}

typedef struct VertexBuffer {
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
} VertexBuffer;


class Pipeline {
public:
    typedef struct content {
        VkPipelineLayout pipelineLayout;
        VkRenderPass renderPass;
        VkRect2D scissor{};
        VkViewport viewport;
    } content;

    typedef struct pipeline {
        VkPipelineDynamicStateCreateInfo            dynamicState{};
        VkPipelineVertexInputStateCreateInfo        vertexInput{};
        VkPipelineInputAssemblyStateCreateInfo      inputAssembly{};
        VkPipelineViewportStateCreateInfo           viewportState{};
        VkPipelineRasterizationStateCreateInfo      rasterizer{};
        VkPipelineColorBlendStateCreateInfo         colorBlending{};
        VkCommandPool commandPool;
        uint32_t imageIndex;
    } pipeline;

    std::vector<VkCommandBuffer> commandBuffers = std::vector<VkCommandBuffer>();
    std::vector<VkSemaphore> imageSemaphores = std::vector<VkSemaphore>();
    std::vector<VkSemaphore> renderSemaphores = std::vector<VkSemaphore>();
    std::vector<VkFence> inFlight = std::vector<VkFence>();

    std::map<std::string, VkPipelineShaderStageCreateInfo> shaderPrograms;
    std::vector<Vertex> vertices;

    content* contentData;
    pipeline* pipelineData;
    VertexBuffer* vertexbuffer;
    VkPipeline shaderPipeline;

    static Pipeline Create(std::string ShaderFolder, std::vector<Vertex> vertices);
    static std::vector<char> LoadFileContents(std::string path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);

        if (!file.is_open()) throw std::runtime_error("Failed to open shader file");
        size_t filesize = (size_t)file.tellg();
        std::vector<char> buf(filesize);

        file.seekg(0);
        file.read(buf.data(), filesize);
        file.close();
        return buf;
    }
    static VkShaderModule CreateShader(std::vector<char> shaderSource);

    void Bind();
private:
    void CreatePipeline();
    void Synchronize();
    void CreateCommandBuffer();
    VkGraphicsPipelineCreateInfo CreatePipelineInfo();
};

VkShaderModule Pipeline::CreateShader(std::vector<char> shaderSource) {

    VkShaderModule shader;
    VkShaderModuleCreateInfo shaderCreateInfo{};
    shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderCreateInfo.codeSize = shaderSource.size();
    shaderCreateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderSource.data());

    if (vkCreateShaderModule(vkcontext->device, &shaderCreateInfo, nullptr, &shader) != VK_SUCCESS) throw std::runtime_error("Failed to compile shader!");
    return shader;
}

Pipeline Pipeline::Create(std::string ShaderFolder, std::vector<Vertex> vertices) {

    Pipeline pipeline = Pipeline();
    std::string vShaderPath = ShaderFolder + "/source/vMain.spv";
    std::string fShaderPath = ShaderFolder + "/source/fMain.spv";

    VkShaderModule vert = CreateShader(LoadFileContents(vShaderPath));
    VkShaderModule frag = CreateShader(LoadFileContents(fShaderPath));

    VkPipelineShaderStageCreateInfo vertex{}, fragment{};

    vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertex.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertex.module = vert;
    vertex.pName = "main";

    fragment.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragment.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragment.module = frag;
    fragment.pName = "main";

    std::map<std::string, VkPipelineShaderStageCreateInfo> programs;
    programs["vert"] = vertex;
    programs["frag"] = fragment;

    pipeline.shaderPrograms = programs;
    pipeline.vertices = vertices;

    pipeline.CreatePipeline();
    pipeline.CreateCommandBuffer();

    return pipeline;
}
void Pipeline::CreatePipeline() {

    contentData = static_cast<content*>(malloc(1 * sizeof(content)));
    pipelineData = static_cast<pipeline*>(malloc(1 * sizeof(pipeline)));
    vertexbuffer = static_cast<VertexBuffer*>(malloc(1 * sizeof(VertexBuffer)));


    VkBufferCreateInfo vertexBufferInfo{};
    vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vertexBufferInfo.size = sizeof(vertices[0]) * vertices.size();
    vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(vkcontext->device, &vertexBufferInfo, nullptr, &vertexbuffer->buffer) != VK_SUCCESS) throw std::runtime_error("Failed to create vertex buffer!");

    VkMemoryRequirements memory;
    vkGetBufferMemoryRequirements(vkcontext->device, vertexbuffer->buffer, &memory);
    VkMemoryAllocateInfo memAllocInfo{};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocInfo.allocationSize = memory.size;
    memAllocInfo.memoryTypeIndex = FindMemoryType(memory.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    

    if (vkAllocateMemory(vkcontext->device, &memAllocInfo, nullptr, &vertexbuffer->bufferMemory) != VK_SUCCESS) throw std::runtime_error("Couldn't allocate memory");
    vkBindBufferMemory(vkcontext->device, vertexbuffer->buffer, vertexbuffer->bufferMemory, 0);

    void* data;
    vkMapMemory(vkcontext->device, vertexbuffer->bufferMemory, 0, vertexBufferInfo.size, 0, &data);
    memcpy(data, vertices.data(), (size_t)vertexBufferInfo.size);
    vkUnmapMemory(vkcontext->device, vertexbuffer->bufferMemory);

    VkRect2D scissor{};
    contentData->scissor.offset = {0, 0};
    contentData->scissor.extent = vkcontext->extent;

    std::vector<VkDynamicState> dynamicState = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    pipelineData->dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    pipelineData->dynamicState.dynamicStateCount = dynamicState.size();
    pipelineData->dynamicState.pDynamicStates = dynamicState.data();
    pipelineData->dynamicState.flags = 0;
    pipelineData->dynamicState.pNext = NULL;

    pipelineData->vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pipelineData->vertexInput.vertexBindingDescriptionCount = 1;

    VkVertexInputBindingDescription desc = Vertex::GetBindingDescription();

    pipelineData->vertexInput.pVertexBindingDescriptions = &desc;
    pipelineData->vertexInput.vertexAttributeDescriptionCount = Vertex::GetAttributeDescription().size();
    pipelineData->vertexInput.pVertexAttributeDescriptions = Vertex::GetAttributeDescription().data();
    pipelineData->vertexInput.flags = 0;
    pipelineData->vertexInput.pNext = NULL;

    pipelineData->inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    pipelineData->inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pipelineData->inputAssembly.primitiveRestartEnable = VK_FALSE;
    pipelineData->inputAssembly.flags = 0;
    pipelineData->inputAssembly.pNext = NULL;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = vkcontext->extent.width;
    viewport.height = vkcontext->extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    contentData->viewport = viewport;

    pipelineData->viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    pipelineData->viewportState.viewportCount = 1;
    pipelineData->viewportState.scissorCount = 1;
    pipelineData->viewportState.pViewports = &viewport;
    pipelineData->viewportState.pScissors = &scissor;
    pipelineData->viewportState.flags = 0;
    pipelineData->viewportState.pNext = NULL;

    pipelineData->rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    pipelineData->rasterizer.depthClampEnable = VK_FALSE;
    pipelineData->rasterizer.rasterizerDiscardEnable = VK_FALSE;
    pipelineData->rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    pipelineData->rasterizer.lineWidth = 1;
    pipelineData->rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    pipelineData->rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    pipelineData->rasterizer.depthBiasEnable = VK_FALSE;
    pipelineData->rasterizer.depthBiasConstantFactor = 0;
    pipelineData->rasterizer.depthBiasClamp = 0;
    pipelineData->rasterizer.depthBiasSlopeFactor = 0;
    pipelineData->rasterizer.flags = 0;
    pipelineData->rasterizer.pNext = NULL;
    


    VkPipelineColorBlendAttachmentState color{};
    color.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color.blendEnable = VK_FALSE;

    pipelineData->colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    pipelineData->colorBlending.logicOpEnable = VK_FALSE;
    pipelineData->colorBlending.attachmentCount = 1;
    pipelineData->colorBlending.pAttachments = &color;
    pipelineData->colorBlending.flags = 0;
    pipelineData->colorBlending.pNext = NULL;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    if (vkCreatePipelineLayout(vkcontext->device, &pipelineLayoutInfo, nullptr, &contentData->pipelineLayout) != VK_SUCCESS) throw std::runtime_error("Failed to create pipelineLayout");

    VkGraphicsPipelineCreateInfo pipelineInfo{};

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = vkcontext->format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderpassInfo{};
    renderpassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderpassInfo.attachmentCount = 1;
    renderpassInfo.subpassCount = 1;
    renderpassInfo.pAttachments = &colorAttachment;
    renderpassInfo.pSubpasses = &subpass;

    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pVertexInputState = &pipelineData->vertexInput;
    pipelineInfo.pInputAssemblyState = &pipelineData->inputAssembly;
    pipelineInfo.pViewportState = &pipelineData->viewportState;
    pipelineInfo.pRasterizationState = &pipelineData->rasterizer;
    pipelineInfo.pColorBlendState = &pipelineData->colorBlending;
    pipelineInfo.pDynamicState = &pipelineData->dynamicState;
    
    if (vkCreateRenderPass(vkcontext->device, &renderpassInfo, nullptr, &contentData->renderPass) != VK_SUCCESS) throw std::runtime_error("Failed to create RenderPass!");

    std::cout << "Hello\n";
    for (size_t i = 0; i < swapchainImageViews.size(); i++) {

        VkImageView attachments[] = {swapchainImageViews[i]};

        VkFramebufferCreateInfo fbCreateInfo{};
        fbCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbCreateInfo.renderPass = contentData->renderPass;
        fbCreateInfo.attachmentCount = 1;
        fbCreateInfo.pAttachments = attachments;
        fbCreateInfo.width = vkcontext->extent.width;
        fbCreateInfo.height = vkcontext->extent.height;
        fbCreateInfo.layers = 1;
        
        if (vkCreateFramebuffer(vkcontext->device, &fbCreateInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) throw std::runtime_error("Framebuffer");
    }

    pipelineInfo.layout = contentData->pipelineLayout;
    pipelineInfo.renderPass = contentData->renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VkPipelineShaderStageCreateInfo shaderStages[] = {shaderPrograms["vert"], shaderPrograms["frag"]};

    pipelineInfo.pStages = shaderStages;

    if (vkCreateGraphicsPipelines(vkcontext->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shaderPipeline) != VK_SUCCESS) throw std::runtime_error("Couldn't create VkPipeline!");
}
void Pipeline::CreateCommandBuffer() {

    commandBuffers.resize(maxFrames);
    renderSemaphores.resize(maxFrames);
    imageSemaphores.resize(maxFrames);
    inFlight.resize(maxFrames);

    queuefamily _family = FindQueueFamily(vkcontext->physicalDevice);
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = _family.graphicsFamily.value();

    if (vkCreateCommandPool(vkcontext->device,
                            &poolInfo, nullptr,
                            &pipelineData->commandPool) != VK_SUCCESS) throw std::runtime_error("CommandPool");

    VkCommandBufferAllocateInfo commandBufferAllocInfo{};
    commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocInfo.commandBufferCount = maxFrames;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocInfo.commandPool = pipelineData->commandPool;

    if (vkAllocateCommandBuffers(vkcontext->device,
                                &commandBufferAllocInfo,
                                commandBuffers.data()) != VK_SUCCESS) throw std::runtime_error("CommandBuffer");

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < maxFrames; i++) {
        if (vkCreateSemaphore(vkcontext->device, &semaphoreCreateInfo, nullptr, &imageSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(vkcontext->device, &semaphoreCreateInfo, nullptr, &renderSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(vkcontext->device, &fenceInfo, nullptr, &inFlight[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Couldn't create semaphores!");
        }
    }


}
void Pipeline::Bind() {
    vkWaitForFences(vkcontext->device, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;
    
    vkAcquireNextImageKHR(vkcontext->device, vkcontext->swapchain, UINT64_MAX, imageSemaphores[currentFrame], VK_NULL_HANDLE, &pipelineData->imageIndex);
    vkResetFences(vkcontext->device, 1, &inFlight[currentFrame]);
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);

    if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS) throw std::runtime_error("Begin Info");
    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = contentData->renderPass;
    renderPassBeginInfo.framebuffer = framebuffers[pipelineData->imageIndex];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    renderPassBeginInfo.renderArea.extent = vkcontext->extent;

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffers[currentFrame], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, shaderPipeline);

    contentData->viewport.width = static_cast<uint32_t>(vkcontext->extent.width);
    contentData->viewport.height = static_cast<uint32_t>(vkcontext->extent.height);

    vkCmdSetViewport(commandBuffers[currentFrame], 0, 1, &contentData->viewport);
    vkCmdSetScissor(commandBuffers[currentFrame], 0, 1, &contentData->scissor);

    VkBuffer buffer[] = {vertexbuffer->buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffers[currentFrame], 0, 1, buffer, offsets);
    vkCmdDraw(commandBuffers[currentFrame], 3, 1, 0, 0);
    vkCmdEndRenderPass(commandBuffers[currentFrame]);

    if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS) throw std::runtime_error("Failed to record command buffer!");

    VkSemaphore waitSemaphores[] = {imageSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

    VkSemaphore signal[] = {renderSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signal;

    if (vkQueueSubmit(vkcontext->presentQueue, 1, &submitInfo, inFlight[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signal;

    VkSwapchainKHR swapChains[] = {vkcontext->swapchain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &pipelineData->imageIndex;

    vkQueuePresentKHR(vkcontext->presentQueue, &presentInfo);
}


void MainLoop() {

    std::vector<Vertex> triangle = {
        {{ 0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
        {{ 0.5f,  0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{-0.2f,  0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
    };



    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = vkcontext->extent.width;
    viewport.height = vkcontext->extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissors{};
    scissors.offset = {0, 0};
    scissors.extent = vkcontext->extent;

    Pipeline p = Pipeline::Create("shaders", triangle);

    while (!glfwWindowShouldClose(vkcontext->window)) {
        currentFrame = (currentFrame + 1) % maxFrames;
        glfwPollEvents();

        p.Bind();
    }
}


void Initialize() {
    vkcontext = static_cast<context*>(malloc(1 * sizeof(context)));

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    vkcontext->window = glfwCreateWindow(1200, 800, "Vulkan", nullptr, nullptr);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VK";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "VK";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    uint32_t extensionCount;
    const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);

    std::vector<const char*> required;
    for (int i = 0; i < extensionCount; i++) {
        required.emplace_back(extensions[i]);
    }

    VkInstanceCreateInfo iCreateInfo{};
    iCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    iCreateInfo.pApplicationInfo = &appInfo;
#if defined(APPLE_DEBUG_TEST)
    required.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    required.emplace_back("VK_KHR_get_physical_device_properties2");
    iCreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    iCreateInfo.enabledExtensionCount = (uint32_t)required.size();
    iCreateInfo.ppEnabledExtensionNames = required.data();
    iCreateInfo.enabledLayerCount = 0;

    if (validation && !CheckValidationLayerSupport()) std::cout << "Validation layers not found!\n";
    if (validation) {
        //iCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        //iCreateInfo.ppEnabledExtensionNames = validationLayers.data();
    }
    else iCreateInfo.enabledLayerCount = 0;
    std::vector<const char*> _exts = GetRequiredExtensions();
    //iCreateInfo.enabledExtensionCount = static_cast<uint32_t>(_exts.size());
    //iCreateInfo.ppEnabledExtensionNames = _exts.data();

    if (vkCreateInstance(&iCreateInfo, nullptr, &vkcontext->instance) != VK_SUCCESS) throw std::runtime_error("Couldn't create VkInstance");
    
    uint32_t deviceCount;
    std::vector<VkPhysicalDevice> devices;
    vkEnumeratePhysicalDevices(vkcontext->instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("No physical devices found!");

    devices.resize(deviceCount);
    vkEnumeratePhysicalDevices(vkcontext->instance, &deviceCount, devices.data());

    VkResult result = glfwCreateWindowSurface(vkcontext->instance, vkcontext->window, nullptr, &vkcontext->surface);

    vkcontext->physicalDevice = FindPhysicalDevice(devices);
    CreateDevice();
    CreateSwapchain();
    MainLoop();
}

#endif
