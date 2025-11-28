#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <random>
#include <numbers>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t PARTICLE_COUNT = 2000;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct UniformBufferObject
{
    float deltaTime = 1.0f;
};

struct Particle
{
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color)),
        };
    }
};

class HelloTriangleApplication
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *window = nullptr;
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    uint32_t queueIndex = ~0;
    std::unique_ptr<vk::raii::Queue> queue = nullptr;
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainSurfaceFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    std::vector<vk::raii::Buffer> shaderStorageBuffers;
    std::vector<vk::raii::DeviceMemory> shaderStorageBuffersMemory;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::raii::CommandBuffer> computeCommandBuffers;

    vk::raii::Semaphore semaphore = nullptr;
    uint64_t timelineValue = 0;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t currentFrame = 0;

    double lastFrameTime = 0.0f;

    double lastTime = 0.0f;

    std::vector<const char *> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName};

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createComputeDescriptorSetLayout();
        createGraphicsPipeline();
        createComputePipeline();
        createCommandPool();
        createShaderStorageBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createComputeDescriptorSets();
        createCommandBuffers();
        createComputeCommandBuffers();
        createSyncObjects();
    }

    void createShaderStorageBuffers()
    {
        shaderStorageBuffers.clear();
        shaderStorageBuffersMemory.clear();

        std::default_random_engine rndEngine((unsigned)time(nullptr));
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

        // Create an initial set of particles (we could likely just as easily read this from a stream)
        std::vector<Particle> particles(PARTICLE_COUNT);
        for (auto &particle : particles)
        {
            float r = 0.25f * sqrtf(rndDist(rndEngine));
            float theta = rndDist(rndEngine) * 2.0f * M_PI;
            float x = r * cosf(theta) * HEIGHT / WIDTH;
            float y = r * sinf(theta);
            particle.position = glm::vec2(x, y);
            particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
            particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
        }

        vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

        // Create our staging buffer, used to transfer the particles to the GPU
        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, particles.data(), (size_t)bufferSize);
        stagingBufferMemory.unmapMemory();

        // Create and initialize all of our Particle buffers that will remain on the GPU, non-host visible or coherent
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            // Copy initial particle data to all storage buffers
            vk::raii::Buffer shaderStorageBufferTemp({});
            vk::raii::DeviceMemory shaderStorageBufferMemoryTemp({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, shaderStorageBufferTemp, shaderStorageBufferMemoryTemp);
            copyBuffer(stagingBuffer, shaderStorageBufferTemp, bufferSize);
            shaderStorageBuffers.emplace_back(std::move(shaderStorageBufferTemp));
            shaderStorageBuffersMemory.emplace_back(std::move(shaderStorageBufferMemoryTemp));
        }
    }

    void createImageViews()
    {
        assert(swapChainImageViews.empty());

        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = swapChainSurfaceFormat.format,
            .components = {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity}, // CHECK: Unsure if this is necessary or if this is the default?
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };

        for (auto image : swapChainImages)
        {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(device, imageViewCreateInfo);
        }
    }

    [[nodiscard]] vk::raii::ImageView createImageView(vk::raii::Image &image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
    {
        vk::ImageViewCreateInfo viewInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {aspectFlags, 0, mipLevels, 0, 1}};

        return vk::raii::ImageView(device, viewInfo);
    }

    void createComputeDescriptorSetLayout()
    {
        std::array bindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };

        computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
    }

    void createComputeDescriptorSets()
    {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *computeDescriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()};

        computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject),
            };

            vk::DescriptorBufferInfo storageBufferInfoLastFrame{
                .buffer = shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT],
                .offset = 0,
                .range = sizeof(Particle) * PARTICLE_COUNT,
            };

            vk::DescriptorBufferInfo storageBufferInfoCurrentFrame{
                .buffer = shaderStorageBuffers[i],
                .offset = 0,
                .range = sizeof(Particle) * PARTICLE_COUNT,
            };

            std::array descriptorWrites{
                vk::WriteDescriptorSet{
                    .dstSet = computeDescriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet = computeDescriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &storageBufferInfoLastFrame},
                vk::WriteDescriptorSet{
                    .dstSet = computeDescriptorSets[i],
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &storageBufferInfoCurrentFrame,
                },
            };

            device.updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createDescriptorPool()
    {
        std::array poolSize{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 2),
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
            .pPoolSizes = poolSize.data()};
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createComputePipeline()
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = shaderModule,
            .pName = "compMain",
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*computeDescriptorSetLayout,
        };

        computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::ComputePipelineCreateInfo pipelineInfo{
            .stage = computeShaderStageInfo,
            .layout = *computePipelineLayout,
        };

        computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createLogicalDevice()
    {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports both graphics and present
        for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
        {
            if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
            {
                // found a queue family that supports both graphics and present
                queueIndex = qfpIndex;
                break;
            }
        }
        if (queueIndex == ~0)
        {
            throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
        }

        // query for Vulkan 1.3 features
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT, vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR> featureChain = {
            {.features = {.samplerAnisotropy = true}},            // vk::PhysicalDeviceFeatures2
            {.synchronization2 = true, .dynamicRendering = true}, // vk::PhysicalDeviceVulkan13Features
            {.extendedDynamicState = true},                       // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
            {.timelineSemaphore = true},                          // vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR
        };

        // create a Device
        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
        vk::DeviceCreateInfo deviceCreateInfo{.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
                                              .queueCreateInfoCount = 1,
                                              .pQueueCreateInfos = &deviceQueueCreateInfo,
                                              .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
                                              .ppEnabledExtensionNames = requiredDeviceExtension.data()};

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        queue = std::make_unique<vk::raii::Queue>(device, queueIndex, 0);
    }

    void pickPhysicalDevice()
    {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto devIter = std::ranges::find_if(
            devices,
            [&](auto const &device)
            {
                // Check if the device supports the Vulkan 1.3 API version
                bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

                // Check if any of the queue families support graphics operations
                auto queueFamilies = device.getQueueFamilyProperties();
                bool supportsGraphics =
                    std::ranges::any_of(queueFamilies, [](auto const &qfp)
                                        { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

                // Check if all required device extensions are available
                auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
                bool supportsAllRequiredExtensions =
                    std::ranges::all_of(requiredDeviceExtension,
                                        [&availableDeviceExtensions](auto const &requiredDeviceExtension)
                                        {
                                            return std::ranges::any_of(availableDeviceExtensions,
                                                                       [requiredDeviceExtension](auto const &availableDeviceExtension)
                                                                       { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
                                        });

                auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
                bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

                return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
            });
        if (devIter != devices.end())
        {
            physicalDevice = *devIter;
        }
        else
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createSwapChain()
    {
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
        std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
        swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);

        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;

        vk::SwapchainCreateInfoKHR swapChainCreateInfo{
            .flags = vk::SwapchainCreateFlagsKHR(),
            .surface = surface,
            .minImageCount = minImageCount,
            .imageFormat = swapChainSurfaceFormat.format,
            .imageColorSpace = swapChainSurfaceFormat.colorSpace,
            .imageExtent = swapChainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = surfaceCapabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)),
            .clipped = true,
            .oldSwapchain = VK_NULL_HANDLE};

        swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
    {
        for (const auto &availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox)
            {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        for (const auto &availableFormat : availableFormats)
        {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
    }

    uint32_t findQueueFamilies(vk::raii::PhysicalDevice physicalDevice)
    {
        // find the index of the first queue family that supports graphics
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty =
            std::find_if(queueFamilyProperties.begin(),
                         queueFamilyProperties.end(),
                         [](vk::QueueFamilyProperties const &qfp)
                         { return qfp.queueFlags & vk::QueueFlagBits::eGraphics; });

        return static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers)
            return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
            .messageSeverity = severityFlags,
            .messageType = messageTypeFlags,
            .pfnUserCallback = &debugCallback};
        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    void createInstance()
    {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion14};

        std::vector<char const *> requiredLayers;
        if (enableValidationLayers)
        {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        auto layerProperties = context.enumerateInstanceLayerProperties();
        if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const &requiredLayer)
                                { return std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty)
                                                              { return strcmp(layerProperty.layerName, requiredLayer) == 0; }); }))
        {
            throw std::runtime_error("One or more required layers are not supported!");
        }

        auto requiredExtensions = getRequiredExtensions();

        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (auto const &requiredExtension : requiredExtensions)
        {
            if (std::ranges::none_of(extensionProperties,
                                     [requiredExtension](auto const &extensionProperty)
                                     { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
            {
                throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
            }
        }
        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames = requiredLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data()};

        instance = vk::raii::Instance(context, createInfo);

        auto extensionProps = context.enumerateInstanceExtensionProperties();

        std::cout << "available extensions:\n";

        for (const auto &extension : extensionProps)
        {
            std::cout << '\t' << extension.extensionName << '\n';
        }
    }

    void createCommandPool()
    {
        vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = queueIndex};
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory)
    {
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = mipLevels,
            .arrayLayers = 1,
            .samples = numSamples,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };

        image = vk::raii::Image(device, imageInfo);

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};

        imageMemory = vk::raii::DeviceMemory(device, allocInfo);
        image.bindMemory(imageMemory, 0);
    }

    void copyBufferToImage(const vk::raii::Buffer &buffer, vk::raii::Image &image, uint32_t width, uint32_t height)
    {
        auto commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1}};

        commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});

        endSingleTimeCommands(*commandBuffer);
    }

    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands()
    {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1};

        std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(device.allocateCommandBuffers(allocInfo).front()));
        vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        commandBuffer->begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(const vk::raii::CommandBuffer &commandBuffer)
    {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer};
        queue->submit(submitInfo, nullptr);
        queue->waitIdle();
    }

    void createUniformBuffers()
    {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter && (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size)
    {
        std::unique_ptr<vk::raii::CommandBuffer> commandCopyBuffer = beginSingleTimeCommands();
        commandCopyBuffer->copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
        endSingleTimeCommands(*commandCopyBuffer);
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory)
    {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive};
        buffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(bufferMemory, 0);
    }

    void createCommandBuffers()
    {
        commandBuffers.clear();

        vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};

        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void createComputeCommandBuffers()
    {
        computeCommandBuffers.clear();

        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
        };

        computeCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void recordComputeCommandBuffer()
    {
        computeCommandBuffers[currentFrame].begin({});
        computeCommandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
        computeCommandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, {computeDescriptorSets[currentFrame]}, {});
        computeCommandBuffers[currentFrame].dispatch(PARTICLE_COUNT / 256, 1, 1);
        computeCommandBuffers[currentFrame].end();
    }

    void recordCommandBuffer(uint32_t imageIndex)
    {
        commandBuffers[currentFrame].reset();
        commandBuffers[currentFrame].begin({});

        transition_image_layout(
            swapChainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);

        vk::RenderingAttachmentInfo attachmentInfo = {
            .imageView = swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor,
        };

        vk::RenderingInfo renderingInfo = {
            .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentInfo,
        };

        commandBuffers[currentFrame].beginRendering(renderingInfo);
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[currentFrame].bindVertexBuffers(0, {*shaderStorageBuffers[currentFrame]}, {0});
        commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
        commandBuffers[currentFrame].draw(PARTICLE_COUNT, 1, 0, 0);
        commandBuffers[currentFrame].endRendering();

        transition_image_layout(
            swapChainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe,
            vk::ImageAspectFlagBits::eColor);

        commandBuffers[currentFrame].end();
    }

    void drawFrame()
    {

        auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, nullptr, *inFlightFences[currentFrame]);
        while (vk::Result::eTimeout == device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
            ;

        device.resetFences(*inFlightFences[currentFrame]);

        uint64_t computeWaitValue = timelineValue;
        uint64_t computeSignalValue = ++timelineValue;
        uint64_t graphicsWaitValue = computeSignalValue;
        uint64_t graphicsSignalValue = ++timelineValue;

        updateUniformBuffer(currentFrame);

        {
            recordComputeCommandBuffer();

            vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
                .waitSemaphoreValueCount = 1,
                .pWaitSemaphoreValues = &computeWaitValue,
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues = &computeSignalValue,
            };

            vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eComputeShader};

            vk::SubmitInfo computeSubmitInfo{
                .pNext = &computeTimelineInfo,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*semaphore,
                .pWaitDstStageMask = waitStages,
                .commandBufferCount = 1,
                .pCommandBuffers = &*computeCommandBuffers[currentFrame],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &*semaphore,
            };

            queue->submit(computeSubmitInfo, nullptr);
        }
        {
            recordCommandBuffer(imageIndex);

            vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eVertexInput;
            vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
                .waitSemaphoreValueCount = 1,
                .pWaitSemaphoreValues = &graphicsWaitValue,
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues = &graphicsSignalValue,
            };

            vk::SubmitInfo graphicsSubmitInfo{
                .pNext = &graphicsTimelineInfo,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &*semaphore,
                .pWaitDstStageMask = &waitStage,
                .commandBufferCount = 1,
                .pCommandBuffers = &*commandBuffers[currentFrame],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &*semaphore,
            };

            queue->submit(graphicsSubmitInfo, nullptr);

            vk::SemaphoreWaitInfo waitInfo{
                .semaphoreCount = 1,
                .pSemaphores = &*semaphore,
                .pValues = &graphicsSignalValue,
            };

            while (vk::Result::eTimeout == device.waitSemaphores(waitInfo, UINT64_MAX))
                ;

            vk::PresentInfoKHR presentInfo{
                .waitSemaphoreCount = 0, // Q: No binary semaphores needed. Is this better, why?
                .pWaitSemaphores = nullptr,
                .swapchainCount = 1,
                .pSwapchains = &*swapChain,
                .pImageIndices = &imageIndex,
            };

            result = queue->presentKHR(presentInfo);
            switch (result)
            {
            case vk::Result::eSuccess:
                break;
            case vk::Result::eSuboptimalKHR:
                std::cout << "vk::Queue::presentKHR returned vk::Result::eSuboptimalKHR!\n";
                break;
            default:
                break;
            }

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
    }

    void createGraphicsPipeline()
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shaderModule,
            .pName = "vertMain",
        };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shaderModule,
            .pName = "fragMain",
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = {
            vertShaderStageInfo,
            fragShaderStageInfo,
        };

        auto bindingDescription = Particle::getBindingDescription();
        auto attributeDescriptions = Particle::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data(),
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::ePointList, // NOTE: We're not rendering verts, just points.
            .primitiveRestartEnable = vk::False,
        };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1,
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .depthBiasSlopeFactor = 1.0f, // CHECK: This may need to be removed
            .lineWidth = 1.0f,
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False,
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
            {.stageCount = 2,
             .pStages = shaderStages,
             .pVertexInputState = &vertexInputInfo,
             .pInputAssemblyState = &inputAssembly,
             .pViewportState = &viewportState,
             .pRasterizationState = &rasterizer,
             .pMultisampleState = &multisampling,
             .pColorBlendState = &colorBlending,
             .pDynamicState = &dynamicState,
             .layout = pipelineLayout,
             .renderPass = nullptr},
            {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format}};

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        UniformBufferObject ubo{};
        ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.0f;
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void createSyncObjects()
    {
        inFlightFences.clear();

        vk::SemaphoreTypeCreateInfo semaphoreType{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        };
        semaphore = vk::raii::Semaphore(device, {.pNext = &semaphoreType});
        timelineValue = 0;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::FenceCreateInfo fenceInfo{};
            inFlightFences.emplace_back(device, fenceInfo);
        }
    }

    void transition_image_layout(
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlags2 srcAccessMask,
        vk::AccessFlags2 dstAccessMask,
        vk::PipelineStageFlags2 srcStageMask,
        vk::PipelineStageFlags2 dstStageMask,
        vk::ImageAspectFlags imageAspectFlags)
    {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = imageAspectFlags,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1}};

        vk::DependencyInfo dependencyInfo = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier};

        commandBuffers[currentFrame].pipelineBarrier2(dependencyInfo);
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *)
    {
        std::cerr << "validation layer: type" << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;

        return vk::False;
    }

    std::vector<const char *> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers)
        {
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }

        return extensions;
    }

    void createSurface()
    {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
        {
            throw std::runtime_error("failed to create window surface!");
        }

        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
    {
        vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size() * sizeof(char), .pCode = reinterpret_cast<const uint32_t *>(code.data())};
        vk::raii::ShaderModule shaderModule{device, createInfo};
        return shaderModule;
    }

    static std::vector<char> readFile(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("failed to open file!");
        }

        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

        file.close();

        return buffer;
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
            double currentTime = glfwGetTime();
            lastFrameTime = (currentTime - lastTime) * 1000.0;
            lastTime = currentTime;
        }

        device.waitIdle();
    }

    void cleanup() const
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main()
{

    try
    {
        HelloTriangleApplication app;
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}