
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

const int WindowWidth = 800;
const int WindowHeight = 600;
const std::size_t MaxFramesInFlight = 2;

#ifdef NDEBUG
const bool EnableVkValidationLayers = false;
#else
const bool EnableVkValidationLayers = true;
#endif

struct UniformBufferObject
{
    alignas(16) glm::mat4 m_model;
    alignas(16) glm::mat4 m_view;
    alignas(16) glm::mat4 m_proj;
};

struct Vertex
{
    glm::vec2 m_position;
    glm::vec3 m_color;

    static VkVertexInputBindingDescription get_binding_description()
    {
        VkVertexInputBindingDescription binding_description = {};
        binding_description.binding = 0;
        binding_description.stride = sizeof(Vertex);
        binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding_description;
    }

    static std::array<VkVertexInputAttributeDescription, 2> get_attribute_descriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions = {};

        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(Vertex, m_position);

        attribute_descriptions[1].binding = 0;
        attribute_descriptions[1].location = 1;
        attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descriptions[1].offset = offsetof(Vertex, m_color);

        return attribute_descriptions;
    }
};

const std::vector<const char*> ValidationLayers =
{
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> DeviceExtensions =
{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

const std::vector<Vertex> Vertices =
{
    { { -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
    { {  0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },
    { {  0.5f,  0.5f }, { 0.0f, 0.0f, 1.0f } },
    { { -0.5f,  0.5f }, { 1.0f, 1.0f, 1.0f } }
};

const std::vector<std::uint16_t> Indices =
{
    0, 1, 2,
    2, 3, 0
};

#define EXPECT_VK_SUCCESS(expression) \
    { \
        const VkResult result = (expression); \
        if (result != VK_SUCCESS) \
            throw std::runtime_error(std::string(#expression " failed with Vulkan error code ") + std::to_string(result)); \
    }

std::string make_version_string(const std::uint32_t version)
{
    std::stringstream sstr;
    sstr << VK_VERSION_MAJOR(version) << ".";
    sstr << VK_VERSION_MINOR(version) << ".";
    sstr << VK_VERSION_PATCH(version);
    return sstr.str();
}

class HelloTriangleApplication
{
  public:
    void run()
    {
        create_window();
        init_vulkan();
        main_loop();
        cleanup_vulkan();
        destroy_window();
    }

  private:
    GLFWwindow*                     m_window;

    VkInstance                      m_instance;
    VkDebugUtilsMessengerEXT        m_debug_messenger;
    VkSurfaceKHR                    m_surface;

    VkPhysicalDevice                m_physical_device;
    VkDevice                        m_device;

    VkQueue                         m_graphics_queue;
    VkQueue                         m_present_queue;

    VkSwapchainKHR                  m_swap_chain;
    VkFormat                        m_swap_chain_surface_format;
    VkExtent2D                      m_swap_chain_extent;
    std::vector<VkImage>            m_swap_chain_images;
    std::vector<VkImageView>        m_swap_chain_image_views;
    std::vector<VkFramebuffer>      m_swap_chain_framebuffers;

    VkRenderPass                    m_render_pass;
    VkDescriptorSetLayout           m_descriptor_set_layout;
    VkPipelineLayout                m_pipeline_layout;
    VkPipeline                      m_graphics_pipeline;

    VkCommandPool                   m_command_pool;
    VkCommandPool                   m_transient_command_pool;

    VkBuffer                        m_vertex_buffer;
    VkDeviceMemory                  m_vertex_buffer_memory;
    VkBuffer                        m_index_buffer;
    VkDeviceMemory                  m_index_buffer_memory;

    std::vector<VkBuffer>           m_uniform_buffers;
    std::vector<VkDeviceMemory>     m_uniform_buffers_memory;

    VkDescriptorPool                m_descriptor_pool;
    std::vector<VkDescriptorSet>    m_descriptor_sets;

    std::vector<VkCommandBuffer>    m_command_buffers;

    std::vector<VkSemaphore>        m_image_available_semaphores;
    std::vector<VkSemaphore>        m_render_finished_semaphores;
    std::vector<VkFence>            m_in_flight_fences;

    bool                            m_framebuffer_resized = false;

    void create_window()
    {
        std::cout << "Creating window..." << std::endl;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        m_window = glfwCreateWindow(WindowWidth, WindowHeight, "Vulkan Tutorial", nullptr, nullptr);

        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, framebuffer_resize_callback);
    }

    static void framebuffer_resize_callback(GLFWwindow* window, const int width, const int height)
    {
        HelloTriangleApplication* app =
            reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));

        app->m_framebuffer_resized = true;
    }

    void init_vulkan()
    {
        query_vk_instance_extensions();
        create_vk_instance();
        setup_vk_debug_messenger();
        create_vk_surface();
        pick_vk_physical_device();
        create_vk_logical_device();
        create_vk_swap_chain();
        create_vk_swap_chain_image_views();
        create_vk_render_pass();
        create_vk_descriptor_set_layout();
        create_vk_graphics_pipeline();
        create_vk_framebuffers();
        create_vk_command_pools();
        create_vk_vertex_buffer();
        create_vk_index_buffer();
        create_vk_uniform_buffers();
        create_vk_descriptor_pool();
        create_vk_descriptor_sets();
        create_vk_command_buffers();
        create_vk_sync_objects();
    }

    static void query_vk_instance_extensions()
    {
        std::cout << "Querying Vulkan instance extensions..." << std::endl;

        std::uint32_t extension_count;
        if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Failed to enumerate Vulkan instance extensions");

        std::vector<VkExtensionProperties> extension_props(extension_count);
        EXPECT_VK_SUCCESS(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extension_props.data()));

        if (extension_count > 0)
        {
            std::cout << extension_count << " instance extension(s) found:" << std::endl;
            for (const VkExtensionProperties& ext : extension_props)
            {
                std::cout << "    " << ext.extensionName << " (version " << ext.specVersion
                    << ", or " << make_version_string(ext.specVersion) << ")" << std::endl;
            }
        }
        else std::cout << "No instance extension found." << std::endl;
    }

    static bool check_vk_validation_layer_support()
    {
        std::uint32_t layer_count;
        if (vkEnumerateInstanceLayerProperties(&layer_count, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Failed to enumerate Vulkan instance validation layers");

        std::vector<VkLayerProperties> layer_props(layer_count);
        EXPECT_VK_SUCCESS(vkEnumerateInstanceLayerProperties(&layer_count, layer_props.data()));

        for (const char* layer : ValidationLayers)
        {
            bool found = false;

            for (const VkLayerProperties& candidate_layer_prop : layer_props)
            {
                if (std::strcmp(layer, candidate_layer_prop.layerName) == 0)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
                return false;
        }

        return true;
    }

    static std::vector<const char*> get_required_instance_extensions()
    {
        std::uint32_t glfw_extension_count;
        const char** glfw_extension_names = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

        std::vector<const char*> extensions(glfw_extension_names, glfw_extension_names + glfw_extension_count);

        if (EnableVkValidationLayers)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }

    // Conforms to the PFN_vkDebugUtilsMessengerCallbackEXT callback signature.
    static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT      message_severity,
        VkDebugUtilsMessageTypeFlagsEXT             message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        void*                                       user_data)
    {
        std::string severity;
        switch (message_severity)
        {
          case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            severity = "Diagnostics";
            break;
          case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            severity = "Info";
            break;
          case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            severity = "Warning";
            break;
          case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            severity = "Error";
            break;
          default:
            severity = "Unknown Severity";
            break;
        }

        std::cerr << "[Validation Layer] [" << severity << "] " << callback_data->pMessage << std::endl;

        return VK_FALSE;
    }

    void create_vk_instance()
    {
        std::cout << "Creating Vulkan instance..." << std::endl;

        if (EnableVkValidationLayers && !check_vk_validation_layer_support())
            throw std::runtime_error("One or more requested validation layers are not supported");

        const std::vector<const char*> instance_extensions = get_required_instance_extensions();

        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Vulkan Tutorial";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "No Engine";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        if (EnableVkValidationLayers)
        {
            create_info.enabledLayerCount = static_cast<std::uint32_t>(ValidationLayers.size());
            create_info.ppEnabledLayerNames = ValidationLayers.data();
        }
        else
        {
            create_info.enabledLayerCount = 0;
            create_info.ppEnabledLayerNames = nullptr;
        }
        create_info.enabledExtensionCount = static_cast<std::uint32_t>(instance_extensions.size());
        create_info.ppEnabledExtensionNames = instance_extensions.data();

        if (vkCreateInstance(&create_info, nullptr, &m_instance) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan instance");
    }

    void setup_vk_debug_messenger()
    {
        if (EnableVkValidationLayers)
        {
            std::cout << "Setting up Vulkan debug messenger..." << std::endl;

            auto vkCreateDebugUtilsMessengerEXTFn =
                reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
            if (vkCreateDebugUtilsMessengerEXTFn == nullptr)
                throw std::runtime_error("Failed to load vkCreateDebugUtilsMessengerEXT() function");

            VkDebugUtilsMessengerCreateInfoEXT create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            create_info.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            create_info.messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            create_info.pfnUserCallback = vk_debug_callback;
            create_info.pUserData = nullptr;

            if (vkCreateDebugUtilsMessengerEXTFn(m_instance, &create_info, nullptr, &m_debug_messenger) != VK_SUCCESS)
                throw std::runtime_error("Failed to create Vulkan debug messenger");
        }
    }

    void create_vk_surface()
    {
        if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan window surface");
    }

    struct QueueFamilyIndices
    {
        std::optional<std::uint32_t> m_graphics_family;
        std::optional<std::uint32_t> m_present_family;

        bool is_complete() const
        {
            return m_graphics_family.has_value() && m_present_family.has_value();
        }
    };

    QueueFamilyIndices find_vk_queue_families(VkPhysicalDevice physical_device) const
    {
        QueueFamilyIndices indices;

        std::uint32_t queue_family_count;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_family_props(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_family_props.data());

        for (std::uint32_t i = 0, e = static_cast<std::uint32_t>(queue_family_props.size()); i < e; ++i)
        {
            const VkQueueFamilyProperties& queue_family = queue_family_props[i];

            if (queue_family.queueCount > 0 && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.m_graphics_family = i;

            VkBool32 is_present_supported = false;
            if (vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, m_surface, &is_present_supported) != VK_SUCCESS)
                throw std::runtime_error("Failed to query Vulkan device surface support");

            if (queue_family.queueCount > 0 && is_present_supported)
                indices.m_present_family = i;

            if (indices.is_complete())
                break;
        }

        return indices;
    }

    static bool check_vk_device_extension_support(VkPhysicalDevice physical_device)
    {
        std::uint32_t extension_count;
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);

        std::vector<VkExtensionProperties> available_extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_extensions.data());

        std::set<std::string> required_extensions(std::cbegin(DeviceExtensions), std::cend(DeviceExtensions));

        for (const VkExtensionProperties extension : available_extensions)
            required_extensions.erase(extension.extensionName);

        return required_extensions.empty();
    }

    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR        m_capabilities;
        std::vector<VkSurfaceFormatKHR> m_formats;
        std::vector<VkPresentModeKHR>   m_present_modes;
    };

    SwapChainSupportDetails query_vk_swap_chain_support(VkPhysicalDevice physical_device) const
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, m_surface, &details.m_capabilities);

        std::uint32_t format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, m_surface, &format_count, nullptr);

        if (format_count > 0)
        {
            details.m_formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, m_surface, &format_count, details.m_formats.data());
        }

        std::uint32_t present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            physical_device,
            m_surface,
            &present_mode_count,
            nullptr);

        if (present_mode_count > 0)
        {
            details.m_present_modes.resize(present_mode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                physical_device,
                m_surface,
                &present_mode_count,
                details.m_present_modes.data());
        }

        return details;
    }

    static VkSurfaceFormatKHR choose_vk_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats)
    {
        if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED)
        {
            // Surface has no preferred format.
            return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        }

        for (const VkSurfaceFormatKHR& format : available_formats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return format;
        }

        // Just settle for the first one.
        assert(!available_formats.empty());
        return available_formats[0];
    }

    static VkPresentModeKHR choose_vk_swap_present_mode(const std::vector<VkPresentModeKHR>& available_modes)
    {
        VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;

        for (const VkPresentModeKHR& mode : available_modes)
        {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                return mode;
            else if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
                best_mode = mode;
        }

        return best_mode;
    }

    VkExtent2D choose_vk_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
            return capabilities.currentExtent;

        int window_width, window_height;
        glfwGetFramebufferSize(m_window, &window_width, &window_height);

        VkExtent2D actual_extent =
        {
            static_cast<std::uint32_t>(window_width),
            static_cast<std::uint32_t>(window_height)
        };

        actual_extent.width =
            std::clamp(
                actual_extent.width,
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width);

        actual_extent.height =
            std::clamp(
                actual_extent.height,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height);

        return actual_extent;
    }

    bool is_vk_device_suitable(VkPhysicalDevice physical_device)
    {
        const QueueFamilyIndices indices = find_vk_queue_families(physical_device);

        const bool extensions_supported = check_vk_device_extension_support(physical_device);

        bool swap_chain_adequate = false;
        if (extensions_supported)
        {
            const SwapChainSupportDetails swap_chain_support = query_vk_swap_chain_support(physical_device);
            swap_chain_adequate = !swap_chain_support.m_formats.empty() && !swap_chain_support.m_present_modes.empty();
        }

        return indices.is_complete() && extensions_supported && swap_chain_adequate;
    }

    void pick_vk_physical_device()
    {
        std::cout << "Picking Vulkan physical device..." << std::endl;

        std::uint32_t device_count;
        if (vkEnumeratePhysicalDevices(m_instance, &device_count, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Failed to enumerate Vulkan physical devices");

        if (device_count == 0)
            throw std::runtime_error("Failed to find a GPU with Vulkan support");

        std::vector<VkPhysicalDevice> devices(device_count);
        EXPECT_VK_SUCCESS(vkEnumeratePhysicalDevices(m_instance, &device_count, devices.data()));

        std::cout << device_count << " device(s) found:" << std::endl;

        for (const VkPhysicalDevice& device : devices)
        {
            VkPhysicalDeviceProperties device_props;
            vkGetPhysicalDeviceProperties(device, &device_props);

            std::cout << "    " << device_props.deviceName << " (driver version: "
                << make_version_string(device_props.driverVersion) << ")" << std::endl;
        }

        m_physical_device = VK_NULL_HANDLE;
        for (const VkPhysicalDevice& device : devices)
        {
            if (is_vk_device_suitable(device))
            {
                m_physical_device = device;
                break;
            }
        }

        if (m_physical_device == VK_NULL_HANDLE)
            throw std::runtime_error("Failed to find a suitable Vulkan device");
    }

    void create_vk_logical_device()
    {
        std::cout << "Creating Vulkan logical device..." << std::endl;

        const QueueFamilyIndices indices = find_vk_queue_families(m_physical_device);
        assert(indices.is_complete());

        float queue_priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;

        const std::set<std::uint32_t> unique_queue_families =
        {
            indices.m_graphics_family.value(),
            indices.m_present_family.value()
        };

        for (const std::uint32_t queue_family : unique_queue_families)
        {
            VkDeviceQueueCreateInfo queue_create_info = {};
            queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_create_info.queueFamilyIndex = indices.m_graphics_family.value();
            queue_create_info.queueCount = 1;
            queue_create_info.pQueuePriorities = &queue_priority;
            queue_create_infos.push_back(queue_create_info);
        }

        VkPhysicalDeviceFeatures device_features = {};

        VkDeviceCreateInfo device_create_info = {};
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.queueCreateInfoCount = static_cast<std::uint32_t>(queue_create_infos.size());
        device_create_info.pQueueCreateInfos = queue_create_infos.data();
        if (EnableVkValidationLayers)
        {
            device_create_info.enabledLayerCount = static_cast<std::uint32_t>(ValidationLayers.size());
            device_create_info.ppEnabledLayerNames = ValidationLayers.data();
        }
        else
        {
            device_create_info.enabledLayerCount = 0;
            device_create_info.ppEnabledLayerNames = nullptr;
        }
        device_create_info.enabledExtensionCount = static_cast<std::uint32_t>(DeviceExtensions.size());
        device_create_info.ppEnabledExtensionNames = DeviceExtensions.data();
        device_create_info.pEnabledFeatures = &device_features;

        if (vkCreateDevice(m_physical_device, &device_create_info, nullptr, &m_device) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan logical device");

        vkGetDeviceQueue(m_device, indices.m_graphics_family.value(), 0, &m_graphics_queue);
        vkGetDeviceQueue(m_device, indices.m_present_family.value(), 0, &m_present_queue);
    }

    void create_vk_swap_chain()
    {
        std::cout << "Creating Vulkan swap chain..." << std::endl;

        const SwapChainSupportDetails swap_chain_support = query_vk_swap_chain_support(m_physical_device);

        const VkSurfaceFormatKHR surface_format = choose_vk_swap_surface_format(swap_chain_support.m_formats);
        const VkPresentModeKHR present_mode = choose_vk_swap_present_mode(swap_chain_support.m_present_modes);
        const VkExtent2D extent = choose_vk_swap_extent(swap_chain_support.m_capabilities);

        std::uint32_t image_count = swap_chain_support.m_capabilities.minImageCount + 1;
        if (swap_chain_support.m_capabilities.maxImageCount > 0)
            image_count = std::min(image_count, swap_chain_support.m_capabilities.maxImageCount);

        const QueueFamilyIndices indices = find_vk_queue_families(m_physical_device);
        assert(indices.is_complete());

        const std::uint32_t queue_family_indices[] =
        {
            indices.m_graphics_family.value(),
            indices.m_present_family.value()
        };

        VkSwapchainCreateInfoKHR create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = m_surface;
        create_info.minImageCount = image_count;
        create_info.imageFormat = surface_format.format;
        create_info.imageColorSpace = surface_format.colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (indices.m_graphics_family == indices.m_present_family)
        {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            create_info.queueFamilyIndexCount = 0;
            create_info.pQueueFamilyIndices = nullptr;
        }
        else
        {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queue_family_indices;
        }
        create_info.preTransform = swap_chain_support.m_capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;
        create_info.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(m_device, &create_info, nullptr, &m_swap_chain) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan swap chain");

        m_swap_chain_surface_format = surface_format.format;
        m_swap_chain_extent = extent;

        vkGetSwapchainImagesKHR(m_device, m_swap_chain, &image_count, nullptr);
        m_swap_chain_images.resize(image_count);
        vkGetSwapchainImagesKHR(m_device, m_swap_chain, &image_count, m_swap_chain_images.data());
    }

    void create_vk_swap_chain_image_views()
    {
        std::cout << "Creating Vulkan swap chain image views..." << std::endl;

        m_swap_chain_image_views.resize(m_swap_chain_images.size());

        for (std::size_t i = 0; i < m_swap_chain_images.size(); ++i)
        {
            VkImageViewCreateInfo create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            create_info.image = m_swap_chain_images[i];
            create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            create_info.format = m_swap_chain_surface_format;
            create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            create_info.subresourceRange.baseMipLevel = 0;
            create_info.subresourceRange.levelCount = 1;
            create_info.subresourceRange.baseArrayLayer = 0;
            create_info.subresourceRange.layerCount = 1;

            if (vkCreateImageView(m_device, &create_info, nullptr, &m_swap_chain_image_views[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create Vulkan image views");
        }
    }

    // Would have liked to use std::byte instead of char but it required too many contortions.
    static std::vector<char> read_file(const std::string& filepath)
    {
        std::ifstream file(filepath, std::ios::ate | std::ios::binary);
        file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        const std::size_t file_size = static_cast<std::size_t>(file.tellg());
        std::vector<char> file_content(file_size);

        file.seekg(0);
        file.read(file_content.data(), file_size);

        return file_content;
    }

    VkShaderModule create_vk_shader_module(const std::vector<char>& code) const
    {
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size();
        create_info.pCode = reinterpret_cast<const std::uint32_t*>(code.data());

        VkShaderModule shader_module;
        if (vkCreateShaderModule(m_device, &create_info, nullptr, &shader_module) != VK_SUCCESS)
            throw std::runtime_error("Failed to create shader module");

        return shader_module;
    }

    void create_vk_render_pass()
    {
        std::cout << "Creating Vulkan render pass..." << std::endl;

        VkAttachmentDescription color_attachment = {};
        color_attachment.format = m_swap_chain_surface_format;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_ref = {};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo render_pass_create_info = {};
        render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_create_info.attachmentCount = 1;
        render_pass_create_info.pAttachments = &color_attachment;
        render_pass_create_info.subpassCount = 1;
        render_pass_create_info.pSubpasses = &subpass;
        render_pass_create_info.dependencyCount = 1;
        render_pass_create_info.pDependencies = &dependency;

        if (vkCreateRenderPass(m_device, &render_pass_create_info, nullptr, &m_render_pass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan render pass");
    }

    void create_vk_descriptor_set_layout()
    {
        std::cout << "Creating Vulkan descriptor set layout..." << std::endl;

        VkDescriptorSetLayoutBinding ubo_layout_binding = {};
        ubo_layout_binding.binding = 0;
        ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubo_layout_binding.descriptorCount = 1;
        ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        ubo_layout_binding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create_info.bindingCount = 1;
        create_info.pBindings = &ubo_layout_binding;

        if (vkCreateDescriptorSetLayout(m_device, &create_info, nullptr, &m_descriptor_set_layout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan descriptor set layout");
    }

    void create_vk_graphics_pipeline()
    {
        std::cout << "Creating Vulkan graphics pipeline..." << std::endl;

        const std::vector<char> vert_shader_code = read_file("shaders/vert.spv");
        const std::vector<char> frag_shader_code = read_file("shaders/frag.spv");

        const VkShaderModule vert_shader_module = create_vk_shader_module(vert_shader_code);
        const VkShaderModule frag_shader_module = create_vk_shader_module(frag_shader_code);

        VkPipelineShaderStageCreateInfo vert_shader_stage_info = {};
        vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vert_shader_stage_info.module = vert_shader_module;
        vert_shader_stage_info.pName = "main";

        VkPipelineShaderStageCreateInfo frag_shader_stage_info = {};
        frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_shader_stage_info.module = frag_shader_module;
        frag_shader_stage_info.pName = "main";

        const VkPipelineShaderStageCreateInfo shader_stages[] =
        {
            vert_shader_stage_info,
            frag_shader_stage_info
        };

        const auto binding_description = Vertex::get_binding_description();
        const auto attribute_descriptions = Vertex::get_attribute_descriptions();

        VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
        vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 1;
        vertex_input_info.pVertexBindingDescriptions = &binding_description;
        vertex_input_info.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attribute_descriptions.size());
        vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(m_swap_chain_extent.width);
        viewport.height = static_cast<float>(m_swap_chain_extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = m_swap_chain_extent;

        VkPipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0F;
        rasterizer.depthBiasSlopeFactor = 0.0f;
        rasterizer.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState color_blend_attachment = {};
        color_blend_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment.blendEnable = VK_FALSE;
        color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo color_blending = {};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.logicOp = VK_LOGIC_OP_COPY;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;
        color_blending.blendConstants[0] = 0.0f;
        color_blending.blendConstants[1] = 0.0f;
        color_blending.blendConstants[2] = 0.0f;
        color_blending.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
        pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_create_info.setLayoutCount = 1;
        pipeline_layout_create_info.pSetLayouts = &m_descriptor_set_layout;
        pipeline_layout_create_info.pushConstantRangeCount = 0;
        pipeline_layout_create_info.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &m_pipeline_layout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan pipeline layout");

        VkGraphicsPipelineCreateInfo pipeline_create_info = {};
        pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_create_info.stageCount = 2;
        pipeline_create_info.pStages = shader_stages;
        pipeline_create_info.pVertexInputState = &vertex_input_info;
        pipeline_create_info.pInputAssemblyState = &input_assembly;
        pipeline_create_info.pViewportState = &viewport_state;
        pipeline_create_info.pRasterizationState = &rasterizer;
        pipeline_create_info.pMultisampleState = &multisampling;
        pipeline_create_info.pDepthStencilState = nullptr;
        pipeline_create_info.pColorBlendState = &color_blending;
        pipeline_create_info.pDynamicState = nullptr;
        pipeline_create_info.layout = m_pipeline_layout;
        pipeline_create_info.renderPass = m_render_pass;
        pipeline_create_info.subpass = 0;
        pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_create_info.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &m_graphics_pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan pipeline");

        vkDestroyShaderModule(m_device, frag_shader_module, nullptr);
        vkDestroyShaderModule(m_device, vert_shader_module, nullptr);
    }

    void create_vk_framebuffers()
    {
        std::cout << "Creating Vulkan framebuffers..." << std::endl;

        m_swap_chain_framebuffers.resize(m_swap_chain_image_views.size());

        for (std::size_t i = 0; i < m_swap_chain_image_views.size(); ++i)
        {
            const VkImageView attachments[] = { m_swap_chain_image_views[i] };

            VkFramebufferCreateInfo framebuffer_create_info = {};
            framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_create_info.renderPass = m_render_pass;
            framebuffer_create_info.attachmentCount = 1;
            framebuffer_create_info.pAttachments = attachments;
            framebuffer_create_info.width = m_swap_chain_extent.width;
            framebuffer_create_info.height = m_swap_chain_extent.height;
            framebuffer_create_info.layers = 1;

            if (vkCreateFramebuffer(m_device, &framebuffer_create_info, nullptr, &m_swap_chain_framebuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create Vulkan framebuffer");
        }
    }

    void create_vk_command_pool(const VkCommandPoolCreateFlags flags, VkCommandPool& command_pool)
    {
        const QueueFamilyIndices queue_family_indices = find_vk_queue_families(m_physical_device);

        VkCommandPoolCreateInfo pool_create_info = {};
        pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_create_info.queueFamilyIndex = queue_family_indices.m_graphics_family.value();
        pool_create_info.flags = flags;

        if (vkCreateCommandPool(m_device, &pool_create_info, nullptr, &command_pool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan command pool");
    }

    void create_vk_command_pools()
    {
        std::cout << "Creating Vulkan command pools..." << std::endl;

        create_vk_command_pool(0, m_command_pool);
        create_vk_command_pool(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_transient_command_pool);
    }

    std::uint32_t find_vk_memory_type(
        const std::uint32_t         type_filter,
        const VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties physical_mem_properties;
        vkGetPhysicalDeviceMemoryProperties(m_physical_device, &physical_mem_properties);

        for (std::uint32_t i = 0; i < physical_mem_properties.memoryTypeCount; ++i)
        {
            if ((type_filter & (1UL << i)) &&
                (physical_mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }

        throw std::runtime_error("Failed to find suitable Vulkan memory type");
    }

    void create_vk_buffer(
        const VkDeviceSize          size,
        const VkBufferUsageFlags    usage,
        const VkMemoryPropertyFlags properties,
        VkBuffer&                   buffer,
        VkDeviceMemory&             buffer_memory) const
    {
        VkBufferCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = size;
        create_info.usage = usage;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(m_device, &create_info, nullptr, &buffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan buffer");

        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(m_device, buffer, &mem_requirements);

        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = find_vk_memory_type(mem_requirements.memoryTypeBits, properties);

        if (vkAllocateMemory(m_device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate Vulkan buffer memory");

        if (vkBindBufferMemory(m_device, buffer, buffer_memory, 0) != VK_SUCCESS)
            throw std::runtime_error("Failed to bind Vulkan buffer memory to buffer");
    }
    
    void allocate_vk_command_buffers(
        const VkCommandPool         command_pool,
        const std::size_t           command_buffer_count,
        VkCommandBuffer*            command_buffers) const
    {
        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<std::uint32_t>(command_buffer_count);

        if (vkAllocateCommandBuffers(m_device, &alloc_info, command_buffers) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate Vulkan command buffer(s)");
    }

    void copy_vk_buffer(
        const VkBuffer              src_buffer,
        const VkBuffer              dst_buffer,
        const VkDeviceSize          size) const
    {
        VkCommandBuffer command_buffer;
        allocate_vk_command_buffers(m_transient_command_pool, 1, &command_buffer);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin recording Vulkan command buffer");

        VkBufferCopy copy_region = {};
        copy_region.srcOffset = 0;
        copy_region.dstOffset = 0;
        copy_region.size = size;
        vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

        vkEndCommandBuffer(command_buffer);

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        vkQueueSubmit(m_graphics_queue, 1, &submit_info, VK_NULL_HANDLE);

        vkQueueWaitIdle(m_graphics_queue);

        vkFreeCommandBuffers(m_device, m_transient_command_pool, 1, &command_buffer);
    }

    void create_vk_vertex_buffer()
    {
        std::cout << "Creating Vulkan vertex buffer..." << std::endl;

        const VkDeviceSize buffer_size = sizeof(Vertices[0]) * Vertices.size();

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        create_vk_buffer(
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        void* data;
        if (vkMapMemory(m_device, staging_buffer_memory, 0, buffer_size, 0, &data) != VK_SUCCESS)
            throw std::runtime_error("Failed to map Vulkan buffer memory to host address space");
        std::memcpy(data, Vertices.data(), static_cast<std::size_t>(buffer_size));
        vkUnmapMemory(m_device, staging_buffer_memory);

        create_vk_buffer(
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_vertex_buffer,
            m_vertex_buffer_memory);

        copy_vk_buffer(staging_buffer, m_vertex_buffer, buffer_size);

        vkDestroyBuffer(m_device, staging_buffer, nullptr);
        vkFreeMemory(m_device, staging_buffer_memory, nullptr);
    }

    void create_vk_index_buffer()
    {
        std::cout << "Creating Vulkan index buffer..." << std::endl;

        const VkDeviceSize buffer_size = sizeof(Indices[0]) * Indices.size();

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        create_vk_buffer(
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        void* data;
        if (vkMapMemory(m_device, staging_buffer_memory, 0, buffer_size, 0, &data) != VK_SUCCESS)
            throw std::runtime_error("Failed to map Vulkan buffer memory to host address space");
        std::memcpy(data, Indices.data(), static_cast<std::size_t>(buffer_size));
        vkUnmapMemory(m_device, staging_buffer_memory);

        create_vk_buffer(
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_index_buffer,
            m_index_buffer_memory);

        copy_vk_buffer(staging_buffer, m_index_buffer, buffer_size);

        vkDestroyBuffer(m_device, staging_buffer, nullptr);
        vkFreeMemory(m_device, staging_buffer_memory, nullptr);
    }

    void create_vk_uniform_buffers()
    {
        std::cout << "Creating Vulkan uniform buffers..." << std::endl;

        const VkDeviceSize buffer_size = sizeof(UniformBufferObject);

        m_uniform_buffers.resize(m_swap_chain_images.size());
        m_uniform_buffers_memory.resize(m_swap_chain_images.size());

        for (std::size_t i = 0; i < m_swap_chain_images.size(); ++i)
        {
            create_vk_buffer(
                buffer_size,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_uniform_buffers[i],
                m_uniform_buffers_memory[i]);
        }
    }

    void create_vk_descriptor_pool()
    {
        VkDescriptorPoolSize pool_size = {};
        pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_size.descriptorCount = static_cast<std::uint32_t>(m_swap_chain_images.size());

        VkDescriptorPoolCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create_info.poolSizeCount = 1;
        create_info.pPoolSizes = &pool_size;
        create_info.maxSets = static_cast<std::uint32_t>(m_swap_chain_images.size());

        if (vkCreateDescriptorPool(m_device, &create_info, nullptr, &m_descriptor_pool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan descriptor pool");
    }

    void create_vk_descriptor_sets()
    {
        m_descriptor_sets.resize(m_swap_chain_images.size());

        const std::vector<VkDescriptorSetLayout> layouts(m_swap_chain_images.size(), m_descriptor_set_layout);
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = m_descriptor_pool;
        alloc_info.descriptorSetCount = static_cast<std::uint32_t>(m_swap_chain_images.size());
        alloc_info.pSetLayouts = layouts.data();

        if (vkAllocateDescriptorSets(m_device, &alloc_info, m_descriptor_sets.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan descriptor sets");

        for (std::size_t i = 0; i < m_swap_chain_images.size(); ++i)
        {
            VkDescriptorBufferInfo buffer_info = {};
            buffer_info.buffer = m_uniform_buffers[i];
            buffer_info.offset = 0;
            buffer_info.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = m_descriptor_sets[i];
            descriptor_write.dstBinding = 0;
            descriptor_write.dstArrayElement = 0;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_write.descriptorCount = 1;
            descriptor_write.pBufferInfo = &buffer_info;
            vkUpdateDescriptorSets(m_device, 1, &descriptor_write, 0, nullptr);
        }
    }

    void create_vk_command_buffers()
    {
        std::cout << "Creating Vulkan command buffers..." << std::endl;

        m_command_buffers.resize(m_swap_chain_framebuffers.size());
        allocate_vk_command_buffers(m_command_pool, m_command_buffers.size(), m_command_buffers.data());

        for (std::size_t i = 0; i < m_command_buffers.size(); ++i)
        {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            if (vkBeginCommandBuffer(m_command_buffers[i], &begin_info) != VK_SUCCESS)
                throw std::runtime_error("Failed to begin recording Vulkan command buffer");

            const VkClearValue clear_color = { 0.0f, 0.0f, 0.0f, 1.0f };

            VkRenderPassBeginInfo render_pass_begin_info = {};
            render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            render_pass_begin_info.renderPass = m_render_pass;
            render_pass_begin_info.framebuffer = m_swap_chain_framebuffers[i];
            render_pass_begin_info.renderArea.offset = { 0, 0 };
            render_pass_begin_info.renderArea.extent = m_swap_chain_extent;
            render_pass_begin_info.clearValueCount = 1;
            render_pass_begin_info.pClearValues = &clear_color;

            vkCmdBeginRenderPass(m_command_buffers[i], &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(m_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics_pipeline);

            const VkBuffer vertex_buffers[] = { m_vertex_buffer };
            const VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(m_command_buffers[i], 0, 1, vertex_buffers, offsets);

            vkCmdBindIndexBuffer(m_command_buffers[i], m_index_buffer, 0, VK_INDEX_TYPE_UINT16);

            vkCmdBindDescriptorSets(
                m_command_buffers[i],
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipeline_layout,
                0,
                1,
                &m_descriptor_sets[i],
                0,
                nullptr);

            vkCmdDrawIndexed(m_command_buffers[i], static_cast<std::uint32_t>(Indices.size()), 1, 0, 0, 0);

            vkCmdEndRenderPass(m_command_buffers[i]);

            if (vkEndCommandBuffer(m_command_buffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to end recording Vulkan command buffer");
        }
    }

    void create_vk_sync_objects()
    {
        std::cout << "Creating Vulkan synchronization objects..." << std::endl;

        m_image_available_semaphores.resize(MaxFramesInFlight);
        m_render_finished_semaphores.resize(MaxFramesInFlight);
        m_in_flight_fences.resize(MaxFramesInFlight);

        VkSemaphoreCreateInfo semaphore_create_info = {};
        semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_create_info = {};
        fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            if (vkCreateSemaphore(m_device, &semaphore_create_info, nullptr, &m_image_available_semaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(m_device, &semaphore_create_info, nullptr, &m_render_finished_semaphores[i]) != VK_SUCCESS ||
                vkCreateFence(m_device, &fence_create_info, nullptr, &m_in_flight_fences[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create Vulkan synchronization objects");
        }
    }

    void main_loop()
    {
        std::cout << "Entering main loop..." << std::endl;

        std::size_t current_frame = 0;

        while (!glfwWindowShouldClose(m_window))
        {
            glfwPollEvents();
            draw_frame(current_frame);
            current_frame = (current_frame + 1) % MaxFramesInFlight;
        }
    }

    void draw_frame(const std::size_t current_frame)
    {
        vkWaitForFences(m_device, 1, &m_in_flight_fences[current_frame], VK_TRUE, std::numeric_limits<std::uint64_t>::max());

        std::uint32_t image_index;
        VkResult result =
            vkAcquireNextImageKHR(
                m_device,
                m_swap_chain,
                std::numeric_limits<std::uint64_t>::max(),
                m_image_available_semaphores[current_frame],
                VK_NULL_HANDLE,
                &image_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreate_vk_swap_chain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Failed to acquire Vulkan swap chain image");

        vkResetFences(m_device, 1, &m_in_flight_fences[current_frame]);

        update_vk_uniform_buffer(image_index);

        const VkSemaphore wait_semaphores[] = { m_image_available_semaphores[current_frame] };
        const VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        const VkSemaphore signal_semaphores[] = { m_render_finished_semaphores[current_frame] };

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = wait_semaphores;
        submit_info.pWaitDstStageMask = wait_stages;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &m_command_buffers[image_index];
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = signal_semaphores;

        if (vkQueueSubmit(m_graphics_queue, 1, &submit_info, m_in_flight_fences[current_frame]) != VK_SUCCESS)
            throw std::runtime_error("Failed to submit Vulkan draw command buffer");

        const VkSwapchainKHR swap_chains[] = { m_swap_chain };

        VkPresentInfoKHR present_info = {};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = signal_semaphores;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = swap_chains;
        present_info.pImageIndices = &image_index;
        present_info.pResults = nullptr;

        result = vkQueuePresentKHR(m_present_queue, &present_info);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_framebuffer_resized)
        {
            m_framebuffer_resized = false;
            recreate_vk_swap_chain();
        }
        else if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to present Vulkan swap chain image");
    }

    void update_vk_uniform_buffer(const std::uint32_t current_image)
    {
        static auto start_time = std::chrono::high_resolution_clock::now();

        const auto current_time = std::chrono::high_resolution_clock::now();
        const float time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - start_time).count();

        // Note: we're using the Y-up convention.
        UniformBufferObject ubo;
        ubo.m_model = glm::rotate(
            glm::mat4(1.0f),                    // initial transform
            time * glm::radians(90.0f),         // angle
            glm::vec3(0.0f, 1.0f, 0.0f));       // axis
        ubo.m_view = glm::lookAt(
            glm::vec3(2.0f, 2.0f, 2.0f),        // eye
            glm::vec3(0.0f, 0.0f, 0.0f),        // center
            glm::vec3(0.0f, 1.0f, 0.0f));       // up
        ubo.m_proj = glm::perspective(
            glm::radians(45.0f),                // vertical FOV
            static_cast<float>(m_swap_chain_extent.width) / m_swap_chain_extent.height,
            0.1f,                               // Z-near
            10.0f);                             // Z-far

        // Account for GLM being initially designed for OpenGL.
        ubo.m_proj[1][1] *= -1.0f;

        // TODO: split into separate function.
        void* data;
        if (vkMapMemory(m_device, m_uniform_buffers_memory[current_image], 0, sizeof(ubo), 0, &data) != VK_SUCCESS)
            throw std::runtime_error("Failed to map Vulkan buffer memory to host address space");
        std::memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(m_device, m_uniform_buffers_memory[current_image]);
    }

    void recreate_vk_swap_chain()
    {
        std::cout << "Recreating Vulkan swap chain..." << std::endl;

        int window_width, window_height;
        glfwGetFramebufferSize(m_window, &window_width, &window_height);

        if (window_width == 0 && window_height == 0)
        {
            std::cout << "Window is minimized, waiting until it is brought back to the foreground..." << std::endl;

            while (window_width == 0 || window_height == 0)
            {
                glfwWaitEvents();
                glfwGetFramebufferSize(m_window, &window_width, &window_height);
            }
        }

        vkDeviceWaitIdle(m_device);

        cleanup_vk_swap_chain();

        create_vk_swap_chain();
        create_vk_swap_chain_image_views();
        create_vk_render_pass();
        create_vk_graphics_pipeline();
        create_vk_framebuffers();
        create_vk_command_buffers();
    }

    void cleanup_vk_swap_chain()
    {
        for (const VkFramebuffer framebuffer : m_swap_chain_framebuffers)
            vkDestroyFramebuffer(m_device, framebuffer, nullptr);

        vkFreeCommandBuffers(
            m_device,
            m_command_pool,
            static_cast<std::uint32_t>(m_command_buffers.size()),
            m_command_buffers.data());

        vkDestroyPipeline(m_device, m_graphics_pipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
        vkDestroyRenderPass(m_device, m_render_pass, nullptr);

        for (const VkImageView image_view : m_swap_chain_image_views)
            vkDestroyImageView(m_device, image_view, nullptr);

        vkDestroySwapchainKHR(m_device, m_swap_chain, nullptr);
    }

    void cleanup_vulkan()
    {
        std::cout << "Cleaning up..." << std::endl;

        vkDeviceWaitIdle(m_device);

        cleanup_vk_swap_chain();

        vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);

        for (std::size_t i = 0; i < m_swap_chain_images.size(); ++i)
        {
            vkDestroyBuffer(m_device, m_uniform_buffers[i], nullptr);
            vkFreeMemory(m_device, m_uniform_buffers_memory[i], nullptr);
        }

        vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);

        vkDestroyBuffer(m_device, m_index_buffer, nullptr);
        vkFreeMemory(m_device, m_index_buffer_memory, nullptr);

        vkDestroyBuffer(m_device, m_vertex_buffer, nullptr);
        vkFreeMemory(m_device, m_vertex_buffer_memory, nullptr);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            vkDestroyFence(m_device, m_in_flight_fences[i], nullptr);
            vkDestroySemaphore(m_device, m_render_finished_semaphores[i], nullptr);
            vkDestroySemaphore(m_device, m_image_available_semaphores[i], nullptr);
        }

        vkDestroyCommandPool(m_device, m_transient_command_pool, nullptr);
        vkDestroyCommandPool(m_device, m_command_pool, nullptr);

        vkDestroyDevice(m_device, nullptr);

        if (EnableVkValidationLayers)
        {
            auto vkDestroyDebugUtilsMessengerEXTFn =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (vkDestroyDebugUtilsMessengerEXTFn == nullptr)
                throw std::runtime_error("Failed to load vkDestroyDebugUtilsMessengerEXT() function");

            vkDestroyDebugUtilsMessengerEXTFn(m_instance, m_debug_messenger, nullptr);
        }

        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkDestroyInstance(m_instance, nullptr);
    }

    void destroy_window()
    {
        std::cout << "Destroying window..." << std::endl;

        glfwDestroyWindow(m_window);
    }
};

int main()
{
    int exit_code = 1;

    glfwInit();

    try
    {

        HelloTriangleApplication app;
        app.run();

        exit_code = 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "." << std::endl;
    }

    glfwTerminate();

    return exit_code;
}
