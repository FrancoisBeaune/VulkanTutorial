
#include <algorithm>
#include <array>
#include <cassert>
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

const int WindowWidth = 800;
const int WindowHeight = 600;

#ifdef NDEBUG
const bool EnableVkValidationLayers = false;
#else
const bool EnableVkValidationLayers = true;
#endif

const std::vector<const char*> ValidationLayers =
{
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> DeviceExtensions =
{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
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
        cleanup();
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
    VkRenderPass                    m_render_pass;
    VkPipelineLayout                m_pipeline_layout;
    VkPipeline                      m_graphics_pipeline;
    std::vector<VkFramebuffer>      m_swap_chain_framebuffers;
    VkCommandPool                   m_command_pool;
    std::vector<VkCommandBuffer>    m_command_buffers;

    void init_vulkan()
    {
        query_vk_instance_extensions();
        create_vk_instance();
        setup_vk_debug_messenger();
        create_vk_surface();
        pick_vk_physical_device();
        create_vk_logical_device();
        create_vk_swap_chain();
        retrieve_vk_swap_chain_images();
        create_vk_swap_chain_image_views();
        create_vk_render_pass();
        create_vk_graphics_pipeline();
        create_vk_framebuffers();
        create_vk_command_pool();
        allocate_vk_command_buffers();
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

    static VkExtent2D choose_vk_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
            return capabilities.currentExtent;

        VkExtent2D actual_extent =
        {
            static_cast<std::uint32_t>(WindowWidth),
            static_cast<std::uint32_t>(WindowHeight)
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
    }

    void retrieve_vk_swap_chain_images()
    {
        std::uint32_t image_count;
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

        VkRenderPassCreateInfo render_pass_create_info = {};
        render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_create_info.attachmentCount = 1;
        render_pass_create_info.pAttachments = &color_attachment;
        render_pass_create_info.subpassCount = 1;
        render_pass_create_info.pSubpasses = &subpass;

        if (vkCreateRenderPass(m_device, &render_pass_create_info, nullptr, &m_render_pass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan render pass");
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

        VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
        vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 0;
        vertex_input_info.pVertexBindingDescriptions = nullptr;
        vertex_input_info.vertexAttributeDescriptionCount = 0;
        vertex_input_info.pVertexAttributeDescriptions = nullptr;

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
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
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
        pipeline_layout_create_info.setLayoutCount = 0;
        pipeline_layout_create_info.pSetLayouts = nullptr;
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

    void create_vk_command_pool()
    {
        std::cout << "Creating Vulkan command pool..." << std::endl;

        const QueueFamilyIndices queue_family_indices = find_vk_queue_families(m_physical_device);

        VkCommandPoolCreateInfo pool_create_info = {};
        pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_create_info.queueFamilyIndex = queue_family_indices.m_graphics_family.value();
        pool_create_info.flags = 0;

        if (vkCreateCommandPool(m_device, &pool_create_info, nullptr, &m_command_pool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan command pool");
    }

    void allocate_vk_command_buffers()
    {
        std::cout << "Allocating Vulkan command buffers..." << std::endl;

        m_command_buffers.resize(m_swap_chain_framebuffers.size());

        VkCommandBufferAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = m_command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<std::uint32_t>(m_command_buffers.size());

        if (vkAllocateCommandBuffers(m_device, &alloc_info, m_command_buffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate Vulkan command buffer");

        for (std::size_t i = 0; i < m_command_buffers.size(); ++i)
        {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            begin_info.pInheritanceInfo = nullptr;

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
            vkCmdDraw(m_command_buffers[i], 3, 1, 0, 0);
            vkCmdEndRenderPass(m_command_buffers[i]);

            if (vkEndCommandBuffer(m_command_buffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to end recording Vulkan command buffer");
        }
    }

    void create_window()
    {
        std::cout << "Creating window..." << std::endl;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        m_window = glfwCreateWindow(WindowWidth, WindowHeight, "Vulkan Tutorial", nullptr, nullptr);
    }

    void main_loop()
    {
        std::cout << "Entering main loop..." << std::endl;

        while (!glfwWindowShouldClose(m_window))
        {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        std::cout << "Cleaning up..." << std::endl;

        vkDestroyCommandPool(m_device, m_command_pool, nullptr);

        for (const VkFramebuffer framebuffer : m_swap_chain_framebuffers)
            vkDestroyFramebuffer(m_device, framebuffer, nullptr);

        vkDestroyPipeline(m_device, m_graphics_pipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
        vkDestroyRenderPass(m_device, m_render_pass, nullptr);

        for (const VkImageView image_view : m_swap_chain_image_views)
            vkDestroyImageView(m_device, image_view, nullptr);

        vkDestroySwapchainKHR(m_device, m_swap_chain, nullptr);
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
