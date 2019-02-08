
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <exception>
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
    GLFWwindow*                 m_window;
    VkInstance                  m_instance;
    VkDebugUtilsMessengerEXT    m_debug_messenger;
    VkSurfaceKHR                m_surface;
    VkPhysicalDevice            m_physical_device;
    VkDevice                    m_device;
    VkQueue                     m_graphics_queue;
    VkQueue                     m_present_queue;
    VkSwapchainKHR              m_swap_chain;
    VkFormat                    m_swap_chain_surface_format;
    VkExtent2D                  m_swap_chain_extent;
    std::vector<VkImage>        m_swap_chain_images;
    std::vector<VkImageView>    m_swap_chain_image_views;

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

    QueueFamilyIndices find_vk_queue_families(VkPhysicalDevice physical_device)
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

    bool check_vk_device_extension_support(VkPhysicalDevice physical_device)
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

    SwapChainSupportDetails query_vk_swap_chain_support(VkPhysicalDevice physical_device)
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

    VkSurfaceFormatKHR choose_vk_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats)
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

    VkPresentModeKHR choose_vk_swap_present_mode(const std::vector<VkPresentModeKHR>& available_modes)
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
