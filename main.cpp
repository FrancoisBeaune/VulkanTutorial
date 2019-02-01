
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
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

const std::vector<const char*> VkValidationLayers =
{
    "VK_LAYER_LUNARG_standard_validation"
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
        init_vulkan();
        create_window();
        main_loop();
        cleanup();
    }

  private:
    GLFWwindow*                 m_window;
    VkInstance                  m_instance;
    VkDebugUtilsMessengerEXT    m_debug_messenger;
    VkPhysicalDevice            m_physical_device;

    void init_vulkan()
    {
        query_vk_instance_extensions();
        create_vk_instance();
        setup_vk_debug_messenger();
        pick_vk_physical_device();
    }

    void query_vk_instance_extensions() const
    {
        std::cout << "Querying Vulkan instance extensions..." << std::endl;

        std::uint32_t extension_count;
        if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Failed to enumerate Vulkan instance extensions");

        std::vector<VkExtensionProperties> extension_props(extension_count);
        EXPECT_VK_SUCCESS(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extension_props.data()));

        if (extension_count > 0)
        {
            std::cout << extension_count << " extension(s) found:" << std::endl;
            for (const VkExtensionProperties& ext : extension_props)
                std::cout << "    " << ext.extensionName << " (version " << ext.specVersion << ", or " << make_version_string(ext.specVersion) << ")" << std::endl;
        }
        else std::cout << "No extension found." << std::endl;
    }

    bool check_vk_validation_layer_support() const
    {
        std::uint32_t layer_count;
        if (vkEnumerateInstanceLayerProperties(&layer_count, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Failed to enumerate Vulkan instance validation layers");

        std::vector<VkLayerProperties> layer_props(layer_count);
        EXPECT_VK_SUCCESS(vkEnumerateInstanceLayerProperties(&layer_count, layer_props.data()));

        for (const char* layer : VkValidationLayers)
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

    std::vector<const char*> get_required_extensions() const
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

        const auto extensions = get_required_extensions();

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
            create_info.enabledLayerCount = static_cast<std::uint32_t>(VkValidationLayers.size());
            create_info.ppEnabledLayerNames = VkValidationLayers.data();
        }
        else
        {
            create_info.enabledLayerCount = 0;
            create_info.ppEnabledLayerNames = nullptr;
        }
        create_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        if (vkCreateInstance(&create_info, nullptr, &m_instance) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan instance");
    }

    void setup_vk_debug_messenger()
    {
        if (EnableVkValidationLayers)
        {
            std::cout << "Setting up Vulkan debug messenger..." << std::endl;

            auto vkCreateDebugUtilsMessengerEXTFn =
                reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
            if (vkCreateDebugUtilsMessengerEXTFn == nullptr)
                throw std::runtime_error("Failed to load vkCreateDebugUtilsMessengerEXT() function");

            VkDebugUtilsMessengerCreateInfoEXT create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            create_info.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
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

    struct QueueFamilyIndices
    {
        std::optional<std::uint32_t> m_graphics_family;

        bool is_complete() const
        {
            return m_graphics_family.has_value();
        }
    };

    QueueFamilyIndices find_vk_queue_families(const VkPhysicalDevice& device) const
    {
        QueueFamilyIndices indices;

        std::uint32_t queue_family_count;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_family_props(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_family_props.data());

        for (std::size_t i = 0, e = queue_family_props.size(); i < e; ++i)
        {
            const VkQueueFamilyProperties& props = queue_family_props[i];

            if (props.queueCount > 0 && props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.m_graphics_family = static_cast<std::uint32_t>(i);

            if (indices.is_complete())
                break;
        }

        return indices;
    }

    bool is_vk_device_suitable(const VkPhysicalDevice& device) const
    {
        const QueueFamilyIndices indices = find_vk_queue_families(device);
        return indices.is_complete();
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

            std::cout << "    " << device_props.deviceName << " (driver version: " << make_version_string(device_props.driverVersion) << ")" << std::endl;
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

        if (EnableVkValidationLayers)
        {
            auto vkDestroyDebugUtilsMessengerEXTFn =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (vkDestroyDebugUtilsMessengerEXTFn == nullptr)
                throw std::runtime_error("Failed to load vkDestroyDebugUtilsMessengerEXT() function");

            vkDestroyDebugUtilsMessengerEXTFn(m_instance, m_debug_messenger, nullptr);
        }

        vkDestroyInstance(m_instance, nullptr);

        glfwDestroyWindow(m_window);
    }
};

int main()
{
    int exit_code = 1;

    try
    {
        glfwInit();

        HelloTriangleApplication app;
        app.run();

        glfwTerminate();

        exit_code = 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
    }

    return exit_code;
}
