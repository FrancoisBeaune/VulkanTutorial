#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// GLM preprocessor definitions set in project.
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// NVIDIA Vulkan Ray Tracing helpers.
#include "nv_helpers_vk/BottomLevelASGenerator.h"
#include "nv_helpers_vk/DescriptorSetGenerator.h"
#include "nv_helpers_vk/RaytracingPipelineGenerator.h"
#include "nv_helpers_vk/ShaderBindingTableGenerator.h"
#include "nv_helpers_vk/TopLevelASGenerator.h"
#include "nv_helpers_vk/VKHelpers.h"

#include "vku.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
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
#include <thread>
#include <unordered_map>
#include <vector>

#define FORCE_VSYNC

const int WindowWidth = 800;
const int WindowHeight = 600;
const std::size_t MaxFramesInFlight = 2;

enum class RenderingMode
{
    Rasterization,
    RayTracing
};

//const RenderingMode CurrentRenderingMode = RenderingMode::Rasterization;
const RenderingMode CurrentRenderingMode = RenderingMode::RayTracing;

#ifdef NDEBUG
const std::string ModelPath = "models/chalet.obj";
const std::string TexturePath = "textures/chalet.jpg";
#else
const std::string ModelPath = "models/teapot.obj";
const std::string TexturePath = "textures/wood.png";
#endif

#ifndef FORCE_VSYNC
const std::size_t RenderFrameRate = 120; // Hz
#endif

#ifdef NDEBUG
const bool EnableValidationLayers = false;
#else
const bool EnableValidationLayers = true;
#endif

const std::vector<const char*> ValidationLayers =
{
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> DeviceExtensions =
{
    VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_NV_RAY_TRACING_EXTENSION_NAME
};

const std::vector<const char*> InstanceExtensions =
{
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
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

struct UniformBufferObject
{
    alignas(16) glm::mat4 m_model;
    alignas(16) glm::mat4 m_view;
    alignas(16) glm::mat4 m_proj;
};

struct Vertex
{
    glm::vec3 m_position;
    glm::vec2 m_tex_coords;
    glm::vec3 m_color;

    static VkVertexInputBindingDescription get_binding_description()
    {
        VkVertexInputBindingDescription binding_description = {};
        binding_description.binding = 0;
        binding_description.stride = sizeof(Vertex);
        binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding_description;
    }

    static std::array<VkVertexInputAttributeDescription, 3> get_attribute_descriptions()
    {
        std::array<VkVertexInputAttributeDescription, 3> attribute_descriptions = {};

        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(Vertex, m_position);

        attribute_descriptions[2].binding = 0;
        attribute_descriptions[2].location = 1;
        attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attribute_descriptions[2].offset = offsetof(Vertex, m_tex_coords);

        attribute_descriptions[1].binding = 0;
        attribute_descriptions[1].location = 2;
        attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descriptions[1].offset = offsetof(Vertex, m_color);

        return attribute_descriptions;
    }

    bool operator==(const Vertex& rhs) const
    {
        return
            m_position == rhs.m_position &&
            m_tex_coords == rhs.m_tex_coords &&
            m_color == rhs.m_color;
    }
};

struct Material
{
    glm::vec3 m_diffuse;
};

struct GeometryInstance
{
    VkBuffer        m_vertex_buffer;
    std::uint32_t   m_vertex_count;
    VkDeviceSize    m_vertex_offset;
    VkBuffer        m_index_buffer;
    std::uint32_t   m_index_count;
    VkDeviceSize    m_index_offset;
    glm::mat4x4     m_transform;
};

struct AccelerationStructure
{
    VkBuffer                    m_scratch_buffer = VK_NULL_HANDLE;
    VkDeviceMemory              m_scratch_mem = VK_NULL_HANDLE;
    VkBuffer                    m_result_buffer = VK_NULL_HANDLE;
    VkDeviceMemory              m_result_mem = VK_NULL_HANDLE;
    VkBuffer                    m_instances_buffer = VK_NULL_HANDLE;
    VkDeviceMemory              m_instances_mem = VK_NULL_HANDLE;
    VkAccelerationStructureNV   m_structure = VK_NULL_HANDLE;
};

//
// Derivation of the 0x9E3779B97F4A7C17u constant:
//
//   #include <cmath>
//   #include <cstdint>
//   #include <ios>
//   #include <iostream>
//   
//   int main()
//   {
//       const long double num = std::pow(2.0l, 64.0l);
//       const long double phi = 1.61803398874989484820459l;   // (1 + sqrt(5)) / 2
//       const long double x = num / phi;
//       const std::uint64_t y = static_cast<std::uint64_t>(x);
//       const std::uint64_t z = (y % 2 == 0) ? y + 1 : y;
//       std::cout << "0x" << std::hex << std::uppercase << z << "u" << std::endl;
//   }
//
// Run with a compiler that implements 'long double' with at least 80 bits (i.e. not Visual C++).
//

std::size_t combine_hashes(
    const std::size_t h1,
    const std::size_t h2)
{
    // Inspired by boost::hash_combine(): https://www.boost.org/doc/libs/1_69_0/doc/html/hash/combine.html
    return h1 ^ (h2 + 0x9E3779B97F4A7C17u + (h1 << 6) + (h1 >> 2));
}

std::size_t combine_hashes(
    const std::size_t h1,
    const std::size_t h2,
    const std::size_t h3)
{
    return combine_hashes(combine_hashes(h1, h2), h3);
}

namespace std
{
    template <>
    struct hash<Vertex>
    {
        size_t operator()(const Vertex& vertex) const
        {
            return
                combine_hashes(
                    hash<glm::vec3>()(vertex.m_position),
                    hash<glm::vec2>()(vertex.m_tex_coords),
                    hash<glm::vec3>()(vertex.m_color));
        }
    };
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
    GLFWwindow*                                 m_window;
    int                                         m_window_width;
    int                                         m_window_height;

    VkInstance                                  m_instance;
    VkDebugUtilsMessengerEXT                    m_debug_messenger;
    VkSurfaceKHR                                m_surface;

    VkPhysicalDevice                            m_physical_device;
    VkSampleCountFlagBits                       m_msaa_samples = VK_SAMPLE_COUNT_1_BIT;
    VkPhysicalDeviceRayTracingPropertiesNV      m_phsical_device_rt_props;

    VkDevice                                    m_device;

    VkQueue                                     m_graphics_queue;
    VkQueue                                     m_present_queue;

    VkSwapchainKHR                              m_swap_chain;
    VkFormat                                    m_swap_chain_surface_format;
    VkExtent2D                                  m_swap_chain_extent;
    std::vector<VkImage>                        m_swap_chain_images;
    std::vector<VkImageView>                    m_swap_chain_image_views;
    std::vector<VkFramebuffer>                  m_swap_chain_framebuffers;

    VkRenderPass                                m_render_pass;
    VkDescriptorSetLayout                       m_descriptor_set_layout;
    VkPipelineLayout                            m_graphics_pipeline_layout;
    VkPipeline                                  m_graphics_pipeline;

    VkCommandPool                               m_command_pool;
    VkCommandPool                               m_transient_command_pool;

    VkImage                                     m_color_image;
    VkDeviceMemory                              m_color_image_memory;
    VkImageView                                 m_color_image_view;

    VkImage                                     m_depth_image;
    VkDeviceMemory                              m_depth_image_memory;
    VkImageView                                 m_depth_image_view;

    std::vector<Vertex>                         m_vertices;
    std::vector<std::uint32_t>                  m_indices;
    std::vector<Material>                       m_materials;

    VkBuffer                                    m_vertex_buffer;
    VkDeviceMemory                              m_vertex_buffer_memory;
    VkBuffer                                    m_index_buffer;
    VkDeviceMemory                              m_index_buffer_memory;
    VkBuffer                                    m_material_buffer;
    VkDeviceMemory                              m_material_buffer_memory;
    std::vector<VkBuffer>                       m_uniform_buffers;
    std::vector<VkDeviceMemory>                 m_uniform_buffers_memory;

    // Vulkan Ray Tracing.
    AccelerationStructure                       m_rt_bottom_level_as;
    nv_helpers_vk::TopLevelASGenerator          m_rt_top_level_as_gen;
    AccelerationStructure                       m_rt_top_level_as;
    nv_helpers_vk::DescriptorSetGenerator       m_rt_descriptor_set_generator;
    VkDescriptorPool                            m_rt_descriptor_pool;
    VkDescriptorSetLayout                       m_rt_descriptor_set_layout;
    VkDescriptorSet                             m_rt_descriptor_set;
    VkPipelineLayout                            m_rt_pipeline_layout;
    VkPipeline                                  m_rt_pipeline;
    std::uint32_t                               m_rt_ray_gen_index;
    std::uint32_t                               m_rt_ray_miss_index;
    std::uint32_t                               m_rt_hit_group_index;
    nv_helpers_vk::ShaderBindingTableGenerator  m_rt_sbt_generator;
    VkBuffer                                    m_rt_shader_binding_table_buffer;
    VkDeviceMemory                              m_rt_shader_binding_table_buffer_memory;

    std::uint32_t                               m_texture_mip_levels;
    VkImage                                     m_texture_image;
    VkDeviceMemory                              m_texture_image_memory;
    VkImageView                                 m_texture_image_view;
    VkSampler                                   m_texture_sampler;

    VkDescriptorPool                            m_descriptor_pool;
    std::vector<VkDescriptorSet>                m_descriptor_sets;

    std::vector<VkCommandBuffer>                m_command_buffers;

    std::vector<VkSemaphore>                    m_image_available_semaphores;
    std::vector<VkSemaphore>                    m_render_finished_semaphores;
    std::vector<VkFence>                        m_in_flight_fences;

    std::atomic<bool>                           m_framebuffer_resized = false;

    // "Game state".
    float                                       m_rotation_angle = 0.0f;

    void create_window()
    {
        std::cout << "Creating window..." << std::endl;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        m_window = glfwCreateWindow(WindowWidth, WindowHeight, "Vulkan Tutorial", nullptr, nullptr);
        glfwGetFramebufferSize(m_window, &m_window_width, &m_window_height);

        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, framebuffer_resize_callback);
        glfwSetKeyCallback(m_window, key_callback);
    }

    static void framebuffer_resize_callback(GLFWwindow* window, const int width, const int height)
    {
        HelloTriangleApplication* app =
            reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));

        app->m_framebuffer_resized = true;
    }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    void init_vulkan()
    {
        std::cout << "Initializing Vulkan..." << std::endl;

        print_vk_instance_extensions();
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

        create_vk_command_pools();

        create_vk_color_resources();
        create_vk_depth_resources();
        create_vk_framebuffers();

        load_model();

        create_vk_vertex_buffer();
        create_vk_index_buffer();
        create_vk_material_buffer();
        create_vk_uniform_buffers();

        create_vk_texture_image();
        create_vk_texture_image_view();
        create_vk_texture_sampler();

        // Vulkan Ray Tracing.
        create_vk_rt_acceleration_structures();
        create_vk_rt_descriptor_set();
        create_vk_rt_pipeline();
        create_vk_rt_shader_binding_table();

        create_vk_descriptor_pool();
        create_vk_descriptor_sets();

        create_vk_command_buffers();

        create_vk_sync_objects();
    }

    static void print_vk_instance_extensions()
    {
        std::uint32_t extension_count;
        if (vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Failed to enumerate Vulkan instance extensions");

        std::vector<VkExtensionProperties> extension_props(extension_count);
        EXPECT_VK_SUCCESS(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extension_props.data()));

        if (extension_count > 0)
        {
            std::cout << extension_count << " Vulkan instance extension(s) found:" << std::endl;
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

        if (EnableValidationLayers)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        extensions.insert(std::end(extensions), std::cbegin(InstanceExtensions), std::cend(InstanceExtensions));

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

        if (EnableValidationLayers && !check_vk_validation_layer_support())
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
        if (EnableValidationLayers)
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
        if (EnableValidationLayers)
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

    static QueueFamilyIndices find_vk_queue_families(
        const VkPhysicalDevice  physical_device,
        const VkSurfaceKHR      surface)
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
            if (vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &is_present_supported) != VK_SUCCESS)
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

    static SwapChainSupportDetails query_vk_swap_chain_support(
        const VkPhysicalDevice  physical_device,
        const VkSurfaceKHR      surface)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &details.m_capabilities);

        std::uint32_t format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);

        if (format_count > 0)
        {
            details.m_formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, details.m_formats.data());
        }

        std::uint32_t present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            physical_device,
            surface,
            &present_mode_count,
            nullptr);

        if (present_mode_count > 0)
        {
            details.m_present_modes.resize(present_mode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                physical_device,
                surface,
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
#ifdef FORCE_VSYNC
        return VK_PRESENT_MODE_FIFO_KHR;
#else
        VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;

        for (const VkPresentModeKHR& mode : available_modes)
        {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                return mode;
            else if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
                best_mode = mode;
        }

        return best_mode;
#endif
    }

    static VkExtent2D choose_vk_swap_extent(
        const VkSurfaceCapabilitiesKHR& capabilities,
        const int                       window_width,
        const int                       window_height)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
            return capabilities.currentExtent;

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

    static bool is_vk_device_suitable(
        const VkPhysicalDevice  physical_device,
        const VkSurfaceKHR      surface)
    {
        const QueueFamilyIndices indices = find_vk_queue_families(physical_device, surface);

        const bool extensions_supported = check_vk_device_extension_support(physical_device);

        bool swap_chain_adequate = false;
        if (extensions_supported)
        {
            const SwapChainSupportDetails swap_chain_support = query_vk_swap_chain_support(physical_device, surface);
            swap_chain_adequate = !swap_chain_support.m_formats.empty() && !swap_chain_support.m_present_modes.empty();
        }

        VkPhysicalDeviceFeatures supported_features;
        vkGetPhysicalDeviceFeatures(physical_device, &supported_features);

        return
            indices.is_complete() &&
            extensions_supported &&
            swap_chain_adequate &&
            supported_features.samplerAnisotropy;
    }

    static VkSampleCountFlagBits get_vk_max_usable_sample_count(
        const VkPhysicalDevice  physical_device)
    {
        VkPhysicalDeviceProperties device_props;
        vkGetPhysicalDeviceProperties(physical_device, &device_props);

        const VkSampleCountFlags counts =
            std::min(
                device_props.limits.framebufferColorSampleCounts,
                device_props.limits.framebufferDepthSampleCounts);

        if (counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
        else if (counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
        else if (counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
        else if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
        else if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
        else if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;
        else return VK_SAMPLE_COUNT_1_BIT;
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

        std::cout << device_count << " Vulkan physical device(s) found:" << std::endl;

        for (const VkPhysicalDevice& device : devices)
        {
            VkPhysicalDeviceProperties device_props;
            vkGetPhysicalDeviceProperties(device, &device_props);

            std::cout << "    " << device_props.deviceName << " (driver version: "
                << make_version_string(device_props.driverVersion) << ")" << std::endl;

            std::uint32_t extension_count;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

            std::vector<VkExtensionProperties> available_extensions(extension_count);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

            if (extension_count > 0)
            {
                std::cout << "    " << extension_count << " device extension(s) found:" << std::endl;
                for (const VkExtensionProperties& ext : available_extensions)
                {
                    std::cout << "        " << ext.extensionName << " (version " << ext.specVersion
                        << ", or " << make_version_string(ext.specVersion) << ")" << std::endl;
                }
            }
            else std::cout << "    No device extension found." << std::endl;
        }

        m_physical_device = VK_NULL_HANDLE;
        for (const VkPhysicalDevice& device : devices)
        {
            if (is_vk_device_suitable(device, m_surface))
            {
                m_physical_device = device;
                m_msaa_samples = get_vk_max_usable_sample_count(device);
                break;
            }
        }

        if (m_physical_device == VK_NULL_HANDLE)
            throw std::runtime_error("Failed to find a suitable Vulkan device");

        m_phsical_device_rt_props = {};
        m_phsical_device_rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;

        VkPhysicalDeviceProperties2 device_props = {};
        device_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        device_props.pNext = &m_phsical_device_rt_props;

        vkGetPhysicalDeviceProperties2(m_physical_device, &device_props);
    }

    void create_vk_logical_device()
    {
        std::cout << "Creating Vulkan logical device..." << std::endl;

        const QueueFamilyIndices indices = find_vk_queue_families(m_physical_device, m_surface);
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

        VkPhysicalDeviceFeatures physical_device_features = {};
        physical_device_features.sampleRateShading = VK_TRUE;
        physical_device_features.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo device_create_info = {};
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.queueCreateInfoCount = static_cast<std::uint32_t>(queue_create_infos.size());
        device_create_info.pQueueCreateInfos = queue_create_infos.data();
        if (EnableValidationLayers)
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
        device_create_info.pEnabledFeatures = &physical_device_features;

        if (vkCreateDevice(m_physical_device, &device_create_info, nullptr, &m_device) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan logical device");

        vkGetDeviceQueue(m_device, indices.m_graphics_family.value(), 0, &m_graphics_queue);
        vkGetDeviceQueue(m_device, indices.m_present_family.value(), 0, &m_present_queue);
    }

    void create_vk_swap_chain()
    {
        std::cout << "Creating Vulkan swap chain..." << std::endl;

        const SwapChainSupportDetails swap_chain_support =
            query_vk_swap_chain_support(m_physical_device, m_surface);

        const VkSurfaceFormatKHR surface_format = choose_vk_swap_surface_format(swap_chain_support.m_formats);
        const VkPresentModeKHR present_mode = choose_vk_swap_present_mode(swap_chain_support.m_present_modes);
        const VkExtent2D extent = choose_vk_swap_extent(swap_chain_support.m_capabilities, m_window_width, m_window_height);

        std::uint32_t image_count = swap_chain_support.m_capabilities.minImageCount + 1;
        if (swap_chain_support.m_capabilities.maxImageCount > 0)
            image_count = std::min(image_count, swap_chain_support.m_capabilities.maxImageCount);

        const QueueFamilyIndices indices = find_vk_queue_families(m_physical_device, m_surface);
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
        // VK_IMAGE_USAGE_STORAGE_BIT required by Vulkan ray tracing.
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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
            m_swap_chain_image_views[i] =
                vku_create_image_view(
                    m_device,
                    m_swap_chain_images[i],
                    1,
                    m_swap_chain_surface_format,
                    VK_IMAGE_ASPECT_COLOR_BIT);
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
        color_attachment.samples = m_msaa_samples;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_attachment_ref = {};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depth_attachment = {};
        depth_attachment.format = vku_find_depth_format(m_physical_device);
        depth_attachment.samples = m_msaa_samples;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_attachment_ref = {};
        depth_attachment_ref.attachment = 1;
        depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription color_resolve_attachment = {};
        color_resolve_attachment.format = m_swap_chain_surface_format;
        color_resolve_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_resolve_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_resolve_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_resolve_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_resolve_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_resolve_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_resolve_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_resolve_attachment_ref = {};
        color_resolve_attachment_ref.attachment = 2;
        color_resolve_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // TODO: why not VK_IMAGE_LAYOUT_PRESENT_SRC_KHR?

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;
        subpass.pResolveAttachments = &color_resolve_attachment_ref;
        subpass.pDepthStencilAttachment = &depth_attachment_ref;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        const std::array<VkAttachmentDescription, 3> attachments =
        {
            color_attachment,
            depth_attachment,
            color_resolve_attachment
        };

        VkRenderPassCreateInfo render_pass_create_info = {};
        render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_create_info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        render_pass_create_info.pAttachments = attachments.data();
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

        VkDescriptorSetLayoutBinding sampler_layout_binding = {};
        sampler_layout_binding.binding = 1;
        sampler_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sampler_layout_binding.descriptorCount = 1;
        sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        sampler_layout_binding.pImmutableSamplers = nullptr;

        const std::array<VkDescriptorSetLayoutBinding, 2> bindings =
        {
            ubo_layout_binding,
            sampler_layout_binding
        };

        VkDescriptorSetLayoutCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        create_info.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(m_device, &create_info, nullptr, &m_descriptor_set_layout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan descriptor set layout");
    }

    void create_vk_graphics_pipeline()
    {
        std::cout << "Creating Vulkan graphics pipeline..." << std::endl;

        const std::vector<char> vert_shader_code = read_file("shaders/basic_vertex_shader.spv");
        const std::vector<char> frag_shader_code = read_file("shaders/basic_fragment_shader.spv");

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
        multisampling.sampleShadingEnable = VK_TRUE;
        multisampling.rasterizationSamples = m_msaa_samples;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
        depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable = VK_TRUE;
        depth_stencil.depthWriteEnable = VK_TRUE;
        depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depth_stencil.depthBoundsTestEnable = VK_FALSE;
        depth_stencil.minDepthBounds = 0.0f;
        depth_stencil.maxDepthBounds = 1.0f;
        depth_stencil.stencilTestEnable = VK_FALSE;
        depth_stencil.front = {};
        depth_stencil.back = {};

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

        if (vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &m_graphics_pipeline_layout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan graphics pipeline layout");

        VkGraphicsPipelineCreateInfo pipeline_create_info = {};
        pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_create_info.stageCount = 2;
        pipeline_create_info.pStages = shader_stages;
        pipeline_create_info.pVertexInputState = &vertex_input_info;
        pipeline_create_info.pInputAssemblyState = &input_assembly;
        pipeline_create_info.pViewportState = &viewport_state;
        pipeline_create_info.pRasterizationState = &rasterizer;
        pipeline_create_info.pMultisampleState = &multisampling;
        pipeline_create_info.pDepthStencilState = &depth_stencil;
        pipeline_create_info.pColorBlendState = &color_blending;
        pipeline_create_info.pDynamicState = nullptr;
        pipeline_create_info.layout = m_graphics_pipeline_layout;
        pipeline_create_info.renderPass = m_render_pass;
        pipeline_create_info.subpass = 0;
        pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_create_info.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &m_graphics_pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan graphics pipeline");

        vkDestroyShaderModule(m_device, frag_shader_module, nullptr);
        vkDestroyShaderModule(m_device, vert_shader_module, nullptr);
    }

    void create_vk_command_pool(const VkCommandPoolCreateFlags flags, VkCommandPool& command_pool)
    {
        const QueueFamilyIndices queue_family_indices = find_vk_queue_families(m_physical_device, m_surface);

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

        // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT required because we rewrite command buffers
        // for each frame during ray tracing.
        create_vk_command_pool(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, m_command_pool);
        create_vk_command_pool(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_transient_command_pool);
    }

    void create_vk_color_resources()
    {
        std::cout << "Creating Vulkan color resources..." << std::endl;

        vku_create_image(
            m_physical_device,
            m_device,
            m_swap_chain_extent.width,
            m_swap_chain_extent.height,
            1,
            m_msaa_samples,
            m_swap_chain_surface_format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_color_image,
            m_color_image_memory);

        m_color_image_view =
            vku_create_image_view(
                m_device,
                m_color_image,
                1,
                m_swap_chain_surface_format,
                VK_IMAGE_ASPECT_COLOR_BIT);

        vku_transition_image_layout(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            m_color_image,
            1,
            m_swap_chain_surface_format,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    void create_vk_depth_resources()
    {
        std::cout << "Creating Vulkan depth resources..." << std::endl;

        const VkFormat depth_format = vku_find_depth_format(m_physical_device);

        vku_create_image(
            m_physical_device,
            m_device,
            m_swap_chain_extent.width,
            m_swap_chain_extent.height,
            1,
            m_msaa_samples,
            depth_format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_depth_image,
            m_depth_image_memory);

        m_depth_image_view =
            vku_create_image_view(
                m_device,
                m_depth_image,
                1,
                depth_format,
                VK_IMAGE_ASPECT_DEPTH_BIT);

        vku_transition_image_layout(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            m_depth_image,
            1,
            depth_format,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    void create_vk_framebuffers()
    {
        std::cout << "Creating Vulkan framebuffers..." << std::endl;

        m_swap_chain_framebuffers.resize(m_swap_chain_image_views.size());

        for (std::size_t i = 0; i < m_swap_chain_image_views.size(); ++i)
        {
            const std::array<VkImageView, 3> attachments =
            {
                m_color_image_view,
                m_depth_image_view,
                m_swap_chain_image_views[i]
            };

            VkFramebufferCreateInfo framebuffer_create_info = {};
            framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_create_info.renderPass = m_render_pass;
            framebuffer_create_info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
            framebuffer_create_info.pAttachments = attachments.data();
            framebuffer_create_info.width = m_swap_chain_extent.width;
            framebuffer_create_info.height = m_swap_chain_extent.height;
            framebuffer_create_info.layers = 1;

            if (vkCreateFramebuffer(m_device, &framebuffer_create_info, nullptr, &m_swap_chain_framebuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create Vulkan framebuffer");
        }
    }

    void load_model()
    {
        std::cout << "Loading " << ModelPath << "..." << std::endl;

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, ModelPath.c_str()))
            throw std::runtime_error(warn + err);

        std::cout << "    Initial vertex count: " << attrib.vertices.size() << std::endl;

        std::unordered_map<Vertex, std::uint32_t> unique_vertices;

        for (const auto& shape : shapes)
        {
            for (const auto& index : shape.mesh.indices)
            {
                Vertex vertex;
                vertex.m_position =
                {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };
                vertex.m_tex_coords =
                {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };
                vertex.m_color = { 1.0f, 1.0f, 1.0f };

                std::uint32_t vertex_index;
                const auto vertex_it = unique_vertices.find(vertex);
                if (vertex_it != std::cend(unique_vertices))
                    vertex_index = vertex_it->second;
                else
                {
                    vertex_index = static_cast<std::uint32_t>(m_vertices.size());
                    m_vertices.push_back(vertex);
                    unique_vertices.insert(std::make_pair(vertex, vertex_index));
                }

                m_indices.push_back(vertex_index);
            }
        }

        std::cout << "    Optimized vertex count: " << m_vertices.size() << std::endl;

        Material material;
        material.m_diffuse = glm::vec3(0.8f, 0.2f, 0.3f);
        m_materials.push_back(material);
    }

    void create_vk_vertex_buffer()
    {
        std::cout << "Creating Vulkan vertex buffer..." << std::endl;

        const VkDeviceSize buffer_size = sizeof(m_vertices[0]) * m_vertices.size();

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        vku_create_buffer(
            m_physical_device,
            m_device,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        vku_copy_host_to_device(
            m_device,
            staging_buffer_memory,
            m_vertices.data(),
            buffer_size);

        // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT allows ray tracing shaders to access the vertex buffer.
        vku_create_buffer(
            m_physical_device,
            m_device,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_vertex_buffer,
            m_vertex_buffer_memory);

        vku_copy_buffer_sync(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            m_vertex_buffer,
            staging_buffer,
            buffer_size);

        vkDestroyBuffer(m_device, staging_buffer, nullptr);
        vkFreeMemory(m_device, staging_buffer_memory, nullptr);
    }

    void create_vk_index_buffer()
    {
        std::cout << "Creating Vulkan index buffer..." << std::endl;

        const VkDeviceSize buffer_size = sizeof(m_indices[0]) * m_indices.size();

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        vku_create_buffer(
            m_physical_device,
            m_device,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        void* data;
        if (vkMapMemory(m_device, staging_buffer_memory, 0, buffer_size, 0, &data) != VK_SUCCESS)
            throw std::runtime_error("Failed to map Vulkan buffer memory to host address space");
        std::memcpy(data, m_indices.data(), static_cast<std::size_t>(buffer_size));
        vkUnmapMemory(m_device, staging_buffer_memory);

        // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT allows ray tracing shaders to access the index buffer.
        vku_create_buffer(
            m_physical_device,
            m_device,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_index_buffer,
            m_index_buffer_memory);

        vku_copy_buffer_sync(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            m_index_buffer,
            staging_buffer,
            buffer_size);

        vkDestroyBuffer(m_device, staging_buffer, nullptr);
        vkFreeMemory(m_device, staging_buffer_memory, nullptr);
    }

    void create_vk_material_buffer()
    {
        std::cout << "Creating Vulkan material buffer..." << std::endl;

        const VkDeviceSize buffer_size = sizeof(m_materials[0]) * m_materials.size();

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        vku_create_buffer(
            m_physical_device,
            m_device,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        vku_copy_host_to_device(
            m_device,
            staging_buffer_memory,
            m_materials.data(),
            buffer_size);

        vku_create_buffer(
            m_physical_device,
            m_device,
            buffer_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_material_buffer,
            m_material_buffer_memory);

        vku_copy_buffer_sync(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            m_material_buffer,
            staging_buffer,
            buffer_size);

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
            vku_create_buffer(
                m_physical_device,
                m_device,
                buffer_size,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_uniform_buffers[i],
                m_uniform_buffers_memory[i]);
        }
    }

    void create_vk_rt_acceleration_structures()
    {
        std::cout << "Creating Vulkan ray tracing acceleration structures..." << std::endl;

        VkCommandBuffer command_buffer =
            vku_begin_single_time_commands(m_device, m_transient_command_pool);

        m_rt_bottom_level_as = create_vk_rt_bottom_level_as(command_buffer);

        m_rt_top_level_as_gen.AddInstance(m_rt_bottom_level_as.m_structure, glm::mat4x4(1.0f), 0, 0);
        m_rt_top_level_as.m_structure = m_rt_top_level_as_gen.CreateAccelerationStructure(m_device, VK_TRUE);

        VkDeviceSize scratch_size = 0;              // bytes
        VkDeviceSize result_size = 0;               // bytes
        VkDeviceSize instance_descs_size = 0;       // bytes
        m_rt_top_level_as_gen.ComputeASBufferSizes(
            m_device,
            m_rt_top_level_as.m_structure,
            &scratch_size,
            &result_size,
            &instance_descs_size);

        // TODO: replace by vku function?
        nv_helpers_vk::createBuffer(
            m_physical_device,
            m_device,
            scratch_size,
            VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
            &m_rt_top_level_as.m_scratch_buffer,
            &m_rt_top_level_as.m_scratch_mem,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // TODO: replace by vku function?
        nv_helpers_vk::createBuffer(
            m_physical_device,
            m_device,
            result_size,
            VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
            &m_rt_top_level_as.m_result_buffer,
            &m_rt_top_level_as.m_result_mem,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // TODO: replace by vku function?
        nv_helpers_vk::createBuffer(
            m_physical_device,
            m_device,
            instance_descs_size,
            VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
            &m_rt_top_level_as.m_instances_buffer,
            &m_rt_top_level_as.m_instances_mem,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        m_rt_top_level_as_gen.Generate(
            m_device,
            command_buffer,
            m_rt_top_level_as.m_structure,
            m_rt_top_level_as.m_scratch_buffer,
            0,
            m_rt_top_level_as.m_result_buffer,
            m_rt_top_level_as.m_result_mem,
            m_rt_top_level_as.m_instances_buffer,
            m_rt_top_level_as.m_instances_mem,
            VK_NULL_HANDLE);    // TODO: pass m_rt_top_level_as.m_structure for update only

        vku_end_single_time_commands(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            command_buffer);
    }

    AccelerationStructure create_vk_rt_bottom_level_as(VkCommandBuffer command_buffer) const
    {
        GeometryInstance geometry_instance;
        geometry_instance.m_vertex_buffer = m_vertex_buffer;
        geometry_instance.m_vertex_count = static_cast<std::uint32_t>(m_vertices.size());
        geometry_instance.m_vertex_offset = 0;
        geometry_instance.m_index_buffer = m_index_buffer;
        geometry_instance.m_index_count = static_cast<std::uint32_t>(m_indices.size());
        geometry_instance.m_index_offset = 0;
        geometry_instance.m_transform = glm::mat4x4(1.0f);

        nv_helpers_vk::BottomLevelASGenerator bottom_level_as_gen;
        bottom_level_as_gen.AddVertexBuffer(
            geometry_instance.m_vertex_buffer,
            geometry_instance.m_vertex_offset,
            geometry_instance.m_vertex_count,
            sizeof(Vertex),
            geometry_instance.m_index_buffer,
            geometry_instance.m_index_offset,
            geometry_instance.m_index_count,
            VK_NULL_HANDLE,
            0);

        AccelerationStructure bottom_level_as;
        bottom_level_as.m_structure = bottom_level_as_gen.CreateAccelerationStructure(m_device, VK_FALSE);

        VkDeviceSize scratch_size = 0;  // bytes
        VkDeviceSize result_size = 0;   // bytes
        bottom_level_as_gen.ComputeASBufferSizes(m_device, bottom_level_as.m_structure, &scratch_size, &result_size);

        // TODO: replace by vku function?
        nv_helpers_vk::createBuffer(
            m_physical_device,
            m_device,
            scratch_size,
            VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
            &bottom_level_as.m_scratch_buffer,
            &bottom_level_as.m_scratch_mem,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // TODO: replace by vku function?
        nv_helpers_vk::createBuffer(
            m_physical_device,
            m_device,
            result_size,
            VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
            &bottom_level_as.m_result_buffer,
            &bottom_level_as.m_result_mem,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        bottom_level_as_gen.Generate(
            m_device,
            command_buffer,
            bottom_level_as.m_structure,
            bottom_level_as.m_scratch_buffer,
            0,
            bottom_level_as.m_result_buffer,
            bottom_level_as.m_result_mem,
            VK_FALSE,
            VK_NULL_HANDLE);

        return bottom_level_as;
    }

    void create_vk_rt_descriptor_set()
    {
        std::cout << "Creating Vulkan ray tracing descriptor set..." << std::endl;

        //
        // Make sure data from the vertex and index buffers are present on the GPU.
        //

        VkCommandBuffer command_buffer =
            vku_begin_single_time_commands(m_device, m_transient_command_pool);

        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        barrier.buffer = m_vertex_buffer;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr);

        barrier.buffer = m_index_buffer;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr);

        vku_end_single_time_commands(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            command_buffer);

        //
        // Create descriptor set.
        //

        // Location 0: top-level acceleration structure.
        m_rt_descriptor_set_generator.AddBinding(
            0,
            1,
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV,
            VK_SHADER_STAGE_RAYGEN_BIT_NV);

        // Location 1: ray tracing output image.
        m_rt_descriptor_set_generator.AddBinding(
            1,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_SHADER_STAGE_RAYGEN_BIT_NV);

        // Location 2: camera information.
        m_rt_descriptor_set_generator.AddBinding(
            2,
            1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_RAYGEN_BIT_NV);

        // Location 3: vertex buffer.
        m_rt_descriptor_set_generator.AddBinding(
            3,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV);

        // Location 4: index buffer.
        m_rt_descriptor_set_generator.AddBinding(
            4,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV);

        // Location 5: material buffer.
        m_rt_descriptor_set_generator.AddBinding(
            5,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV);

        // Location 6: textures.
        m_rt_descriptor_set_generator.AddBinding(
            6,
            1,  // TODO: number of textures
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV);

        m_rt_descriptor_pool = m_rt_descriptor_set_generator.GeneratePool(m_device);
        m_rt_descriptor_set_layout = m_rt_descriptor_set_generator.GenerateLayout(m_device);

        m_rt_descriptor_set =
            m_rt_descriptor_set_generator.GenerateSet(
                m_device,
                m_rt_descriptor_pool,
                m_rt_descriptor_set_layout);

        //
        // Bind resources.
        //

        // Bind top-level acceleration structure.
        VkWriteDescriptorSetAccelerationStructureNV descriptor_acceleration_structure_info = {};
        descriptor_acceleration_structure_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
        descriptor_acceleration_structure_info.accelerationStructureCount = 1;
        descriptor_acceleration_structure_info.pAccelerationStructures = &m_rt_top_level_as.m_structure;
        m_rt_descriptor_set_generator.Bind(m_rt_descriptor_set, 0, { descriptor_acceleration_structure_info });

        // Bind camera information.
        // TODO: which uniform buffer to bind? We have one per swap chain image!
        VkDescriptorBufferInfo descriptor_camera_info = {};
        descriptor_camera_info.buffer = m_uniform_buffers[0];
        descriptor_camera_info.range = sizeof(UniformBufferObject);
        m_rt_descriptor_set_generator.Bind(m_rt_descriptor_set, 2, { descriptor_camera_info });

        // Bind vertex buffer.
        VkDescriptorBufferInfo descriptor_vertex_buffer_info = {};
        descriptor_vertex_buffer_info.buffer = m_vertex_buffer;
        descriptor_vertex_buffer_info.range = VK_WHOLE_SIZE;
        m_rt_descriptor_set_generator.Bind(m_rt_descriptor_set, 3, { descriptor_vertex_buffer_info });

        // Bind index buffer.
        VkDescriptorBufferInfo descriptor_index_buffer_info = {};
        descriptor_index_buffer_info.buffer = m_index_buffer;
        descriptor_index_buffer_info.range = VK_WHOLE_SIZE;
        m_rt_descriptor_set_generator.Bind(m_rt_descriptor_set, 4, { descriptor_index_buffer_info });

        // Bind material buffer.
        VkDescriptorBufferInfo descriptor_material_buffer_info = {};
        descriptor_material_buffer_info.buffer = m_material_buffer;
        descriptor_material_buffer_info.range = VK_WHOLE_SIZE;
        m_rt_descriptor_set_generator.Bind(m_rt_descriptor_set, 5, { descriptor_material_buffer_info });

        // Bind textures.
        VkDescriptorImageInfo descriptor_texture_info = {};
        descriptor_texture_info.sampler = m_texture_sampler;
        descriptor_texture_info.imageView = m_texture_image_view;
        descriptor_texture_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        m_rt_descriptor_set_generator.Bind(m_rt_descriptor_set, 6, { descriptor_texture_info });

        // Copy bound resource handles into descriptor set.
        m_rt_descriptor_set_generator.UpdateSetContents(m_device, m_rt_descriptor_set);
    }

    void create_vk_rt_pipeline()
    {
        std::cout << "Creating Vulkan ray tracing pipeline..." << std::endl;

        const std::vector<char> ray_gen_shader_code = read_file("shaders/ray_gen.spv");
        const std::vector<char> ray_miss_shader_code = read_file("shaders/ray_miss.spv");
        const std::vector<char> ray_closest_hit_code = read_file("shaders/ray_closest_hit.spv");

        const VkShaderModule ray_gen_shader_module = create_vk_shader_module(ray_gen_shader_code);
        const VkShaderModule ray_miss_shader_module = create_vk_shader_module(ray_miss_shader_code);
        const VkShaderModule ray_closest_hit_module = create_vk_shader_module(ray_closest_hit_code);

        nv_helpers_vk::RayTracingPipelineGenerator pipeline_generator;
        pipeline_generator.SetMaxRecursionDepth(1);

        m_rt_ray_gen_index = pipeline_generator.AddRayGenShaderStage(ray_gen_shader_module);
        m_rt_ray_miss_index = pipeline_generator.AddMissShaderStage(ray_miss_shader_module);

        m_rt_hit_group_index = pipeline_generator.StartHitGroup();
        pipeline_generator.AddHitShaderStage(ray_closest_hit_module, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV);
        pipeline_generator.EndHitGroup();

        pipeline_generator.Generate(
            m_device,
            m_rt_descriptor_set_layout,
            &m_rt_pipeline,
            &m_rt_pipeline_layout);

        vkDestroyShaderModule(m_device, ray_closest_hit_module, nullptr);
        vkDestroyShaderModule(m_device, ray_miss_shader_module, nullptr);
        vkDestroyShaderModule(m_device, ray_gen_shader_module, nullptr);
    }

    void create_vk_rt_shader_binding_table()
    {
        std::cout << "Creating Vulkan ray tracing shader binding table..." << std::endl;

        m_rt_sbt_generator.AddRayGenerationProgram(m_rt_ray_gen_index, {});
        m_rt_sbt_generator.AddMissProgram(m_rt_ray_miss_index, {});
        m_rt_sbt_generator.AddHitGroup(m_rt_hit_group_index, {});

        const VkDeviceSize sbt_size = m_rt_sbt_generator.ComputeSBTSize(m_phsical_device_rt_props);

        // TODO: replace by vku function?
        nv_helpers_vk::createBuffer(
            m_physical_device,
            m_device,
            sbt_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            &m_rt_shader_binding_table_buffer,
            &m_rt_shader_binding_table_buffer_memory,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        m_rt_sbt_generator.Generate(m_device,
            m_rt_pipeline,
            m_rt_shader_binding_table_buffer,
            m_rt_shader_binding_table_buffer_memory);
    }

    // TODO: move to vku.
    void generate_vk_mipmaps(
        const VkImage           image,
        const std::uint32_t     width,
        const std::uint32_t     height,
        const std::uint32_t     mip_levels,
        const VkFormat          format)
    {
        VkFormatProperties format_properties;
        vkGetPhysicalDeviceFormatProperties(m_physical_device, format, &format_properties);

        if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
            throw std::runtime_error("Texture image format does not support blitting with linear filtering");

        VkCommandBuffer command_buffer =
            vku_begin_single_time_commands(m_device, m_transient_command_pool);

        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        std::int32_t mip_width = static_cast<std::int32_t>(width);
        std::int32_t mip_height = static_cast<std::int32_t>(height);

        for (std::uint32_t i = 1; i < mip_levels; ++i)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            VkImageBlit blit = {};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.srcOffsets[0] = { 0, 0, 0 };
            blit.srcOffsets[1] = { mip_width, mip_height, 1 };
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            blit.dstOffsets[0] = { 0, 0, 0 };
            blit.dstOffsets[1] =
            {
                mip_width > 1 ? mip_width / 2 : 1,
                mip_height > 1 ? mip_height / 2 : 1,
                1
            };

            vkCmdBlitImage(
                command_buffer,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR);  // TODO: try VK_FILTER_CUBIC_IMG

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            if (mip_width > 1) mip_width /= 2;
            if (mip_height > 1) mip_height /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        vku_end_single_time_commands(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            command_buffer);
    }

    void create_vk_texture_image()
    {
        std::cout << "Loading " << TexturePath << "..." << std::endl;

        int tex_width, tex_height, tex_channels;
        stbi_uc* texels = stbi_load(TexturePath.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);

        if (texels == nullptr)
            throw std::runtime_error("Failed to load texture file");

        const VkDeviceSize texture_size = tex_width * tex_height * 4;

        // TODO: wouldn't using std::ceil() be more correct?
        m_texture_mip_levels =
            static_cast<std::uint32_t>(
                std::log2(
                    std::max(tex_width, tex_height))) + 1;

        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;

        vku_create_buffer(
            m_physical_device,
            m_device,
            texture_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            staging_buffer,
            staging_buffer_memory);

        vku_copy_host_to_device(
            m_device,
            staging_buffer_memory,
            texels,
            texture_size);

        stbi_image_free(texels);

        vku_create_image(
            m_physical_device,
            m_device,
            static_cast<std::uint32_t>(tex_width),
            static_cast<std::uint32_t>(tex_height),
            static_cast<std::uint32_t>(m_texture_mip_levels),
            VK_SAMPLE_COUNT_1_BIT,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_texture_image,
            m_texture_image_memory);

        vku_transition_image_layout(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            m_texture_image,
            static_cast<std::uint32_t>(m_texture_mip_levels),
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        vku_copy_buffer_to_image(
            m_device,
            m_graphics_queue,
            m_transient_command_pool,
            m_texture_image,
            staging_buffer,
            static_cast<std::uint32_t>(tex_width),
            static_cast<std::uint32_t>(tex_height));

        generate_vk_mipmaps(
            m_texture_image,
            static_cast<std::uint32_t>(tex_width),
            static_cast<std::uint32_t>(tex_height),
            m_texture_mip_levels,
            VK_FORMAT_R8G8B8A8_UNORM);

        vkDestroyBuffer(m_device, staging_buffer, nullptr);
        vkFreeMemory(m_device, staging_buffer_memory, nullptr);
    }

    void create_vk_texture_image_view()
    {
        m_texture_image_view =
            vku_create_image_view(
                m_device,
                m_texture_image,
                m_texture_mip_levels,
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void create_vk_texture_sampler()
    {
        VkSamplerCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        create_info.magFilter = VK_FILTER_LINEAR;
        create_info.minFilter = VK_FILTER_LINEAR;
        create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        create_info.anisotropyEnable = VK_TRUE;
        create_info.maxAnisotropy = 16;
        create_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        create_info.unnormalizedCoordinates = VK_FALSE;
        create_info.compareEnable = VK_FALSE;
        create_info.compareOp = VK_COMPARE_OP_ALWAYS;
        create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        create_info.mipLodBias = 0.0f;
        create_info.minLod = 0.0f;
        create_info.maxLod = static_cast<float>(m_texture_mip_levels);

        if (vkCreateSampler(m_device, &create_info, nullptr, &m_texture_sampler) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan sampler");
    }

    void create_vk_descriptor_pool()
    {
        std::array<VkDescriptorPoolSize, 2> pool_sizes = {};

        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_sizes[0].descriptorCount = static_cast<std::uint32_t>(m_swap_chain_images.size());

        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_sizes[1].descriptorCount = static_cast<std::uint32_t>(m_swap_chain_images.size());

        VkDescriptorPoolCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        create_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        create_info.pPoolSizes = pool_sizes.data();
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

            VkDescriptorImageInfo image_info = {};
            image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_info.imageView = m_texture_image_view;
            image_info.sampler = m_texture_sampler;

            std::array<VkWriteDescriptorSet, 2> descriptor_writes = {};

            descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[0].dstSet = m_descriptor_sets[i];
            descriptor_writes[0].dstBinding = 0;
            descriptor_writes[0].dstArrayElement = 0;
            descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_writes[0].descriptorCount = 1;
            descriptor_writes[0].pBufferInfo = &buffer_info;

            descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[1].dstSet = m_descriptor_sets[i];
            descriptor_writes[1].dstBinding = 1;
            descriptor_writes[1].dstArrayElement = 0;
            descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptor_writes[1].descriptorCount = 1;
            descriptor_writes[1].pImageInfo = &image_info;

            vkUpdateDescriptorSets(
                m_device,
                static_cast<std::uint32_t>(descriptor_writes.size()),
                descriptor_writes.data(),
                0,
                nullptr);
        }
    }

    void create_vk_command_buffers()
    {
        std::cout << "Creating Vulkan command buffers..." << std::endl;

        m_command_buffers.resize(m_swap_chain_framebuffers.size());
        vku_allocate_command_buffers(
            m_device,
            m_command_pool,
            static_cast<std::uint32_t>(m_command_buffers.size()),
            m_command_buffers.data());

        for (std::size_t i = 0; i < m_command_buffers.size(); ++i)
        {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            if (vkBeginCommandBuffer(m_command_buffers[i], &begin_info) != VK_SUCCESS)
                throw std::runtime_error("Failed to begin recording Vulkan command buffer");

            std::array<VkClearValue, 2> clear_values;
            clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
            clear_values[1].depthStencil = { 1.0f, 0 };

            VkRenderPassBeginInfo render_pass_begin_info = {};
            render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            render_pass_begin_info.renderPass = m_render_pass;
            render_pass_begin_info.framebuffer = m_swap_chain_framebuffers[i];
            render_pass_begin_info.renderArea.offset = { 0, 0 };
            render_pass_begin_info.renderArea.extent = m_swap_chain_extent;
            render_pass_begin_info.clearValueCount = static_cast<std::uint32_t>(clear_values.size());
            render_pass_begin_info.pClearValues = clear_values.data();

            vkCmdBeginRenderPass(m_command_buffers[i], &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(m_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics_pipeline);

            const VkBuffer vertex_buffers[] = { m_vertex_buffer };
            const VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(m_command_buffers[i], 0, 1, vertex_buffers, offsets);

            vkCmdBindIndexBuffer(m_command_buffers[i], m_index_buffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdBindDescriptorSets(
                m_command_buffers[i],
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_graphics_pipeline_layout,
                0,
                1,
                &m_descriptor_sets[i],
                0,
                nullptr);

            vkCmdDrawIndexed(m_command_buffers[i], static_cast<std::uint32_t>(m_indices.size()), 1, 0, 0, 0);

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

        std::atomic<bool> stop = false;
        std::thread render_thread([this, &stop]()
        {
            std::cout << "Starting render thread..." << std::endl;

            const auto game_begin_time = std::chrono::high_resolution_clock::now();
            std::size_t current_frame = 0;

            while (!stop)
            {
                const auto frame_begin_time = std::chrono::high_resolution_clock::now();

                // "Game update".
                const float total_elapsed_time = std::chrono::duration<float, std::chrono::seconds::period>(frame_begin_time - game_begin_time).count();
                m_rotation_angle = total_elapsed_time * glm::radians(20.0f);

                // "Game render".
                draw_frame(current_frame);
                current_frame = (current_frame + 1) % MaxFramesInFlight;

#ifndef FORCE_VSYNC
                // Compute time spent rendering this frame.
                const float frame_time =
                    std::chrono::duration<float, std::chrono::seconds::period>(
                        std::chrono::high_resolution_clock::now() - frame_begin_time).count();

                // Sleep time until next frame.
                const float target_frame_time = 1.0f / RenderFrameRate;
                const float sleep_time = target_frame_time - frame_time;
                if (sleep_time > 0.0f)
                {
                    const auto sleep_time_us = static_cast<std::uint64_t>(sleep_time * 1000000.0f);
                    std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us));
                }
#endif
            }

            std::cout << "Ending render thread..." << std::endl;
        });

        while (!glfwWindowShouldClose(m_window))
        {
            glfwWaitEvents();
        }

        stop = true;
        render_thread.join();
    }

    void draw_frame(const std::size_t current_frame)
    {
        vkWaitForFences(m_device, 1, &m_in_flight_fences[current_frame], VK_TRUE, std::numeric_limits<std::uint64_t>::max());

        if (m_window_width == 0 || m_window_height == 0 || m_framebuffer_resized)
        {
            glfwGetFramebufferSize(m_window, &m_window_width, &m_window_height);
            if (m_window_width == 0 || m_window_height == 0)
                return;

            m_framebuffer_resized = false;
            recreate_vk_swap_chain();
        }

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

        if (CurrentRenderingMode == RenderingMode::RayTracing)
        {
            // TODO: copied from create_vk_command_buffers(), but begin_info.flags was changed from
            // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT to VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT.

            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(m_command_buffers[image_index], &begin_info) != VK_SUCCESS)
                throw std::runtime_error("Failed to begin recording Vulkan command buffer");

            VkImageSubresourceRange subresource_range;
            subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresource_range.baseMipLevel = 0;
            subresource_range.levelCount = 1;
            subresource_range.baseArrayLayer = 0;
            subresource_range.layerCount = 1;

            nv_helpers_vk::imageBarrier(
                m_command_buffers[image_index],
                m_swap_chain_images[image_index],
                subresource_range,
                0,          // srcAccessMask
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_GENERAL);

            update_vk_rt_render_target(m_swap_chain_image_views[image_index]);

            std::array<VkClearValue, 2> clear_values;
            clear_values[0].color = { 1.0f, 0.0f, 0.0f, 1.0f };
            clear_values[1].depthStencil = { 1.0f, 0 };

            VkRenderPassBeginInfo render_pass_begin_info = {};
            render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            render_pass_begin_info.renderPass = m_render_pass;
            render_pass_begin_info.framebuffer = m_swap_chain_framebuffers[image_index];
            render_pass_begin_info.renderArea.offset = { 0, 0 };
            render_pass_begin_info.renderArea.extent = m_swap_chain_extent;
            render_pass_begin_info.clearValueCount = static_cast<std::uint32_t>(clear_values.size());
            render_pass_begin_info.pClearValues = clear_values.data();

            vkCmdBeginRenderPass(
                m_command_buffers[image_index],
                &render_pass_begin_info,
                VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(
                m_command_buffers[image_index],
                VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
                m_rt_pipeline);

            vkCmdBindDescriptorSets(
                m_command_buffers[image_index],
                VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
                m_rt_pipeline_layout,
                0,          // firstSet
                1,          // descriptorSetCount
                &m_rt_descriptor_set,
                0,          // dynamicOffsetCount
                nullptr);   // pDynamicOffsets

            const VkDeviceSize ray_gen_offset = m_rt_sbt_generator.GetRayGenOffset();
            const VkDeviceSize ray_miss_offset = m_rt_sbt_generator.GetMissOffset();
            const VkDeviceSize ray_miss_stride = m_rt_sbt_generator.GetMissEntrySize();
            const VkDeviceSize hit_group_offset = m_rt_sbt_generator.GetHitGroupOffset();
            const VkDeviceSize hit_group_stride = m_rt_sbt_generator.GetHitGroupEntrySize();

            vkCmdTraceRaysNV(
                m_command_buffers[image_index],
                m_rt_shader_binding_table_buffer, ray_gen_offset,
                m_rt_shader_binding_table_buffer, ray_miss_offset, ray_miss_stride,
                m_rt_shader_binding_table_buffer, hit_group_offset, hit_group_stride,
                VK_NULL_HANDLE, 0, 0,
                m_swap_chain_extent.width,
                m_swap_chain_extent.height,
                1);

            vkCmdEndRenderPass(m_command_buffers[image_index]);

            if (vkEndCommandBuffer(m_command_buffers[image_index]) != VK_SUCCESS)
                throw std::runtime_error("Failed to end recording Vulkan command buffer");
        }

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
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
            recreate_vk_swap_chain();
        else if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to present Vulkan swap chain image");
    }

    void update_vk_uniform_buffer(const std::size_t image_index)
    {
        // Note: we're using the Y-up convention.
        UniformBufferObject ubo;
        ubo.m_model = glm::rotate(
            glm::mat4(1.0f),                    // initial transform
            m_rotation_angle,                   // angle
            glm::vec3(0.0f, 1.0f, 0.0f));       // axis
        ubo.m_view = glm::lookAt(
            glm::vec3(2.0f, 1.5f, 2.0f),        // eye
            glm::vec3(0.0f, 0.2f, 0.0f),        // center
            glm::vec3(0.0f, 1.0f, 0.0f));       // up
        ubo.m_proj = glm::perspective(
            glm::radians(45.0f),                // vertical FOV
            static_cast<float>(m_swap_chain_extent.width) / m_swap_chain_extent.height,
            0.1f,                               // Z-near
            10.0f);                             // Z-far

        // Account for GLM being initially designed for OpenGL.
        ubo.m_proj[1][1] *= -1.0f;

        vku_copy_host_to_device(
            m_device,
            m_uniform_buffers_memory[image_index],
            &ubo,
            static_cast<VkDeviceSize>(sizeof(ubo)));
    }

    void update_vk_rt_render_target(const VkImageView target_image_view)
    {
        // Bind target image.
        VkDescriptorImageInfo descriptor_output_image_info = {};
        descriptor_output_image_info.sampler = nullptr;
        descriptor_output_image_info.imageView = target_image_view;
        descriptor_output_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        m_rt_descriptor_set_generator.Bind(m_rt_descriptor_set, 1, { descriptor_output_image_info });

        // Copy bound resource handles into descriptor set.
        m_rt_descriptor_set_generator.UpdateSetContents(m_device, m_rt_descriptor_set);
    }

    void recreate_vk_swap_chain()
    {
        std::cout << "Recreating Vulkan swap chain..." << std::endl;

        vkDeviceWaitIdle(m_device);

        cleanup_vk_swap_chain();

        create_vk_swap_chain();
        create_vk_swap_chain_image_views();

        create_vk_render_pass();
        create_vk_graphics_pipeline();

        create_vk_color_resources();
        create_vk_depth_resources();
        create_vk_framebuffers();

        create_vk_command_buffers();
    }

    void cleanup_vk_swap_chain()
    {
        std::cout << "Cleaning up Vulkan swap chain..." << std::endl;

        vkDestroyImageView(m_device, m_depth_image_view, nullptr);
        vkDestroyImage(m_device, m_depth_image, nullptr);
        vkFreeMemory(m_device, m_depth_image_memory, nullptr);

        vkDestroyImageView(m_device, m_color_image_view, nullptr);
        vkDestroyImage(m_device, m_color_image, nullptr);
        vkFreeMemory(m_device, m_color_image_memory, nullptr);

        for (const VkFramebuffer framebuffer : m_swap_chain_framebuffers)
            vkDestroyFramebuffer(m_device, framebuffer, nullptr);

        vkFreeCommandBuffers(
            m_device,
            m_command_pool,
            static_cast<std::uint32_t>(m_command_buffers.size()),
            m_command_buffers.data());

        vkDestroyPipeline(m_device, m_graphics_pipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_graphics_pipeline_layout, nullptr);
        vkDestroyRenderPass(m_device, m_render_pass, nullptr);

        for (const VkImageView image_view : m_swap_chain_image_views)
            vkDestroyImageView(m_device, image_view, nullptr);

        vkDestroySwapchainKHR(m_device, m_swap_chain, nullptr);
    }

    void cleanup_vulkan()
    {
        std::cout << "Cleaning up Vulkan..." << std::endl;

        vkDeviceWaitIdle(m_device);

        vkDestroyBuffer(m_device, m_rt_shader_binding_table_buffer, nullptr);
        vkFreeMemory(m_device, m_rt_shader_binding_table_buffer_memory, nullptr);

        vkDestroyPipelineLayout(m_device, m_rt_pipeline_layout, nullptr);
        vkDestroyPipeline(m_device, m_rt_pipeline, nullptr);

        vkDestroyDescriptorSetLayout(m_device, m_rt_descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(m_device, m_rt_descriptor_pool, nullptr);

        destroy_vk_rt_acceleration_structure(m_rt_top_level_as);
        destroy_vk_rt_acceleration_structure(m_rt_bottom_level_as);

        cleanup_vk_swap_chain();

        vkDestroySampler(m_device, m_texture_sampler, nullptr);

        vkDestroyImageView(m_device, m_texture_image_view, nullptr);
        vkDestroyImage(m_device, m_texture_image, nullptr);
        vkFreeMemory(m_device, m_texture_image_memory, nullptr);

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

        if (EnableValidationLayers)
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

    void destroy_vk_rt_acceleration_structure(const AccelerationStructure& as)
    {
        vkDestroyBuffer(m_device, as.m_scratch_buffer, nullptr);
        vkFreeMemory(m_device, as.m_scratch_mem, nullptr);

        vkDestroyBuffer(m_device, as.m_result_buffer, nullptr);
        vkFreeMemory(m_device, as.m_result_mem, nullptr);

        vkDestroyBuffer(m_device, as.m_instances_buffer, nullptr);
        vkFreeMemory(m_device, as.m_instances_mem, nullptr);

        vkDestroyAccelerationStructureNV(m_device, as.m_structure, nullptr);
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
