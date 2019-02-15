#include "vku.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>

std::uint32_t vku_find_memory_type(
    const VkPhysicalDevice          physical_device,
    const std::uint32_t             type_filter,
    const VkMemoryPropertyFlags     properties)
{
    VkPhysicalDeviceMemoryProperties physical_mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &physical_mem_properties);

    for (std::uint32_t i = 0; i < physical_mem_properties.memoryTypeCount; ++i)
    {
        if ((type_filter & (1UL << i)) &&
            (physical_mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }

    throw std::runtime_error("Failed to find suitable Vulkan memory type");
}

void vku_allocate_command_buffers(
    const VkDevice                  device,
    const VkCommandPool             command_pool,
    const std::uint32_t             command_buffer_count,
    VkCommandBuffer*                command_buffers)
{
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = command_buffer_count;

    if (vkAllocateCommandBuffers(device, &alloc_info, command_buffers) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate Vulkan command buffer(s)");
}

void vku_create_buffer(
    const VkPhysicalDevice          physical_device,
    const VkDevice                  device,
    const VkDeviceSize              size,
    const VkBufferUsageFlags        usage,
    const VkMemoryPropertyFlags     properties,
    VkBuffer&                       buffer,
    VkDeviceMemory&                 buffer_memory)
{
    VkBufferCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_info.size = size;
    create_info.usage = usage;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &create_info, nullptr, &buffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan buffer");

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
        vku_find_memory_type(physical_device, mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate Vulkan buffer memory");

    if (vkBindBufferMemory(device, buffer, buffer_memory, 0) != VK_SUCCESS)
        throw std::runtime_error("Failed to bind Vulkan buffer memory to buffer");
}

void vku_create_image(
    const VkPhysicalDevice          physical_device,
    const VkDevice                  device,
    const std::uint32_t             width,
    const std::uint32_t             height,
    const VkFormat                  format,
    const VkImageTiling             tiling,
    const VkImageUsageFlags         usage,
    const VkMemoryPropertyFlags     properties,
    VkImage&                        image,
    VkDeviceMemory&                 image_memory)
{
    VkImageCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    create_info.imageType = VK_IMAGE_TYPE_2D;
    create_info.format = format;
    create_info.extent.width = width;
    create_info.extent.height = height;
    create_info.extent.depth = 1;
    create_info.mipLevels = 1;
    create_info.arrayLayers = 1;
    create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    create_info.tiling = tiling;
    create_info.usage = usage;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &create_info, nullptr, &image) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan image");

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device, image, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
        vku_find_memory_type(physical_device, mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &alloc_info, nullptr, &image_memory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate Vulkan image memory");

    if (vkBindImageMemory(device, image, image_memory, 0) != VK_SUCCESS)
        throw std::runtime_error("Failed to bind Vulkan image memory to buffer");
}

VkCommandBuffer vku_begin_single_time_commands(
    const VkDevice                  device,
    const VkCommandPool             command_pool)
{
    VkCommandBuffer command_buffer;
    vku_allocate_command_buffers(device, command_pool, 1, &command_buffer);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin recording Vulkan command buffer");

    return command_buffer;
}

void vku_end_single_time_commands(
    const VkDevice                  device,
    const VkQueue                   queue,
    const VkCommandPool             command_pool,
    const VkCommandBuffer           command_buffer)
{
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);

    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

void vku_copy_host_to_device(
    const VkDevice                  device,
    const VkDeviceMemory            dest,
    const void*                     source,
    const VkDeviceSize              size)
{
    void* mapped_dest;
    if (vkMapMemory(device, dest, 0, size, 0, &mapped_dest) != VK_SUCCESS)
        throw std::runtime_error("Failed to map Vulkan buffer memory to host address space");

    std::memcpy(mapped_dest, source, static_cast<std::size_t>(size));

    vkUnmapMemory(device, dest);
}

void vku_copy_buffer_sync(
    const VkDevice                  device,
    const VkQueue                   queue,
    const VkCommandPool             command_pool,
    const VkBuffer                  dst_buffer,
    const VkBuffer                  src_buffer,
    const VkDeviceSize              size)
{
    VkCommandBuffer command_buffer =
        vku_begin_single_time_commands(device, command_pool);

    VkBufferCopy copy_region = {};
    copy_region.size = size;
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

    vku_end_single_time_commands(device, queue, command_pool, command_buffer);
}

void vku_copy_buffer_to_image(
    const VkDevice                  device,
    const VkQueue                   queue,
    const VkCommandPool             command_pool,
    const VkImage                   image,
    const VkBuffer                  buffer,
    const std::uint32_t             width,
    const std::uint32_t             height)
{
    VkCommandBuffer command_buffer =
        vku_begin_single_time_commands(device, command_pool);

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(
        command_buffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);

    vku_end_single_time_commands(device, queue, command_pool, command_buffer);
}

void vku_transition_image_layout(
    const VkDevice                  device,
    const VkQueue                   queue,
    const VkCommandPool             command_pool,
    const VkImage                   image,
    const VkFormat                  format,
    const VkImageLayout             old_layout,
    const VkImageLayout             new_layout)
{
    VkCommandBuffer command_buffer =
        vku_begin_single_time_commands(device, command_pool);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = 0;

    VkPipelineStageFlags source_stage;
    VkPipelineStageFlags destination_stage;

    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
        new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else throw std::runtime_error("Unsupported Vulkan layout transition");

    vkCmdPipelineBarrier(
        command_buffer,
        source_stage,
        destination_stage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    vku_end_single_time_commands(device, queue, command_pool, command_buffer);
}
