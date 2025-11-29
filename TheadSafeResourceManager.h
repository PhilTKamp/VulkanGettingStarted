#ifndef THREAD_SAFE_RESOURCE_MANAGER_H
#define THREAD_SAFE_RESOURCE_MANAGER_H

#include <vulkan/vulkan_raii.hpp>
#include <vector>

class ThreadSafeResourceManager
{
private:
  std::mutex resourceMutex;
  std::vector<vk::raii::CommandPool> commandPools;
  std::vector<vk::raii::CommandBuffer> commandBuffers;

public:
  /** Creates the command pools for each thread. */
  void createThreadCommandPools(vk::raii::Device &device, uint32_t queueFamilyIndex, uint32_t threadCount);

  /** Retrieves a command pool for a specific thread. */
  vk::raii::CommandPool &getCommandPool(uint32_t threadIndex);

  /** Allocates all of the command buffers, one for each thread. */
  void allocateCommandBuffers(vk::raii::Device &device, uint32_t threadCount, uint32_t buffersPerThread);

  /** Retrieves a command buffer for a specified index */
  vk::raii::CommandBuffer &getCommandBuffer(uint32_t index);
};

#endif