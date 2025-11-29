#include "TheadSafeResourceManager.h"

void ThreadSafeResourceManager::createThreadCommandPools(vk::raii::Device &device, uint32_t queueFamilyIndex, uint32_t threadCount)
{
  std::lock_guard<std::mutex> lock(resourceMutex);

  commandPools.clear();

  for (uint32_t i = 0; i < threadCount; i++)
  {
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndex,
    };

    commandPools.emplace_back(device, poolInfo);
  }
}

vk::raii::CommandPool &ThreadSafeResourceManager::getCommandPool(uint32_t threadIndex)
{
  std::lock_guard<std::mutex> lock(resourceMutex);
  return commandPools[threadIndex];
}

void ThreadSafeResourceManager::allocateCommandBuffers(vk::raii::Device &device, uint32_t threadCount, uint32_t buffersPerThread)
{
  std::lock_guard<std::mutex> lock(resourceMutex);

  commandBuffers.clear();
  for (uint32_t i = 0; i < threadCount; i++)
  {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPools[i],
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = buffersPerThread,
    };

    auto threadBuffers = device.allocateCommandBuffers(allocInfo);
    for (const auto &buffer : threadBuffers)
    {
      commandBuffers.emplace_back(std::move(buffer));
    }
  }
}

vk::raii::CommandBuffer &ThreadSafeResourceManager::getCommandBuffer(uint32_t index)
{
  std::lock_guard<std::mutex> lock(resourceMutex);
  return commandBuffers[index];
}
