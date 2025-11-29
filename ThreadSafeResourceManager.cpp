#include "TheadSafeResourceManager.h"

void ThreadSafeResourceManager::createThreadCommandPools(vk::raii::Device &device, uint32_t queueFamilyIndex, uint32_t threadCount)
{
  std::lock_guard<std::mutex> lock(resourceMutex);

  commandPools.clear();
  commandBuffers.clear();

  for (uint32_t i = 0; i < threadCount; i++)
  {
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndex,
    };
    try
    {
      commandPools.emplace_back(device, poolInfo);
    }
    catch (const std::exception &)
    {
      throw; // Re-throw the exception to be caught by caller. Q: Again, why? Why bother catching it??
    }
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

  if (commandPools.size() < threadCount)
  {
    throw std::runtime_error("Not enough command pools for thread count!");
  }

  for (uint32_t i = 0; i < threadCount; i++)
  {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPools[i],
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = buffersPerThread,
    };

    try
    {

      auto threadBuffers = device.allocateCommandBuffers(allocInfo);
      for (auto &buffer : threadBuffers)
      {
        commandBuffers.emplace_back(std::move(buffer));
      }
    }
    catch (const std::exception &)
    {
      throw; // Re-throw the exception to be caught by the caller. Q: But why? why bother catching it if were just throwing it?
    }
  }
}

vk::raii::CommandBuffer &ThreadSafeResourceManager::getCommandBuffer(uint32_t index)
{
  if (index >= commandBuffers.size())
  {
    throw std::runtime_error("Command buffer index out of range: " + std::to_string(index) + " (available: " + std::to_string(commandBuffers.size()) + ")");
  }
  return commandBuffers[index];
}
