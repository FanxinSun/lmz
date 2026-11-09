"""Vulkan compute decoder for portable GPU decoding.

Provides discovery, initialization, and rANS compute dispatch for Vulkan 1.1+
compute devices (AMD, Intel, NVIDIA, and Apple Silicon via MoltenVK) without
requiring an SDK at build time or third-party packages.
Pure ctypes against the system loader, standard library only.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
import sys
import time
from pathlib import Path

# Vulkan basic types and constants
VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
VK_STRUCTURE_TYPE_SUBMIT_INFO = 4
VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8
VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12
VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 11
VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16
VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29
VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34
VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35
VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39
VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR = 1000082000
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR = 1000177000

VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 0x00000001
VK_QUEUE_COMPUTE_BIT = 0x00000002
VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
VK_SHARING_MODE_EXCLUSIVE = 0

VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004

VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020
VK_PIPELINE_BIND_POINT_COMPUTE = 1

VK_QUERY_TYPE_TIMESTAMP = 2
VK_QUERY_RESULT_64_BIT = 0x00000002
VK_QUERY_RESULT_WAIT_BIT = 0x00000001
VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001
VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x00002000

VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002
VK_SUBGROUP_FEATURE_BALLOT_BIT = 0x00000008
VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256
VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF

# Module globals
_vulkan_lib = None
_vulkan_probed = False
_vulkan_result: tuple[bool, str] = (False, "not probed")
_last_error: str | None = None
_last_timing: dict | None = None
_global_context = None


def _find_vulkan_lib():
    """Locate the system Vulkan shared library."""
    global _vulkan_lib
    if _vulkan_lib is not None:
        return _vulkan_lib

    candidates = []
    if sys.platform == "darwin":
        candidates = ["libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"]
    elif sys.platform == "win32":
        candidates = ["vulkan-1.dll"]
    else:
        candidates = ["libvulkan.so.1", "libvulkan.so"]

    found = ctypes.util.find_library("vulkan")
    if found:
        candidates.insert(0, found)

    standard_paths = [
        "/usr/lib", "/usr/local/lib", "/opt/homebrew/lib",
        "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"
    ]
    dyld_fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
    if dyld_fallback:
        standard_paths = dyld_fallback.split(":") + standard_paths

    for cand in candidates:
        try:
            _vulkan_lib = ctypes.CDLL(cand)
            return _vulkan_lib
        except (OSError, Exception):
            pass
        for p in standard_paths:
            full = os.path.join(p, cand)
            if os.path.isfile(full):
                try:
                    _vulkan_lib = ctypes.CDLL(full)
                    return _vulkan_lib
                except (OSError, Exception):
                    pass
    return None


# --- CTYPES STRUCTURES ---

class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    ]


class VkExtensionProperties(ctypes.Structure):
    _fields_ = [
        ("extensionName", ctypes.c_char * 256),
        ("specVersion", ctypes.c_uint32),
    ]


class VkLayerProperties(ctypes.Structure):
    _fields_ = [
        ("layerName", ctypes.c_char * 256),
        ("specVersion", ctypes.c_uint32),
        ("implementationVersion", ctypes.c_uint32),
        ("description", ctypes.c_char * 256),
    ]


class VkPhysicalDeviceLimits(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_uint64 * 63)]  # 63 * 8 = 504 bytes, enforces 8-byte alignment

    @property
    def maxStorageBufferRange(self) -> int:
        raw_bytes = bytes(self.raw)
        return struct.unpack_from("<I", raw_bytes, 28)[0]   # offset checked against vulkan_core.h

    @property
    def maxComputeSharedMemorySize(self) -> int:
        raw_bytes = bytes(self.raw)
        return struct.unpack_from("<I", raw_bytes, 216)[0]

    @property
    def maxComputeWorkGroupInvocations(self) -> int:
        raw_bytes = bytes(self.raw)
        return struct.unpack_from("<I", raw_bytes, 232)[0]

    @property
    def timestampPeriod(self) -> float:
        raw_bytes = bytes(self.raw)
        return struct.unpack_from("<f", raw_bytes, 424)[0]


class VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * VK_MAX_PHYSICAL_DEVICE_NAME_SIZE),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("limits", VkPhysicalDeviceLimits),
        ("sparseProperties", ctypes.c_uint32 * 5),
    ]


class VkPhysicalDeviceProperties2(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("properties", VkPhysicalDeviceProperties),
    ]


class VkPhysicalDeviceSubgroupProperties(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("subgroupSize", ctypes.c_uint32),
        ("supportedStages", ctypes.c_uint32),
        ("supportedOperations", ctypes.c_uint32),
        ("quadOperationsInAllStages", ctypes.c_uint32),
    ]


class VkPhysicalDeviceFeatures(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_uint32 * 55)]

    @property
    def shaderInt64(self) -> int:
        return self.raw[40]  # offset 160 / 4

    @shaderInt64.setter
    def shaderInt64(self, val: int):
        self.raw[40] = val


class VkPhysicalDeviceFeatures2(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("features", VkPhysicalDeviceFeatures),
    ]


class VkPhysicalDevice8BitStorageFeaturesKHR(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("storageBuffer8BitAccess", ctypes.c_uint32),
        ("uniformAndStorageBuffer8BitAccess", ctypes.c_uint32),
        ("storagePushConstant8", ctypes.c_uint32),
    ]


class VkPhysicalDeviceShaderFloat16Int8FeaturesKHR(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("shaderFloat16", ctypes.c_uint32),
        ("shaderInt8", ctypes.c_uint32),
    ]


class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [
        ("queueFlags", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularity", ctypes.c_uint32 * 3),
    ]


class VkMemoryType(ctypes.Structure):
    _fields_ = [
        ("propertyFlags", ctypes.c_uint32),
        ("heapIndex", ctypes.c_uint32),
    ]


class VkMemoryHeap(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
    ]


class VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", VkMemoryType * 32),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", VkMemoryHeap * 16),
    ]


class VkDeviceQueueCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueFamilyIndex", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("pQueuePriorities", ctypes.POINTER(ctypes.c_float)),
    ]


class VkDeviceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueCreateInfoCount", ctypes.c_uint32),
        ("pQueueCreateInfos", ctypes.POINTER(VkDeviceQueueCreateInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
        ("pEnabledFeatures", ctypes.POINTER(VkPhysicalDeviceFeatures)),
    ]


class VkBufferCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("usage", ctypes.c_uint32),
        ("sharingMode", ctypes.c_uint32),
        ("queueFamilyIndexCount", ctypes.c_uint32),
        ("pQueueFamilyIndices", ctypes.POINTER(ctypes.c_uint32)),
    ]


class VkMemoryRequirements(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("memoryTypeBits", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
    ]


class VkMemoryAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("allocationSize", ctypes.c_uint64),
        ("memoryTypeIndex", ctypes.c_uint32),
    ]


class VkDescriptorSetLayoutBinding(ctypes.Structure):
    _fields_ = [
        ("binding", ctypes.c_uint32),
        ("descriptorType", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
        ("stageFlags", ctypes.c_uint32),
        ("pImmutableSamplers", ctypes.c_void_p),
    ]


class VkDescriptorSetLayoutCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("bindingCount", ctypes.c_uint32),
        ("pBindings", ctypes.POINTER(VkDescriptorSetLayoutBinding)),
    ]


class VkPushConstantRange(ctypes.Structure):
    _fields_ = [
        ("stageFlags", ctypes.c_uint32),
        ("offset", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
    ]


class VkPipelineLayoutCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("setLayoutCount", ctypes.c_uint32),
        ("pSetLayouts", ctypes.POINTER(ctypes.c_uint64)),
        ("pushConstantRangeCount", ctypes.c_uint32),
        ("pPushConstantRanges", ctypes.POINTER(VkPushConstantRange)),
    ]


class VkShaderModuleCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("codeSize", ctypes.c_size_t),
        ("pCode", ctypes.POINTER(ctypes.c_uint32)),
    ]


class VkPipelineShaderStageCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("stage", ctypes.c_uint32),
        ("module", ctypes.c_uint64),
        ("pName", ctypes.c_char_p),
        ("pSpecializationInfo", ctypes.c_void_p),
    ]


class VkComputePipelineCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("stage", VkPipelineShaderStageCreateInfo),
        ("layout", ctypes.c_uint64),
        ("basePipelineHandle", ctypes.c_uint64),
        ("basePipelineIndex", ctypes.c_int32),
    ]


class VkDescriptorPoolSize(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
    ]


class VkDescriptorPoolCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("maxSets", ctypes.c_uint32),
        ("poolSizeCount", ctypes.c_uint32),
        ("pPoolSizes", ctypes.POINTER(VkDescriptorPoolSize)),
    ]


class VkDescriptorSetAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("descriptorPool", ctypes.c_uint64),
        ("descriptorSetCount", ctypes.c_uint32),
        ("pSetLayouts", ctypes.POINTER(ctypes.c_uint64)),
    ]


class VkDescriptorBufferInfo(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("range", ctypes.c_uint64),
    ]


class VkWriteDescriptorSet(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("dstSet", ctypes.c_uint64),
        ("dstBinding", ctypes.c_uint32),
        ("dstArrayElement", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
        ("descriptorType", ctypes.c_uint32),
        ("pImageInfo", ctypes.c_void_p),
        ("pBufferInfo", ctypes.POINTER(VkDescriptorBufferInfo)),
        ("pTexelBufferView", ctypes.c_void_p),
    ]


class VkCommandPoolCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueFamilyIndex", ctypes.c_uint32),
    ]


class VkCommandBufferAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("commandPool", ctypes.c_uint64),
        ("level", ctypes.c_uint32),
        ("commandBufferCount", ctypes.c_uint32),
    ]


class VkCommandBufferBeginInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pInheritanceInfo", ctypes.c_void_p),
    ]


class VkSubmitInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("waitSemaphoreCount", ctypes.c_uint32),
        ("pWaitSemaphores", ctypes.c_void_p),
        ("pWaitDstStageMask", ctypes.c_void_p),
        ("commandBufferCount", ctypes.c_uint32),
        ("pCommandBuffers", ctypes.POINTER(ctypes.c_void_p)),
        ("signalSemaphoreCount", ctypes.c_uint32),
        ("pSignalSemaphores", ctypes.c_void_p),
    ]


class VkFenceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class VkQueryPoolCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queryType", ctypes.c_uint32),
        ("queryCount", ctypes.c_uint32),
        ("pipelineStatistics", ctypes.c_uint32),
    ]


class PushConsts(ctypes.Structure):
    _fields_ = [
        ("nstr", ctypes.c_uint32),
        ("plane", ctypes.c_uint32),
    ]


def _bind_vk_functions(lib):
    """Set argtypes and restype on Vulkan library functions."""
    lib.vkCreateInstance.argtypes = [ctypes.POINTER(VkInstanceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.vkCreateInstance.restype = ctypes.c_int32

    lib.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vkDestroyInstance.restype = None

    lib.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    lib.vkEnumeratePhysicalDevices.restype = ctypes.c_int32

    lib.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceProperties)]
    lib.vkGetPhysicalDeviceProperties.restype = None

    lib.vkGetPhysicalDeviceProperties2.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceProperties2)]
    lib.vkGetPhysicalDeviceProperties2.restype = None

    lib.vkGetPhysicalDeviceFeatures.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceFeatures)]
    lib.vkGetPhysicalDeviceFeatures.restype = None

    lib.vkGetPhysicalDeviceFeatures2.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceFeatures2)]
    lib.vkGetPhysicalDeviceFeatures2.restype = None

    lib.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(VkQueueFamilyProperties)]
    lib.vkGetPhysicalDeviceQueueFamilyProperties.restype = None

    lib.vkGetPhysicalDeviceMemoryProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceMemoryProperties)]
    lib.vkGetPhysicalDeviceMemoryProperties.restype = None

    lib.vkEnumerateDeviceExtensionProperties.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(VkExtensionProperties)]
    lib.vkEnumerateDeviceExtensionProperties.restype = ctypes.c_int32

    lib.vkEnumerateInstanceExtensionProperties.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(VkExtensionProperties)]
    lib.vkEnumerateInstanceExtensionProperties.restype = ctypes.c_int32

    lib.vkCreateDevice.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkDeviceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.vkCreateDevice.restype = ctypes.c_int32

    lib.vkDestroyDevice.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vkDestroyDevice.restype = None

    lib.vkGetDeviceQueue.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    lib.vkGetDeviceQueue.restype = None

    lib.vkCreateCommandPool.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkCommandPoolCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreateCommandPool.restype = ctypes.c_int32

    lib.vkDestroyCommandPool.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyCommandPool.restype = None

    lib.vkAllocateCommandBuffers.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkCommandBufferAllocateInfo), ctypes.POINTER(ctypes.c_void_p)]
    lib.vkAllocateCommandBuffers.restype = ctypes.c_int32

    lib.vkFreeCommandBuffers.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    lib.vkFreeCommandBuffers.restype = None

    lib.vkBeginCommandBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkCommandBufferBeginInfo)]
    lib.vkBeginCommandBuffer.restype = ctypes.c_int32

    lib.vkEndCommandBuffer.argtypes = [ctypes.c_void_p]
    lib.vkEndCommandBuffer.restype = ctypes.c_int32

    lib.vkCreateFence.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkFenceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreateFence.restype = ctypes.c_int32

    lib.vkDestroyFence.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyFence.restype = None

    lib.vkWaitForFences.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_uint64]
    lib.vkWaitForFences.restype = ctypes.c_int32

    lib.vkResetFences.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkResetFences.restype = ctypes.c_int32

    lib.vkQueueSubmit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(VkSubmitInfo), ctypes.c_uint64]
    lib.vkQueueSubmit.restype = ctypes.c_int32

    lib.vkQueueWaitIdle.argtypes = [ctypes.c_void_p]
    lib.vkQueueWaitIdle.restype = ctypes.c_int32

    lib.vkCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkBufferCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreateBuffer.restype = ctypes.c_int32

    lib.vkDestroyBuffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyBuffer.restype = None

    lib.vkGetBufferMemoryRequirements.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(VkMemoryRequirements)]
    lib.vkGetBufferMemoryRequirements.restype = None

    lib.vkAllocateMemory.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkMemoryAllocateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkAllocateMemory.restype = ctypes.c_int32

    lib.vkFreeMemory.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkFreeMemory.restype = None

    lib.vkBindBufferMemory.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    lib.vkBindBufferMemory.restype = ctypes.c_int32

    lib.vkMapMemory.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    lib.vkMapMemory.restype = ctypes.c_int32

    lib.vkUnmapMemory.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.vkUnmapMemory.restype = None

    lib.vkCreateShaderModule.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkShaderModuleCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreateShaderModule.restype = ctypes.c_int32

    lib.vkDestroyShaderModule.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyShaderModule.restype = None

    lib.vkCreateDescriptorSetLayout.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkDescriptorSetLayoutCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreateDescriptorSetLayout.restype = ctypes.c_int32

    lib.vkDestroyDescriptorSetLayout.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyDescriptorSetLayout.restype = None

    lib.vkCreatePipelineLayout.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkPipelineLayoutCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreatePipelineLayout.restype = ctypes.c_int32

    lib.vkDestroyPipelineLayout.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyPipelineLayout.restype = None

    lib.vkCreateComputePipelines.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(VkComputePipelineCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreateComputePipelines.restype = ctypes.c_int32

    lib.vkDestroyPipeline.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyPipeline.restype = None

    lib.vkCreateDescriptorPool.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkDescriptorPoolCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreateDescriptorPool.restype = ctypes.c_int32

    lib.vkDestroyDescriptorPool.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyDescriptorPool.restype = None

    lib.vkAllocateDescriptorSets.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkDescriptorSetAllocateInfo), ctypes.POINTER(ctypes.c_uint64)]
    lib.vkAllocateDescriptorSets.restype = ctypes.c_int32

    lib.vkUpdateDescriptorSets.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(VkWriteDescriptorSet), ctypes.c_uint32, ctypes.c_void_p]
    lib.vkUpdateDescriptorSets.restype = None

    lib.vkCreateQueryPool.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkQueryPoolCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.vkCreateQueryPool.restype = ctypes.c_int32

    lib.vkDestroyQueryPool.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
    lib.vkDestroyQueryPool.restype = None

    lib.vkGetQueryPoolResults.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.vkGetQueryPoolResults.restype = ctypes.c_int32

    lib.vkCmdResetQueryPool.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32]
    lib.vkCmdResetQueryPool.restype = None

    lib.vkCmdWriteTimestamp.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32]
    lib.vkCmdWriteTimestamp.restype = None

    lib.vkCmdBindPipeline.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
    lib.vkCmdBindPipeline.restype = None

    lib.vkCmdBindDescriptorSets.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    lib.vkCmdBindDescriptorSets.restype = None

    lib.vkCmdPushConstants.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    lib.vkCmdPushConstants.restype = None

    lib.vkCmdDispatch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    lib.vkCmdDispatch.restype = None


def _get_instance_extensions(lib) -> set[bytes]:
    count = ctypes.c_uint32(0)
    res = lib.vkEnumerateInstanceExtensionProperties(None, ctypes.byref(count), None)
    if res != VK_SUCCESS or count.value == 0:
        return set()
    props = (VkExtensionProperties * count.value)()
    res = lib.vkEnumerateInstanceExtensionProperties(None, ctypes.byref(count), props)
    if res != VK_SUCCESS:
        return set()
    return {p.extensionName.split(b"\x00")[0] for p in props}


def _get_device_extensions(lib, phys_dev) -> set[bytes]:
    count = ctypes.c_uint32(0)
    res = lib.vkEnumerateDeviceExtensionProperties(phys_dev, None, ctypes.byref(count), None)
    if res != VK_SUCCESS or count.value == 0:
        return set()
    props = (VkExtensionProperties * count.value)()
    res = lib.vkEnumerateDeviceExtensionProperties(phys_dev, None, ctypes.byref(count), props)
    if res != VK_SUCCESS:
        return set()
    return {p.extensionName.split(b"\x00")[0] for p in props}


def _check_device_support(lib, phys_dev) -> tuple[bool, str, int, float]:
    """Check if physical device satisfies all decoder requirements.
    
    Returns (ok, dev_name, queue_family, timestamp_period).
    """
    props = VkPhysicalDeviceProperties()
    lib.vkGetPhysicalDeviceProperties(phys_dev, ctypes.byref(props))
    dev_name = props.deviceName.decode("utf-8", "replace").strip("\x00")

    # Queue family with compute
    q_count = ctypes.c_uint32(0)
    lib.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, ctypes.byref(q_count), None)
    if q_count.value == 0:
        return False, f"{dev_name}: no queue families", -1, 0.0
    q_props = (VkQueueFamilyProperties * q_count.value)()
    lib.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, ctypes.byref(q_count), q_props)
    compute_q = -1
    for i, q in enumerate(q_props):
        if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0:
            compute_q = i
            break
    if compute_q < 0:
        return False, f"{dev_name}: no compute queue family", -1, 0.0

    # Limits check
    limits = props.limits
    if limits.maxComputeWorkGroupInvocations < 128:
        return False, f"{dev_name}: maxComputeWorkGroupInvocations {limits.maxComputeWorkGroupInvocations} < 128", -1, 0.0
    if limits.maxComputeSharedMemorySize < 19456:
        return False, f"{dev_name}: maxComputeSharedMemorySize {limits.maxComputeSharedMemorySize} < 19456", -1, 0.0

    # Subgroup properties
    sub_props = VkPhysicalDeviceSubgroupProperties(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        pNext=None
    )
    props2 = VkPhysicalDeviceProperties2(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        pNext=ctypes.cast(ctypes.byref(sub_props), ctypes.c_void_p)
    )
    lib.vkGetPhysicalDeviceProperties2(phys_dev, ctypes.byref(props2))

    if sub_props.subgroupSize < 8 or (sub_props.subgroupSize % 8) != 0:
        return False, f"{dev_name}: subgroupSize {sub_props.subgroupSize} not multiple of 8", -1, 0.0
    if (sub_props.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) == 0:
        return False, f"{dev_name}: subgroup ballot not supported", -1, 0.0

    # Features check
    feat_int8 = VkPhysicalDeviceShaderFloat16Int8FeaturesKHR(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
        pNext=None
    )
    feat_8bit = VkPhysicalDevice8BitStorageFeaturesKHR(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
        pNext=ctypes.cast(ctypes.byref(feat_int8), ctypes.c_void_p)
    )
    feat2 = VkPhysicalDeviceFeatures2(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        pNext=ctypes.cast(ctypes.byref(feat_8bit), ctypes.c_void_p)
    )
    lib.vkGetPhysicalDeviceFeatures2(phys_dev, ctypes.byref(feat2))

    if not feat2.features.shaderInt64:
        return False, f"{dev_name}: shaderInt64 not supported", -1, 0.0
    if not feat_8bit.storageBuffer8BitAccess:
        return False, f"{dev_name}: storageBuffer8BitAccess not supported", -1, 0.0
    if not feat_int8.shaderInt8:
        return False, f"{dev_name}: shaderInt8 not supported", -1, 0.0

    return True, dev_name, compute_q, limits.timestampPeriod


class VulkanContext:
    """Retained Vulkan context managing device, queue, pipelines and buffers."""

    def __init__(self, lib, instance, phys_dev, device, queue, queue_family, dev_name, timestamp_period, mem_props):
        self.lib = lib
        self.instance = instance
        self.phys_dev = phys_dev
        self.device = device
        self.queue = queue
        self.queue_family = queue_family
        self.dev_name = dev_name
        self.timestamp_period = timestamp_period
        self.mem_props = mem_props

        # The shader indexes its storage buffers with 32-bit offsets, and a descriptor
        # range may not exceed the device's maxStorageBufferRange. A batch over either
        # limit would wrap or fail validation, so decode_batch declines it instead.
        props = VkPhysicalDeviceProperties()
        self.lib.vkGetPhysicalDeviceProperties(self.phys_dev, ctypes.byref(props))
        self.max_buffer_bytes = min(props.limits.maxStorageBufferRange, 0xFFFFFFFF)

        # Create Command Pool
        pool_ci = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext=None,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.queue_family
        )
        self.cmd_pool = ctypes.c_uint64(0)
        res = self.lib.vkCreateCommandPool(self.device, ctypes.byref(pool_ci), None, ctypes.byref(self.cmd_pool))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkCreateCommandPool failed: {res}")

        # Allocate Command Buffer
        alloc_ci = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext=None,
            commandPool=self.cmd_pool,
            level=0,  # PRIMARY
            commandBufferCount=1
        )
        self.cmd_buf = ctypes.c_void_p()
        res = self.lib.vkAllocateCommandBuffers(self.device, ctypes.byref(alloc_ci), ctypes.byref(self.cmd_buf))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkAllocateCommandBuffers failed: {res}")

        # Create Fence
        fence_ci = VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            pNext=None,
            flags=0
        )
        self.fence = ctypes.c_uint64(0)
        res = self.lib.vkCreateFence(self.device, ctypes.byref(fence_ci), None, ctypes.byref(self.fence))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkCreateFence failed: {res}")

        # Create Query Pool
        query_ci = VkQueryPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            pNext=None,
            flags=0,
            queryType=VK_QUERY_TYPE_TIMESTAMP,
            queryCount=2,
            pipelineStatistics=0
        )
        self.query_pool = ctypes.c_uint64(0)
        res = self.lib.vkCreateQueryPool(self.device, ctypes.byref(query_ci), None, ctypes.byref(self.query_pool))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkCreateQueryPool failed: {res}")

        # Create Descriptor Set Layout (4 storage buffers)
        bindings = (VkDescriptorSetLayoutBinding * 4)(
            VkDescriptorSetLayoutBinding(binding=0, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
            VkDescriptorSetLayoutBinding(binding=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
            VkDescriptorSetLayoutBinding(binding=2, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
            VkDescriptorSetLayoutBinding(binding=3, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers=None),
        )
        layout_ci = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext=None,
            flags=0,
            bindingCount=4,
            pBindings=bindings
        )
        self.desc_layout = ctypes.c_uint64(0)
        res = self.lib.vkCreateDescriptorSetLayout(self.device, ctypes.byref(layout_ci), None, ctypes.byref(self.desc_layout))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkCreateDescriptorSetLayout failed: {res}")

        # Create Pipeline Layout with Push Constants
        pc_range = VkPushConstantRange(
            stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=ctypes.sizeof(PushConsts)
        )
        pipe_layout_ci = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext=None,
            flags=0,
            setLayoutCount=1,
            pSetLayouts=ctypes.pointer(self.desc_layout),
            pushConstantRangeCount=1,
            pPushConstantRanges=ctypes.pointer(pc_range)
        )
        self.pipe_layout = ctypes.c_uint64(0)
        res = self.lib.vkCreatePipelineLayout(self.device, ctypes.byref(pipe_layout_ci), None, ctypes.byref(self.pipe_layout))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkCreatePipelineLayout failed: {res}")

        # Create Descriptor Pool and Allocate Set
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=4
        )
        desc_pool_ci = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext=None,
            flags=0,
            maxSets=1,
            poolSizeCount=1,
            pPoolSizes=ctypes.pointer(pool_size)
        )
        self.desc_pool = ctypes.c_uint64(0)
        res = self.lib.vkCreateDescriptorPool(self.device, ctypes.byref(desc_pool_ci), None, ctypes.byref(self.desc_pool))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkCreateDescriptorPool failed: {res}")

        ds_alloc_info = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext=None,
            descriptorPool=self.desc_pool,
            descriptorSetCount=1,
            pSetLayouts=ctypes.pointer(self.desc_layout)
        )
        self.desc_set = ctypes.c_uint64(0)
        res = self.lib.vkAllocateDescriptorSets(self.device, ctypes.byref(ds_alloc_info), ctypes.byref(self.desc_set))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkAllocateDescriptorSets failed: {res}")

        # Pipelines cache
        self.pipelines: dict[str, ctypes.c_uint64] = {}
        self._init_pipelines()

        # Buffer allocations cache: name -> (handle, memory, size)
        self.buffers: dict[str, tuple[ctypes.c_uint64, ctypes.c_uint64, int]] = {}

    def _init_pipelines(self):
        dir_path = Path(__file__).resolve().parent
        for key, spv_name in [("direct", "lmz_rans.spv"), ("stride", "lmz_rans_stride.spv")]:
            spv_file = dir_path / spv_name
            if not spv_file.is_file():
                continue
            code = spv_file.read_bytes()
            assert len(code) % 4 == 0, f"{spv_name} size not multiple of 4"
            code_arr = (ctypes.c_uint32 * (len(code) // 4)).from_buffer_copy(code)

            sm_ci = VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext=None,
                flags=0,
                codeSize=len(code),
                pCode=code_arr
            )
            sm = ctypes.c_uint64(0)
            res = self.lib.vkCreateShaderModule(self.device, ctypes.byref(sm_ci), None, ctypes.byref(sm))
            if res != VK_SUCCESS:
                raise RuntimeError(f"vkCreateShaderModule for {spv_name} failed: {res}")

            stage_ci = VkPipelineShaderStageCreateInfo(
                sType=18,  # VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
                pNext=None,
                flags=0,
                stage=VK_SHADER_STAGE_COMPUTE_BIT,
                module=sm,
                pName=b"main",
                pSpecializationInfo=None
            )
            pipe_ci = VkComputePipelineCreateInfo(
                sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext=None,
                flags=0,
                stage=stage_ci,
                layout=self.pipe_layout,
                basePipelineHandle=0,
                basePipelineIndex=-1
            )
            pipe = ctypes.c_uint64(0)
            res = self.lib.vkCreateComputePipelines(self.device, 0, 1, ctypes.byref(pipe_ci), None, ctypes.byref(pipe))
            self.lib.vkDestroyShaderModule(self.device, sm, None)
            if res != VK_SUCCESS:
                raise RuntimeError(f"vkCreateComputePipelines for {spv_name} failed: {res}")
            self.pipelines[key] = pipe

    def find_memory_type(self, type_bits: int, required_flags: int, preferred_flags: int = 0) -> int:
        best_idx = None
        for i in range(self.mem_props.memoryTypeCount):
            if (type_bits & (1 << i)) != 0:
                flags = self.mem_props.memoryTypes[i].propertyFlags
                if (flags & required_flags) == required_flags:
                    if preferred_flags and (flags & preferred_flags) == preferred_flags:
                        return i
                    if best_idx is None:
                        best_idx = i
        if best_idx is not None:
            return best_idx
        raise RuntimeError(f"no memory type found for bits {bin(type_bits)} with flags {hex(required_flags)}")

    def ensure_buffer(self, name: str, size: int) -> tuple[ctypes.c_uint64, ctypes.c_uint64]:
        # Align size to 256 bytes
        size = max(64, (size + 255) & ~255)
        if name in self.buffers:
            buf, mem, cur_sz = self.buffers[name]
            if cur_sz >= size:
                return buf, mem
            self.lib.vkDestroyBuffer(self.device, buf, None)
            self.lib.vkFreeMemory(self.device, mem, None)

        buf_ci = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext=None,
            flags=0,
            size=size,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0,
            pQueueFamilyIndices=None
        )
        buf = ctypes.c_uint64(0)
        res = self.lib.vkCreateBuffer(self.device, ctypes.byref(buf_ci), None, ctypes.byref(buf))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkCreateBuffer for {name} ({size} B) failed: {res}")

        reqs = VkMemoryRequirements()
        self.lib.vkGetBufferMemoryRequirements(self.device, buf, ctypes.byref(reqs))

        req_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        pref_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | req_flags
        mem_idx = self.find_memory_type(reqs.memoryTypeBits, req_flags, pref_flags)

        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=None,
            allocationSize=reqs.size,
            memoryTypeIndex=mem_idx
        )
        mem = ctypes.c_uint64(0)
        res = self.lib.vkAllocateMemory(self.device, ctypes.byref(alloc_info), None, ctypes.byref(mem))
        if res != VK_SUCCESS:
            self.lib.vkDestroyBuffer(self.device, buf, None)
            raise RuntimeError(f"vkAllocateMemory for {name} ({reqs.size} B) failed: {res}")

        res = self.lib.vkBindBufferMemory(self.device, buf, mem, 0)
        if res != VK_SUCCESS:
            self.lib.vkDestroyBuffer(self.device, buf, None)
            self.lib.vkFreeMemory(self.device, mem, None)
            raise RuntimeError(f"vkBindBufferMemory for {name} failed: {res}")

        self.buffers[name] = (buf, mem, size)
        return buf, mem

    def upload(self, mem: ctypes.c_uint64, data: bytes | bytearray | memoryview):
        n = len(data)
        if n == 0:
            return
        p = ctypes.c_void_p()
        res = self.lib.vkMapMemory(self.device, mem, 0, n, 0, ctypes.byref(p))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkMapMemory upload failed: {res}")
        ctypes.memmove(p, (ctypes.c_char * n).from_buffer_copy(data), n)
        self.lib.vkUnmapMemory(self.device, mem)

    def download(self, mem: ctypes.c_uint64, dst: bytearray | memoryview, size: int):
        p = ctypes.c_void_p()
        res = self.lib.vkMapMemory(self.device, mem, 0, size, 0, ctypes.byref(p))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkMapMemory download failed: {res}")
        ctypes.memmove((ctypes.c_char * size).from_buffer(dst), p, size)
        self.lib.vkUnmapMemory(self.device, mem)

    def close(self):
        for buf, mem, _ in self.buffers.values():
            self.lib.vkDestroyBuffer(self.device, buf, None)
            self.lib.vkFreeMemory(self.device, mem, None)
        self.buffers.clear()

        for pipe in self.pipelines.values():
            self.lib.vkDestroyPipeline(self.device, pipe, None)
        self.pipelines.clear()

        if self.desc_pool.value:
            self.lib.vkDestroyDescriptorPool(self.device, self.desc_pool, None)
        if self.pipe_layout.value:
            self.lib.vkDestroyPipelineLayout(self.device, self.pipe_layout, None)
        if self.desc_layout.value:
            self.lib.vkDestroyDescriptorSetLayout(self.device, self.desc_layout, None)
        if self.query_pool.value:
            self.lib.vkDestroyQueryPool(self.device, self.query_pool, None)
        if self.fence.value:
            self.lib.vkDestroyFence(self.device, self.fence, None)
        if self.cmd_pool.value:
            self.lib.vkDestroyCommandPool(self.device, self.cmd_pool, None)
        if self.device:
            self.lib.vkDestroyDevice(self.device, None)
        if self.instance:
            self.lib.vkDestroyInstance(self.instance, None)


def _init_context(device_index: int | None = None) -> VulkanContext:
    lib = _find_vulkan_lib()
    if not lib:
        raise RuntimeError("no Vulkan loader or ICD found")
    _bind_vk_functions(lib)

    # Determine instance extensions and flags
    avail_exts = _get_instance_extensions(lib)
    instance_flags = 0
    enabled_exts: list[bytes] = []

    if b"VK_KHR_portability_enumeration" in avail_exts:
        enabled_exts.append(b"VK_KHR_portability_enumeration")
        instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR

    # Validation layer if requested
    enabled_layers: list[bytes] = []
    if os.environ.get("LMZ_VK_VALIDATE") == "1":
        enabled_layers.append(b"VK_LAYER_KHRONOS_validation")

    app_info = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext=None,
        pApplicationName=b"lmz",
        applicationVersion=1,
        pEngineName=b"lmz_engine",
        engineVersion=1,
        apiVersion=(1 << 22) | (1 << 12),  # Vulkan 1.1
    )

    ext_ptrs = (ctypes.c_char_p * len(enabled_exts))(*enabled_exts) if enabled_exts else None
    layer_ptrs = (ctypes.c_char_p * len(enabled_layers))(*enabled_layers) if enabled_layers else None

    create_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext=None,
        flags=instance_flags,
        pApplicationInfo=ctypes.pointer(app_info),
        enabledLayerCount=len(enabled_layers),
        ppEnabledLayerNames=layer_ptrs,
        enabledExtensionCount=len(enabled_exts),
        ppEnabledExtensionNames=ext_ptrs,
    )

    instance = ctypes.c_void_p()
    res = lib.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
    if res != VK_SUCCESS:
        raise RuntimeError(f"vkCreateInstance failed with code {res}")

    try:
        # Enumerate physical devices
        dev_count = ctypes.c_uint32(0)
        res = lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(dev_count), None)
        if res != VK_SUCCESS or dev_count.value == 0:
            raise RuntimeError(f"no Vulkan physical devices found (res={res})")

        devices = (ctypes.c_void_p * dev_count.value)()
        res = lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(dev_count), devices)
        if res != VK_SUCCESS or dev_count.value == 0:
            raise RuntimeError(f"could not enumerate Vulkan devices (res={res})")

        # Select physical device
        target_idx = None
        env_dev = os.environ.get("LMZ_VK_DEVICE")
        if device_index is not None:
            target_idx = device_index
        elif env_dev is not None:
            target_idx = int(env_dev)

        selected_dev = None
        selected_name = None
        selected_q = -1
        selected_ts_period = 0.0

        for idx, dev in enumerate(devices):
            ok, name, q_family, ts_period = _check_device_support(lib, dev)
            if ok:
                if target_idx is None or target_idx == idx:
                    selected_dev = dev
                    selected_name = name
                    selected_q = q_family
                    selected_ts_period = ts_period
                    break

        if selected_dev is None:
            if target_idx is not None and 0 <= target_idx < len(devices):
                _, reason, _, _ = _check_device_support(lib, devices[target_idx])
                raise RuntimeError(f"requested device {target_idx} not usable: {reason}")
            raise RuntimeError("no suitable Vulkan compute device found")

        # Memory properties
        mem_props = VkPhysicalDeviceMemoryProperties()
        lib.vkGetPhysicalDeviceMemoryProperties(selected_dev, ctypes.byref(mem_props))

        # Device extensions
        avail_dev_exts = _get_device_extensions(lib, selected_dev)
        dev_ext_list: list[bytes] = []
        for opt_ext in [b"VK_KHR_portability_subset", b"VK_KHR_8bit_storage", b"VK_KHR_shader_float16_int8"]:
            if opt_ext in avail_dev_exts:
                dev_ext_list.append(opt_ext)

        dev_ext_ptrs = (ctypes.c_char_p * len(dev_ext_list))(*dev_ext_list) if dev_ext_list else None

        # Device features chain
        feat_int8 = VkPhysicalDeviceShaderFloat16Int8FeaturesKHR(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
            pNext=None,
            shaderFloat16=0,
            shaderInt8=1,
        )
        feat_8bit = VkPhysicalDevice8BitStorageFeaturesKHR(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
            pNext=ctypes.cast(ctypes.byref(feat_int8), ctypes.c_void_p),
            storageBuffer8BitAccess=1,
            uniformAndStorageBuffer8BitAccess=0,
            storagePushConstant8=0,
        )
        feat2 = VkPhysicalDeviceFeatures2(
            sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            pNext=ctypes.cast(ctypes.byref(feat_8bit), ctypes.c_void_p),
        )
        feat2.features.shaderInt64 = 1

        priorities = (ctypes.c_float * 1)(1.0)
        q_create_info = VkDeviceQueueCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext=None,
            flags=0,
            queueFamilyIndex=selected_q,
            queueCount=1,
            pQueuePriorities=priorities,
        )

        dev_create_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=ctypes.cast(ctypes.byref(feat2), ctypes.c_void_p),
            flags=0,
            queueCreateInfoCount=1,
            pQueueCreateInfos=ctypes.pointer(q_create_info),
            enabledLayerCount=0,
            ppEnabledLayerNames=None,
            enabledExtensionCount=len(dev_ext_list),
            ppEnabledExtensionNames=dev_ext_ptrs,
            pEnabledFeatures=None,
        )

        device = ctypes.c_void_p()
        res = lib.vkCreateDevice(selected_dev, ctypes.byref(dev_create_info), None, ctypes.byref(device))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkCreateDevice failed with code {res}")

        queue = ctypes.c_void_p()
        lib.vkGetDeviceQueue(device, selected_q, 0, ctypes.byref(queue))

        return VulkanContext(
            lib=lib,
            instance=instance,
            phys_dev=selected_dev,
            device=device,
            queue=queue,
            queue_family=selected_q,
            dev_name=selected_name,
            timestamp_period=selected_ts_period,
            mem_props=mem_props,
        )
    except Exception:
        lib.vkDestroyInstance(instance, None)
        raise


def get_context(device_index: int | None = None) -> VulkanContext:
    global _global_context
    if _global_context is None:
        _global_context = _init_context(device_index)
    return _global_context


def probe() -> tuple[bool, str]:
    """Probe for a usable Vulkan compute device.

    Returns (True, "vulkan:<Device Name>") if a Vulkan driver with compute
    and subgroup ballot support is present, or (False, "<reason>") if unavailable.
    """
    global _vulkan_probed, _vulkan_result
    if _vulkan_probed:
        return _vulkan_result

    lib = _find_vulkan_lib()
    if not lib:
        _vulkan_probed = True
        _vulkan_result = (False, "no Vulkan loader or ICD found")
        return _vulkan_result

    try:
        ctx = get_context()
        _vulkan_probed = True
        _vulkan_result = (True, f"vulkan:{ctx.dev_name}")
        return _vulkan_result
    except Exception as exc:
        _vulkan_probed = True
        _vulkan_result = (False, f"Vulkan probe error: {exc}")
        return _vulkan_result


def available() -> bool:
    """True if a usable Vulkan compute device is detected."""
    ok, _ = probe()
    return ok


def device_name() -> str | None:
    """The detected Vulkan device string, or None."""
    ok, name = probe()
    return name if ok else None


def last_error() -> str | None:
    """The last error encountered during decode_batch, or None."""
    global _last_error
    return _last_error


def last_timing() -> dict | None:
    """GPU execution timing for the most recent decode_batch dispatch."""
    global _last_timing
    return _last_timing


def decode_batch(
    streams: bytes | bytearray | memoryview,
    offsets: bytes | bytearray | memoryview,
    nstr: int,
    plane: int,
    header: bytes | bytearray | memoryview,
    out: bytearray | memoryview | None = None,
    device_index: int | None = None,
    grid: int | None = None,
) -> bytearray | None:
    """Decode a batch of rANS streams on the Vulkan device.

    Parameters
    ----------
    streams : bytes | bytearray
        Concatenated stream bytes.
    offsets : bytes | bytearray
        2*nstr uint64 little-endian (offset, length) pairs.
    nstr : int
        Number of streams to decode.
    plane : int
        Size of each decoded output plane in bytes.
    header : bytes | bytearray
        516-byte shared frequency table header ('R1' + 256 uint16).
    out : bytearray | None
        Optional output buffer of size >= nstr * plane.
    device_index : int | None
        Optional index of the physical device to use.
    grid : int | None
        Optional workgroup grid count for grid-stride execution.

    Returns
    -------
    bytearray | None
        The decoded plane bytes, or None on failure (see last_error()).
    """
    global _last_error, _last_timing
    _last_error = None

    if nstr < 0:
        _last_error = "nstr is negative"
        return None
    if nstr == 0:
        return out if out is not None else bytearray(0)
    if plane <= 0:
        _last_error = f"invalid plane size: {plane}"
        return None
    if len(offsets) < nstr * 16:
        _last_error = f"offsets holds {len(offsets)} bytes, need {nstr * 16}"
        return None
    if header is None or len(header) != 516:
        _last_error = f"a shared table is 516 bytes, not {len(header) if header else 0}"
        return None

    out_needed = nstr * plane
    if out is None:
        out = bytearray(out_needed)
    elif len(out) < out_needed:
        _last_error = f"output buffer too small ({len(out)} < {out_needed})"
        return None

    try:
        ctx = get_context(device_index)
    except Exception as exc:
        _last_error = f"Vulkan context init failed: {exc}"
        return None

    largest = max(len(streams), out_needed, nstr * 16)
    if largest > ctx.max_buffer_bytes:
        _last_error = (f"batch needs a {largest}-byte storage buffer; this device allows "
                       f"{ctx.max_buffer_bytes} (maxStorageBufferRange, capped at the shader's "
                       f"32-bit indexing). Split the batch.")
        return None

    pipe_key = "stride" if (grid is not None) else "direct"
    if pipe_key not in ctx.pipelines:
        _last_error = f"SPIR-V pipeline '{pipe_key}' not available"
        return None
    pipeline = ctx.pipelines[pipe_key]

    try:
        # Upload buffers
        b_streams, m_streams = ctx.ensure_buffer("streams", len(streams))
        b_offsets, m_offsets = ctx.ensure_buffer("offsets", nstr * 16)
        b_out, m_out = ctx.ensure_buffer("out", out_needed)
        b_header, m_header = ctx.ensure_buffer("header", 516)

        ctx.upload(m_streams, streams)
        ctx.upload(m_offsets, offsets[:nstr * 16])
        ctx.upload(m_header, header)

        # Update descriptor set
        buf_infos = (VkDescriptorBufferInfo * 4)(
            VkDescriptorBufferInfo(buffer=b_streams, offset=0, range=VK_WHOLE_SIZE),
            VkDescriptorBufferInfo(buffer=b_offsets, offset=0, range=VK_WHOLE_SIZE),
            VkDescriptorBufferInfo(buffer=b_out, offset=0, range=VK_WHOLE_SIZE),
            VkDescriptorBufferInfo(buffer=b_header, offset=0, range=VK_WHOLE_SIZE),
        )
        writes = (VkWriteDescriptorSet * 4)(
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext=None, dstSet=ctx.desc_set, dstBinding=0, dstArrayElement=0,
                descriptorCount=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pImageInfo=None, pBufferInfo=ctypes.pointer(buf_infos[0]), pTexelBufferView=None
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext=None, dstSet=ctx.desc_set, dstBinding=1, dstArrayElement=0,
                descriptorCount=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pImageInfo=None, pBufferInfo=ctypes.pointer(buf_infos[1]), pTexelBufferView=None
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext=None, dstSet=ctx.desc_set, dstBinding=2, dstArrayElement=0,
                descriptorCount=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pImageInfo=None, pBufferInfo=ctypes.pointer(buf_infos[2]), pTexelBufferView=None
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext=None, dstSet=ctx.desc_set, dstBinding=3, dstArrayElement=0,
                descriptorCount=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pImageInfo=None, pBufferInfo=ctypes.pointer(buf_infos[3]), pTexelBufferView=None
            ),
        )
        ctx.lib.vkUpdateDescriptorSets(ctx.device, 4, writes, 0, None)

        # Determine dispatch size
        if grid is not None:
            gx = grid
        else:
            gx = (nstr + 15) // 16

        # Record Command Buffer
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext=None,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo=None
        )
        ctx.lib.vkBeginCommandBuffer(ctx.cmd_buf, ctypes.byref(begin_info))

        ctx.lib.vkCmdResetQueryPool(ctx.cmd_buf, ctx.query_pool, 0, 2)
        ctx.lib.vkCmdWriteTimestamp(ctx.cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.query_pool, 0)

        ctx.lib.vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        ctx.lib.vkCmdBindDescriptorSets(
            ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipe_layout, 0, 1, ctypes.byref(ctx.desc_set), 0, None
        )

        pc = PushConsts(nstr=nstr, plane=plane)
        ctx.lib.vkCmdPushConstants(
            ctx.cmd_buf, ctx.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, ctypes.sizeof(PushConsts), ctypes.byref(pc)
        )

        ctx.lib.vkCmdDispatch(ctx.cmd_buf, gx, 1, 1)

        ctx.lib.vkCmdWriteTimestamp(ctx.cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, 1)

        ctx.lib.vkEndCommandBuffer(ctx.cmd_buf)

        # Submit and Wait
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=None,
            waitSemaphoreCount=0,
            pWaitSemaphores=None,
            pWaitDstStageMask=None,
            commandBufferCount=1,
            pCommandBuffers=ctypes.pointer(ctx.cmd_buf),
            signalSemaphoreCount=0,
            pSignalSemaphores=None,
        )

        res = ctx.lib.vkResetFences(ctx.device, 1, ctypes.byref(ctx.fence))
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkResetFences failed: {res}")

        t_wall_start = time.perf_counter()
        res = ctx.lib.vkQueueSubmit(ctx.queue, 1, ctypes.byref(submit_info), ctx.fence)
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkQueueSubmit failed: {res}")

        res = ctx.lib.vkWaitForFences(ctx.device, 1, ctypes.byref(ctx.fence), 1, 10_000_000_000)  # 10s
        t_wall_end = time.perf_counter()
        if res != VK_SUCCESS:
            raise RuntimeError(f"vkWaitForFences failed or timed out: {res}")

        # Read timestamps
        ts = (ctypes.c_uint64 * 2)()
        res = ctx.lib.vkGetQueryPoolResults(
            ctx.device, ctx.query_pool, 0, 2, ctypes.sizeof(ts), ctypes.byref(ts), ctypes.sizeof(ctypes.c_uint64),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
        )
        if res == VK_SUCCESS and ts[1] >= ts[0]:
            ticks = ts[1] - ts[0]
            gpu_sec = (ticks * ctx.timestamp_period) * 1e-9
            _last_timing = {
                "gpu_seconds": gpu_sec,
                "gpu_ticks": ticks,
                "timestamp_period_ns": ctx.timestamp_period,
                "wall_seconds": t_wall_end - t_wall_start,
            }
        else:
            _last_timing = {
                "gpu_seconds": t_wall_end - t_wall_start,
                "gpu_ticks": 0,
                "timestamp_period_ns": ctx.timestamp_period,
                "wall_seconds": t_wall_end - t_wall_start,
            }

        # Download output
        ctx.download(m_out, out, out_needed)
        return out

    except Exception as exc:
        _last_error = f"decode_batch execution error: {exc}"
        return None
