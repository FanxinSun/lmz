"""Vulkan compute probe and dynamic loader for portable GPU decoding.

Provides discovery and initialization for Vulkan 1.1+ compute devices
(AMD, Intel, NVIDIA, and mobile SoCs) without requiring an SDK at build time.
The loader dynamically links to the system's Vulkan runtime if available.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys

# Vulkan basic types and constants
VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256

_vulkan_lib = None
_vulkan_probed = False
_vulkan_result: tuple[bool, str] = (False, "not probed")


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

    # Try ctypes.util.find_library first
    found = ctypes.util.find_library("vulkan")
    if found:
        candidates.insert(0, found)

    # Standard system paths
    standard_paths = [
        "/usr/lib", "/usr/local/lib", "/opt/homebrew/lib",
        "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"
    ]

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


class VkPhysicalDeviceLimits(ctypes.Structure):
    _fields_ = [("dummy", ctypes.c_byte * 512)]


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


def probe() -> tuple[bool, str]:
    """Probe for a usable Vulkan compute device.

    Returns (True, "vulkan:<Device Name>") if a Vulkan driver with compute
    support is present, or (False, "<reason>") if unavailable.
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
        # Bind functions
        vkCreateInstance = lib.vkCreateInstance
        vkCreateInstance.argtypes = [
            ctypes.POINTER(VkInstanceCreateInfo),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        vkCreateInstance.restype = ctypes.c_int32

        vkDestroyInstance = lib.vkDestroyInstance
        vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        vkDestroyInstance.restype = None

        vkEnumeratePhysicalDevices = lib.vkEnumeratePhysicalDevices
        vkEnumeratePhysicalDevices.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        vkEnumeratePhysicalDevices.restype = ctypes.c_int32

        vkGetPhysicalDeviceProperties = lib.vkGetPhysicalDeviceProperties
        vkGetPhysicalDeviceProperties.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(VkPhysicalDeviceProperties),
        ]
        vkGetPhysicalDeviceProperties.restype = None

        # Setup minimal instance
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext=None,
            pApplicationName=b"lmz",
            applicationVersion=1,
            pEngineName=b"lmz_engine",
            engineVersion=1,
            apiVersion=(1 << 22) | (1 << 12),  # Vulkan 1.1
        )
        create_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext=None,
            flags=0,
            pApplicationInfo=ctypes.pointer(app_info),
            enabledLayerCount=0,
            ppEnabledLayerNames=None,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
        )

        instance = ctypes.c_void_p()
        res = vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
        if res != VK_SUCCESS:
            _vulkan_probed = True
            _vulkan_result = (False, f"vkCreateInstance failed with code {res}")
            return _vulkan_result

        try:
            device_count = ctypes.c_uint32(0)
            res = vkEnumeratePhysicalDevices(instance, ctypes.byref(device_count), None)
            if res != VK_SUCCESS or device_count.value == 0:
                _vulkan_probed = True
                _vulkan_result = (False, "no Vulkan physical devices found")
                return _vulkan_result

            devices = (ctypes.c_void_p * device_count.value)()
            res = vkEnumeratePhysicalDevices(instance, ctypes.byref(device_count), devices)
            if res != VK_SUCCESS or device_count.value == 0:
                _vulkan_probed = True
                _vulkan_result = (False, "could not enumerate Vulkan devices")
                return _vulkan_result

            props = VkPhysicalDeviceProperties()
            vkGetPhysicalDeviceProperties(devices[0], ctypes.byref(props))
            dev_name = props.deviceName.decode("utf-8", "replace").strip("\x00")

            _vulkan_probed = True
            _vulkan_result = (True, f"vulkan:{dev_name}")
            return _vulkan_result
        finally:
            vkDestroyInstance(instance, None)

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
