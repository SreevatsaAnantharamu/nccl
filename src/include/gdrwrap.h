/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GDRWRAP_H_
#define NCCL_GDRWRAP_H_

#include "nccl.h"
#include "alloc.h"
#include <stdint.h> // for standard [u]intX_t types
#include <stdio.h>
#include <stdlib.h>
#include <mutex>

// These can be used if the GDR library isn't thread safe
std::mutex& getGdrMutex();
#define GDRLOCKCALL(cmd, ret) do {                      \
    std::lock_guard<std::mutex> lock(getGdrMutex());   \
    ret = cmd;                                          \
} while(false)

#define GDRCHECK(cmd) do {                              \
    int e;                                              \
    /* GDRLOCKCALL(cmd, e); */                          \
    e = cmd;                                            \
    if( e != 0 ) {                                      \
      WARN("GDRCOPY failure %d", e);                    \
      return ncclSystemError;                           \
    }                                                   \
} while(false)

// This is required as the GDR memory is mapped WC
#if !defined(__NVCC__)
#if defined(__PPC__)
static inline void wc_store_fence(void) { asm volatile("sync" : : : "memory"); }
#elif defined(__x86_64__)
#include <immintrin.h>
static inline void wc_store_fence(void) { _mm_sfence(); }
#elif defined(__aarch64__)
#ifdef __cplusplus
#include <atomic>
static inline void wc_store_fence(void) { std::atomic_thread_fence(std::memory_order_release); }
#else
#include <stdatomic.h>
static inline void wc_store_fence(void) { atomic_thread_fence(memory_order_release); }
#endif
#endif
#endif

//#define GDR_DIRECT 1
#ifdef GDR_DIRECT
// Call the GDR API library code directly rather than via
// dlopen() wrappers
#include <gdrapi.h>

static ncclResult_t wrap_gdr_symbols(void) { return ncclSuccess; }
static gdr_t wrap_gdr_open(void) { gdr_t g = gdr_open(); return g; }
static ncclResult_t wrap_gdr_close(gdr_t g) { GDRCHECK(gdr_close(g)); return ncclSuccess; }
static ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle) {
  GDRCHECK(gdr_pin_buffer(g, addr, size, p2p_token, va_space, handle));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle) {
  GDRCHECK(gdr_unpin_buffer(g, handle));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info) {
  GDRCHECK(gdr_get_info(g, handle, info));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size) {
  GDRCHECK(gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size) {
  GDRCHECK(gdr_unmap(gdr_t g, gdr_mh_t handle, void **va, size_t size));
  return ncclSuccess;
}
static void wrap_gdr_runtime_get_version(int *major, int *minor) {
  gdr_runtime_get_version(major, minor);
  return ncclSuccess;
}
static void wrap_gdr_driver_get_version(gdr_t g, int *major, int *minor) {
  gdr_driver_get_version(g, major, minor);
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size) {
  GDRCHECK(gdr_copy_to_mapping(handle, map_d_ptr, h_ptr, size));
  return ncclSuccess;
}
static ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size) {
  GDRCHECK(gdr_copy_from_mapping(handle, h_ptr, map_d_ptr, size));
  return ncclSuccess;
}

#else
// Dynamically handle dependency the GDR API library

/* Extracted from gdrapi.h (v2.5.2 Mar 2026) */

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

struct gdr;
typedef struct gdr *gdr_t;

typedef struct gdr_mh_s {
  unsigned long h;
} gdr_mh_t;

typedef enum gdr_pin_flags {
    GDR_PIN_FLAG_DEFAULT = 0,
    GDR_PIN_FLAG_FORCE_PCIE = 1
} gdr_pin_flags_t;

typedef enum gdr_mapping_type {
    GDR_MAPPING_TYPE_NONE = 0,
    GDR_MAPPING_TYPE_WC = 1,
    GDR_MAPPING_TYPE_CACHING = 2,
    GDR_MAPPING_TYPE_DEVICE = 3,
    GDR_MAPPING_TYPE_MAX  //< For internal use. Not an actual type.
} gdr_mapping_type_t;

typedef struct gdr_info_v2 {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    // tm_cycles and cycles_per_ms are deprecated and will be removed in future.
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped:1;
    unsigned wc_mapping:1;
    gdr_mapping_type_t mapping_type;
} gdr_info_v2_t;
typedef gdr_info_v2_t gdr_info_t;

typedef enum gdr_map_flags {
    GDR_MAP_FLAG_DEFAULT = 0,
    GDR_MAP_FLAG_REQ_WC_MAPPING = 1,
    GDR_MAP_FLAG_REQ_CACHE_MAPPING = 2,
    GDR_MAP_FLAG_REQ_DEVICE_MAPPING = 3
} gdr_map_flags_t;

typedef enum gdr_attr {
    GDR_ATTR_USE_PERSISTENT_MAPPING = 1,    // Query whether gdrdrv uses persistent mapping
                                            // or traditional (non-persistent) mapping.

    GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE = 2, // Return non-zero if both gdrdrv and the GPU driver
                                              // support the GDR_PIN_FLAG_FORCE_PCIE feature.
                                              // Note that passing the flag may still lead to a run-time error,
                                              // for example when running on unsupported platforms.
    // For internal use only
    GDR_ATTR_MAX
} gdr_attr_t;

/* End of gdrapi.h */

ncclResult_t wrap_gdr_symbols(void);

gdr_t wrap_gdr_open(void);
ncclResult_t wrap_gdr_close(gdr_t g);
ncclResult_t wrap_gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle);
ncclResult_t wrap_gdr_pin_buffer_v2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t *handle);
ncclResult_t wrap_gdr_unpin_buffer(gdr_t g, gdr_mh_t handle);
ncclResult_t wrap_gdr_get_info_v2(gdr_t g, gdr_mh_t handle, gdr_info_v2_t *info);
#define wrap_gdr_get_info wrap_gdr_get_info_v2
ncclResult_t wrap_gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size);
ncclResult_t wrap_gdr_map_v2(gdr_t g, gdr_mh_t handle, void **ptr_va, size_t size, int flags);
ncclResult_t wrap_gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size);
ncclResult_t wrap_gdr_runtime_get_version(int *major, int *minor);
ncclResult_t wrap_gdr_driver_get_version(gdr_t g, int *major, int *minor);
ncclResult_t wrap_gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size);
ncclResult_t wrap_gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);
ncclResult_t wrap_gdr_get_attribute(gdr_t g, gdr_attr_t attr, int *v);
ncclResult_t wrap_gdr_get_mapping_type_string(gdr_mapping_type_t mapping_type, const char **pstr);

#endif // GDR_DIRECT

// Global GDR driver handle
extern gdr_t ncclGdrCopy;

#include "alloc.h"

typedef struct gdr_mem_desc {
  void *gdrDevMem;
  void *gdrMap;
  size_t gdrOffset;
  size_t gdrMapSize;
  gdr_mh_t gdrMh;
} gdr_mem_desc_t;

static gdr_t ncclGdrInit() {
  int libMajor, libMinor, drvMajor, drvMinor;
  gdr_t handle = NULL;
  // Dynamically load the GDRAPI library symbols
  if (wrap_gdr_symbols() == ncclSuccess) {
    handle = wrap_gdr_open();

    if (handle != NULL) {
      ncclResult_t res;

      // Query the version of libgdrapi
      NCCLCHECKGOTO(wrap_gdr_runtime_get_version(&libMajor, &libMinor), res, error);

      // Query the version of gdrdrv driver
      NCCLCHECKGOTO(wrap_gdr_driver_get_version(handle, &drvMajor, &drvMinor), res, error);

      // Only support GDRAPI 2.5 (Specifically need 2.5.2 for 6.17+ kernels) and later
      if (libMajor < 2 || (libMajor == 2 && libMinor < 5) || drvMajor < 2 || (drvMajor == 2 && drvMinor < 5)) {
        goto error;
      }
      else
        INFO(NCCL_INIT, "GDRCOPY enabled library %d.%d driver %d.%d", libMajor, libMinor, drvMajor, drvMinor);
    }
  }
  return handle;
error:
  if (handle != NULL) (void) wrap_gdr_close(handle);
  return NULL;
}

template <typename T>
static ncclResult_t ncclGdrCudaCalloc(T** ptr, T** devPtr, size_t nelem, void** gdrHandle, struct ncclMemManager* manager) {
  gdr_info_t info;
  size_t mapSize;
  gdr_mh_t mh;
  char *devMem;
  void *gdrMap;

  mapSize = ncclSizeOfT<T>()*nelem;

  // GDRCOPY Pinned buffer has to be a minimum of a GPU_PAGE_SIZE
  ALIGN_SIZE(mapSize, GPU_PAGE_SIZE);
  // GDRCOPY Pinned buffer has to be GPU_PAGE_SIZE aligned too
  NCCLCHECK(ncclCudaCalloc(&devMem, mapSize+GPU_PAGE_SIZE-1, manager));
  uint64_t alignedAddr = (((uint64_t) devMem) + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;
  size_t align = alignedAddr - (uint64_t)devMem;

  //TRACE(NCCL_INIT, "GDRCOPY: Pin buffer 0x%lx (%p) align %zu size %zu", alignedAddr, devMem, align, mapSize);
  const uint32_t flags = GDR_PIN_FLAG_FORCE_PCIE;
  NCCLCHECK(wrap_gdr_pin_buffer_v2(ncclGdrCopy, alignedAddr, mapSize, flags, &mh));

  NCCLCHECK(wrap_gdr_map(ncclGdrCopy, mh, &gdrMap, mapSize));
  //TRACE(NCCL_INIT, "GDRCOPY : mapped %p (0x%lx) at %p", devMem, alignedAddr, gdrMap);

  NCCLCHECK(wrap_gdr_get_info(ncclGdrCopy, mh, &info));

  // Will offset ever be non zero ?
  ssize_t off = info.va - alignedAddr;

  gdr_mem_desc_t* md;
  NCCLCHECK(ncclCalloc(&md, 1));
  md->gdrDevMem = devMem;
  md->gdrMap = gdrMap;
  md->gdrMapSize = mapSize;
  md->gdrOffset = off+align;
  md->gdrMh = mh;
  *gdrHandle = md;

  *ptr = (T *)((char *)gdrMap+off);
  if (devPtr) *devPtr = (T *)(devMem+off+align);

  TRACE(NCCL_INIT, "GDRCOPY : allocated devMem %p gdrMap %p offset %lx mh %lx mapSize %zu at %p",
       md->gdrDevMem, md->gdrMap, md->gdrOffset, md->gdrMh.h, md->gdrMapSize, *ptr);

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclGdrCudaCopy(void *gdrHandle, T* dst, T* src, size_t nelem) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  NCCLCHECK(wrap_gdr_copy_to_mapping(md->gdrMh, dst, src, nelem*ncclSizeOfT<T>()));
  return ncclSuccess;
}

static ncclResult_t ncclGdrCudaFree(void* gdrHandle, struct ncclMemManager* manager) {
  gdr_mem_desc_t *md = (gdr_mem_desc_t*)gdrHandle;
  NCCLCHECK(wrap_gdr_unmap(ncclGdrCopy, md->gdrMh, md->gdrMap, md->gdrMapSize));
  NCCLCHECK(wrap_gdr_unpin_buffer(ncclGdrCopy, md->gdrMh));
  NCCLCHECK(ncclCudaFree(md->gdrDevMem, manager));
  free(md);

  return ncclSuccess;
}

// Helper: Allocate memory accessible from CPU (either GDR or host memory)
template <typename T>
static ncclResult_t allocMemCPUAccessible(T **ptr, T **devPtr, size_t nelem, int host_flags,
                                          void **gdrHandle, struct ncclMemManager* manager, bool forceHost = false) {
  if (ncclGdrCopy && !forceHost) {
    NCCLCHECK(ncclGdrCudaCalloc(ptr, devPtr, nelem, gdrHandle, manager));
  } else {
    NCCLCHECK(ncclCuMemHostAlloc((void **)ptr, NULL, nelem * sizeof(T)));
    memset((void *)*ptr, 0, nelem * sizeof(T));
    *devPtr = *ptr;
    if (gdrHandle) *gdrHandle = NULL;  // Mark as host allocated by nulling GDR handle
  }
  return ncclSuccess;
}

// Helper: Free memory allocated by allocMemCPUAccessible
template <typename T>
static ncclResult_t freeMemCPUAccessible(T *ptr, void *gdrHandle, struct ncclMemManager* manager) {
  if (gdrHandle != NULL) {  // If a GDR handle exists, it was GDR memory
    NCCLCHECK(ncclGdrCudaFree(gdrHandle, manager));
  } else {  // Otherwise, it was host memory (or GDR was off)
    NCCLCHECK(ncclCuMemHostFree(ptr));
  }
  return ncclSuccess;
}

#endif // End include guard
