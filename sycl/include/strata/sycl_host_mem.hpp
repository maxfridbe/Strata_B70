// include/strata/sycl_host_mem.hpp - the SYCL port: host memory that a running kernel can see the host write to.
//
// Strata's decode is a handshake through host-mapped memory: the host raises a flag and writes rows while a kernel on the
// GPU polls for them. On the Arc A770 (xe driver, oneAPI 2026.1) memory from sycl::malloc_host does not work for that: a kernel
// polling it never sees a host store made after the kernel started (measured: 60 million system-scope atomic loads, and
// every other load variant, saw nothing), while the same poll on memory from zeMemAllocHost marked uncached or
// write-combined sees the store within microseconds. Memory the handshake uses is therefore allocated through Level Zero
// directly, with those flags, when that is the backend.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <mutex>
#include <unordered_set>

namespace strata {
inline bool is_level_zero(sycl::queue& q) { return q.get_backend() == sycl::backend::ext_oneapi_level_zero; }

inline std::mutex& l0_mutex() { static std::mutex m; return m; }
inline std::unordered_set<void*>& l0_pointers() { static std::unordered_set<void*> s; return s; }   // what zeMemAllocHost gave out

/// What the host does with the memory while a kernel may be reading it.
enum class HostUse {
    kPlain,      ///< written before a kernel launches, or read by the host after it ends: the default allocation is fine
    kControl,    ///< polled inside a running kernel (a flag): uncached for the CPU and the GPU. Data a LATER kernel reads after it
                 ///< sees the flag is kPlain: a kernel launch refreshes the GPU's view, only a poll inside one launch is stale.
                 ///< Measured: uncached plan and result rows cost 42 ms per window on a path that never waits on the host.
    kRows,       ///< bulk data the host writes while a kernel is polling for it: write-combined, the fast way to write it
};

/// Host memory the GPU can read, and see the host's later writes to (for kControl and kRows). nullptr on failure.
/// Free it with host_free_coherent.
///
/// Measured on the A770: a plain allocation (zeMemAllocHost with no flags, and sycl::malloc_host) gave a polling kernel the
/// stale value for millions of polls in most runs, and only sometimes the new one; with ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED
/// or _BIAS_WRITE_COMBINED the kernel saw the host's store within microseconds every time.
inline void* host_alloc_coherent(size_t bytes, sycl::queue& q, HostUse use = HostUse::kControl) {
    // kPlain keeps the adapter's own allocation: it is faster for the host to read what the GPU wrote, and nothing polls it
    // STRATA_HOST_ALLOC=plain: every buffer from the adapter's allocation (the A/B arm; polled flags are then not seen on the A770)
    static const bool force_plain = [] { const char* e = std::getenv("STRATA_HOST_ALLOC"); return e != nullptr && std::string(e) == "plain"; }();
    if (use == HostUse::kPlain || force_plain || !is_level_zero(q)) return sycl::malloc_host(bytes, q);
    ze_context_handle_t ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    ze_host_mem_alloc_flags_t flags = use == HostUse::kControl ? ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED
                                                               : ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
    ze_host_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, flags};
    void* p = nullptr;
    if (zeMemAllocHost(ctx, &desc, bytes, 64, &p) != ZE_RESULT_SUCCESS) return nullptr;
    std::lock_guard<std::mutex> lk(l0_mutex());
    l0_pointers().insert(p);
    return p;
}

inline void host_free_coherent(void* p, sycl::queue& q) {
    if (p == nullptr) return;
    bool from_l0 = false;
    {
        std::lock_guard<std::mutex> lk(l0_mutex());
        from_l0 = l0_pointers().erase(p) != 0;
    }
    if (from_l0) zeMemFree(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()), p);
    else sycl::free(p, q);
}
}  // namespace strata
