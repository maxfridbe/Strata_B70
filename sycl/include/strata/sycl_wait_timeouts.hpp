// include/strata/sycl_wait_timeouts.hpp - the SYCL port: how many bounded device waits gave up.
//
// A device wait for a flag the host raises is bounded (kSpinMax, sycl_doorbell.hpp) because an unbounded spin that never
// sees its flag wedges the card. The price is that a wait whose flag does not arrive in time simply lets the window go on,
// with whatever the host was supposed to write still missing, and nothing says so. Each wait that gives up counts itself
// here, in host-visible memory, and the verifier reads and clears the count when the window is done.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdlib>
#include "strata/sycl_doorbell.hpp"

namespace strata {
inline constexpr int kWaitKinds = 4;   // 0 = a flag nobody registered, 1.. = registered by wait_register
inline uint32_t* g_wait_timeouts = nullptr;
inline const volatile uint32_t* g_wait_flag[kWaitKinds] = {};
inline const char* g_wait_name[kWaitKinds] = {"other", "", "", ""};

/// Name a flag so a wait on it counts under that name (the verifier registers A, B and the CPU-served flag).
inline void wait_register(int kind, const uint32_t* flag, const char* name) {
    if (kind > 0 && kind < kWaitKinds) { g_wait_flag[kind] = flag; g_wait_name[kind] = name; }
}
inline int wait_kind(const uint32_t* flag) {
    for (int k = 1; k < kWaitKinds; ++k) if (g_wait_flag[k] == flag) return k;
    return 0;
}

inline uint32_t* wait_timeouts_counter(sycl::queue& q) {
    if (g_wait_timeouts == nullptr) {
        g_wait_timeouts = sycl::malloc_host<uint32_t>(kWaitKinds, q);
        if (g_wait_timeouts != nullptr) for (int k = 0; k < kWaitKinds; ++k) g_wait_timeouts[k] = 0;
    }
    return g_wait_timeouts;
}

/// How many polls a device wait makes before it gives up: kSpinMax, or STRATA_SPIN_MAX.
inline uint32_t spin_max() {
    static const uint32_t v = [] {
        const char* e = std::getenv("STRATA_SPIN_MAX");
        return e != nullptr && std::atoll(e) > 0 ? (uint32_t) std::atoll(e) : kSpinMax;
    }();
    return v;
}

/// The waits that gave up since the last call, per kind; clears the counts. Call after the queue has finished the window.
inline uint32_t wait_timeouts_take(uint32_t (&per_kind)[kWaitKinds]) {
    uint32_t total = 0;
    for (int k = 0; k < kWaitKinds; ++k) {
        per_kind[k] = g_wait_timeouts == nullptr ? 0u : sys_atomic_u32(g_wait_timeouts[k]).exchange(0u);
        total += per_kind[k];
    }
    return total;
}
}  // namespace strata
