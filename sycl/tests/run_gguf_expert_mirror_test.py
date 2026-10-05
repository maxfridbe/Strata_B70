"""Compile the real GGUF source with host allocation shims; no SYCL compiler or device required."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SYCL = r'''
#pragma once
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
namespace sycl {
using exception = std::runtime_error;
struct queue {};
inline bool host_fail = false;
inline std::unordered_map<void*, size_t> host_live;
inline void* malloc_host(size_t n, queue&) {
    if (n > (3ull << 30)) throw exception("allocation exceeds 3 GiB");
    if (host_fail) return nullptr;
    void* p = std::malloc(n);
    if (p) host_live.emplace(p, n);
    return p;
}
inline void free(void* p, queue&) {
    if (host_live.erase(p) != 1) throw exception("free of unowned allocation");
    std::free(p);
}
}
'''
DPCT = r'''
#pragma once
#include <sycl/sycl.hpp>
namespace dpct {
inline sycl::queue& get_in_order_queue() { static sycl::queue q; return q; }
}
'''
with tempfile.TemporaryDirectory(prefix="strata-mirror-test-") as tmp:
    d = Path(tmp)
    for name, content in (("sycl/sycl.hpp", SYCL), ("dpct/dpct.hpp", DPCT)):
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    exe = d / "test"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++20", "-O2", "-pthread",
                    "-I", str(d), "-I", str(ROOT / "sycl/include"), "-I", str(ROOT / "include"),
                    str(ROOT / "sycl/tests/gguf_expert_mirror_test.cpp"),
                    str(ROOT / "sycl/src/core/gguf_expert_source.cpp"), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
