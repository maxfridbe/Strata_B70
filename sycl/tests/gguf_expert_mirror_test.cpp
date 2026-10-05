// Host-only regression: run with run_gguf_expert_mirror_test.py (allocation shim, real GGUF reads).
#include "strata/core/gguf_expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <sycl/sycl.hpp>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace strata::kernels::cpu {
ExpertLayout test_layout;
const ExpertLayout& expert_layout() { return test_layout; }
}

// The base implementation normally lives in expert_source.cpp, which also contains GPU paths.
bool strata::core::ExpertSource::copy_blob(int64_t l, int64_t e, uint8_t* dst) {
    const uint8_t* b = blob(l, e);
    if (!b || !dst) return false;
    std::memcpy(dst, b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(l));
    return true;
}

int main() {
    auto& lay = strata::kernels::cpu::test_layout;
    lay.native = true;
    lay.n_layers = 2; lay.n_expert = 4; lay.max_blob = 6;
    lay.bytes = {6, 6};
    lay.fmt.resize(2);
    for (auto& f : lay.fmt) { f.up_off = 2; f.down_off = 4; }
    lay.gguf_off = {1, 9, 17, 25, 33, 41};
    char path[] = "/tmp/strata-mirror-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    unsigned char data[49];
    for (size_t i = 0; i < sizeof data; ++i) data[i] = (unsigned char) i;
    assert(write(fd, data, sizeof data) == sizeof data);
    strata::core::GgufExpertSource src;
    std::string err;
    assert(src.open(path, 2, 4, err));
    assert(src.mirror({{0, 1}, {0, 1}, {-1, 0}}, 256, 2, err, false) == 1);
    const auto* first = src.blob(0, 1);
    assert(first[0] == 3 && first[2] == 11 && first[4] == 19);
    assert(src.mirrored_bytes() == 256 && sycl::host_live.size() == 1);
    assert(src.mirror({{0, 1}, {0, 2}, {1, 0}}, 512, 2, err, true) == 2);
    assert(src.blob(0, 1) == first && src.device_alias(0, 3) == first);
    assert(src.pinned(0, 2) && src.pinned(1, 0) && !src.pinned(1, 1));
    assert(src.blob(1, 0)[0] == 25 && src.blob(1, 0)[4] == 41);
    assert(src.mirrored_bytes() == 768 && sycl::host_live.size() == 2);
    assert(src.mirror({{0, 1}}, 0, 1, err, true) == 0);
    sycl::host_fail = true;
    assert(src.mirror({{1, 1}}, 256, 1, err, true) == -1);
    sycl::host_fail = false;
    assert(src.blob(0, 1) == first && src.mirrored_bytes() == 768);
    // Reject a single blob that cannot fit a Level Zero allocation before allocating anything.
    lay.bytes[1] = (3ull << 30) + 256;
    assert(src.mirror({{1, 1}}, 4ull << 30, 1, err, true) == -1);
    lay.bytes[1] = 6;
    // A failed GGUF read releases only the new chunks; both append and replace preserve the old mirror.
    assert(ftruncate(fd, 1) == 0);
    for (bool append : {true, false}) {
        assert(src.mirror({{1, 1}}, 256, 2, err, append) == -1);
        assert(src.blob(0, 1) == first && sycl::host_live.size() == 2);
    }
    assert(pwrite(fd, data, sizeof data, 0) == sizeof data);
    assert(src.mirror({{1, 2}}, 256, 1, err, false) == 1);
    assert(!src.pinned(0, 1) && src.pinned(1, 2) && src.device_alias(0, 0) == nullptr);
    assert(src.mirrored_bytes() == 256 && sycl::host_live.size() == 1);
    assert(src.mirror({}, 0, 1, err, false) == 0);
    assert(src.mirrored_bytes() == 0 && sycl::host_live.empty());
    src.close();
    close(fd); unlink(path);
    std::puts("gguf_expert_mirror_test: passed");
}
