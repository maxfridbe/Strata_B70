// sycl/src/kernels/native_router_parity.cpp - the native fused router (softmax + top-10 + renormalise, one work-group
// per token) against a host reference, for the 256-expert Coder and the 512-expert geometry. It had no test of its own:
// router_top10_parity covers the generic kernel only.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {
constexpr double kClamp = 6.103515625e-05;

void reference(const float* logits, int n_expert, int k, int* ids, float* w) {
    double mx = logits[0];
    for (int e = 1; e < n_expert; ++e) mx = std::max(mx, (double) logits[e]);
    std::vector<double> p((size_t) n_expert);
    double sum = 0;
    for (int e = 0; e < n_expert; ++e) { p[(size_t) e] = std::exp((double) logits[e] - mx); sum += p[(size_t) e]; }
    for (auto& v : p) v /= sum;
    std::vector<int> idx((size_t) n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return p[(size_t) a] > p[(size_t) b]; });
    double s = 0;
    for (int i = 0; i < k; ++i) { ids[i] = idx[(size_t) i]; w[i] = (float) p[(size_t) idx[(size_t) i]]; s += p[(size_t) idx[(size_t) i]]; }
    s = std::max(s, kClamp);
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) w[i] / s);
}

int run_case(const char* name, int n_expert, int n_tok, const std::vector<float>& logits) {
    constexpr int K = 10;
    auto& q = dpct::get_in_order_queue();
    float* d_l = sycl::malloc_device<float>(logits.size(), q);
    int32_t* d_ids = sycl::malloc_device<int32_t>((size_t) n_tok * K, q);
    float* d_w = sycl::malloc_device<float>((size_t) n_tok * K, q);
    q.memcpy(d_l, logits.data(), logits.size() * 4).wait();
    q.memset(d_ids, 0xff, (size_t) n_tok * K * 4).wait();
    q.memset(d_w, 0, (size_t) n_tok * K * 4).wait();
    strata::kernels::native_router_top10_multi_ne(d_l, d_ids, d_w, n_tok, n_expert, &q);
    q.wait_and_throw();
    std::vector<int32_t> ids((size_t) n_tok * K);
    std::vector<float> w((size_t) n_tok * K);
    q.memcpy(ids.data(), d_ids, ids.size() * 4).wait();
    q.memcpy(w.data(), d_w, w.size() * 4).wait();
    int bad = 0;
    double worst = 0;
    for (int t = 0; t < n_tok; ++t) {
        int rid[K]; float rw[K];
        reference(logits.data() + (size_t) t * n_expert, n_expert, K, rid, rw);
        for (int i = 0; i < K; ++i) {
            const double dw = std::fabs((double) w[(size_t) t * K + i] - rw[i]);
            worst = std::max(worst, std::isnan(dw) ? 1e9 : dw);
            if (ids[(size_t) t * K + i] != rid[i] || !(dw <= 1e-5)) {
                if (bad < 3) std::printf("  %s token %d rank %d: got id %d w %g, want id %d w %g\n", name, t, i, ids[(size_t) t * K + i], w[(size_t) t * K + i], rid[i], rw[i]);
                ++bad;
            }
        }
    }
    std::printf("native_router %-22s ne=%d tokens=%d: %d mismatches, worst weight diff %.3g\n", name, n_expert, n_tok, bad, worst);
    sycl::free(d_l, q); sycl::free(d_ids, q); sycl::free(d_w, q);
    return bad;
}
}  // namespace

int main() {
    int bad = 0;
    std::mt19937 rng(7);
    for (int ne : {256, 512}) {
        for (int n_tok : {1, 6}) {
            std::vector<float> lg((size_t) n_tok * ne);
            std::normal_distribution<float> nd(0.f, 3.f);
            for (auto& v : lg) v = nd(rng);
            bad += run_case("gaussian", ne, n_tok, lg);
            for (auto& v : lg) v = -10.f + 0.01f * (float) (rng() % 1000);
            bad += run_case("negative-offset", ne, n_tok, lg);
            for (auto& v : lg) v = 1.f;
            bad += run_case("all-ties", ne, n_tok, lg);
        }
    }
    std::printf("native_router_parity: %d mismatches\n", bad);
    return bad ? 1 : 0;
}
