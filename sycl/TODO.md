# SYCL port: speedup TODO

Ordered by how little card time each one needs to test. Measured numbers live in
[docs/INTEL_PERFORMANCE.md](../docs/INTEL_PERFORMANCE.md); each item is checked with benchy v1 (`sycl/benchy.sh`) and an
output-identity comparison against the build before it.

## 1. New hardware readiness (no card time)

The B70 trains at PCIe Gen3 x8 in its current slot (it can do Gen5 x16). IQ2_XS and Swift read their ~6,000
mirrored experts over that link at every decode step, and each start reads 30-43 GB from the SSD. A second card
lets a layer split hold the IQ2_XS without a host mirror.

- [x] Multi-card path checked without a second card (2026-10-04): `setup_intel.py --check` with two simulated
      cards now sizes models across both (it used the first card only) and names the Intel path; the config keeps
      `"gpu"`/`layer_split`, the server adds `--layer-split`, `strata-sycl.sh` opens the selector to every card,
      benchy reads every card. A second card of another die needs a two-target build (`AOT=bmg-g21,bmg-g31`,
      checked to compile).
- [ ] On the new box: benchy v1 before any code change, so the hardware's share of any speedup is known.
- [ ] With the second card: a layer split of the IQ2_XS (all experts resident) through setup and benchy.

## 2. Refill lent cache slots in the background (short-prompt test)

Above 32K context the prompt path borrows VRAM cache slots and refills them from the GGUF before the first decode
round: ~1 s per request (950 ms for 1,050 slots). With served 128K/256K configs that is most of the gap between a
2,185-token prompt at 479 tok/s and the 32K config's 790.

- [ ] Decode starts at once; the lent experts are read from the pinned host mirror over PCIe (the device plan
      already reads mirrored experts) while a background thread refills their VRAM slots.
- [ ] Test: Coder and IQ2_XS at 2,185 / 8K / 40K tokens, output tokens identical to the blocking refill, TTFT and
      decode against benchy v1.

## 3. Long prompts (parity tests + 40K-256K runs)

At 40K: expert GEMMs 23%, attention 19%, dequant 16%, QSA select 7%, host grouping 6.5%, gather 6%.

- [ ] **3a. Expert grouping on the GPU.** Each layer's routed (token, expert) pairs are grouped on the host after a
      sync. A device-side bucketing removes the per-layer round trip (6.5% at 40K).
- [ ] **3b. QSA block selection.** Every query against every pooled block: 7% at 40K, 9% of an 80K prompt, more at
      256K.
- [ ] **3c. Prompt attention's K/V gather.** The FP32 fallback kernel (19% at 40K); vector loads, sub-group
      cooperation. The XMX versions lost twice; this stays on the vector units.

## Later

- Fewer decode graph nodes (norm+rope, scores+top-k, gate+quantize fused): launch gaps are 15-20% of a round.
- Two draft branches per verify window (decode is latency-bound).
