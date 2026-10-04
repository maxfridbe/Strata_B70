# Strata on an Intel Arc

Strata's engine is CUDA. On an Intel Arc it runs as **Strata's own engine, ported to SYCL** (`sycl/`, the
section "The engine itself on Intel" below). It sits behind the same Strata server, so the OpenAI and Anthropic
APIs, streaming, tool calls, MCP and the web app are all unchanged. llama.cpp's SYCL backend is the comparison
point: it runs the same GGUF, several times slower.

**Every measured speed is in [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md).** That covers this port, llama.cpp on
the same card, the history of each change, and results other people posted for the B50, the B580 and two B70s.
This file explains how the port works and how to run it.

Everything Intel-specific lives in `sycl/`. No shared file of upstream's is changed, so upstream merges stay clean.
`sycl/setup_intel.py` and `sycl/serve/server_intel.py` wrap upstream's `setup.py` and `serve/server.py` from
outside.

Written for and tested on an **Arc Pro B70 (32 GB)** running the Coder (IQ1_M) on Ubuntu 24.04.

## What you get

| | NVIDIA (Strata engine) | Intel Arc, Strata SYCL port | Intel Arc, llama.cpp |
|---|---|---|---|
| models | all four | the Coder IQ1_M; the original IQ2_XS and Swift 1.5 too (experts beyond VRAM in the host mirror) | the Coder (fits VRAM) |
| model files | the same GGUFs | the same GGUFs + a native pack | the same GGUFs |
| where the model lives | experts in RAM, hot ones on the card | every expert in VRAM (`--stream-experts`: no host copy); shard 2's lookup table read from the SSD by row | all of shard 1 on the card, shard 2 paged from disk |
| RAM needed | 32-64 GB | little (23 GB is fine) | little (23 GB is fine) |
| speculative decoding (MTP) | yes | yes (the base checkpoint's draft layer) | no |
| several cards | yes | layer split (`--layer-split`), tested by a user on 2x B70; `--peer-device` not yet | - |
| logprobs | - | `/v1/chat/completions` | - |
| images | yes | not yet | not yet |
| context | up to 262K | 256K (`--kv-resident`: the KV in pinned host memory, the attended window in VRAM) | 131K |
| speed | | [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md) | [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md) |

## Setup

Build the engine and its runtime image first ("How to build it" below), then:

    python3 sycl/setup_intel.py [setup.py's options, e.g. --model IQ2_XS --context 32768 --port 8085]

This is upstream's `setup.py`, run with the Intel steps swapped in. It imports setup.py and replaces those steps;
setup.py itself is unchanged. The model choice, download, pack, tokenizer, MTP draft layer, and the context and KV
questions are setup's own. What changes:

- **GPU check:** the Arc is found in sysfs (vendor 8086 under `xe` or `i915`) and offered through setup's AMD
  path. That is the path that builds locally and has no images.
- **Engine step:** it uses the SYCL build (`build-sycl-aot/strata`, run in the `strata-sycl-dev` image by
  `sycl/serve/strata-sycl.sh`) instead of compiling CUDA or HIP.
- **RAM rule:** setup's RAM rule does not apply. The CUDA engine keeps every expert in RAM; the port streams them
  from the GGUF into VRAM (`--stream-experts`), so RAM only decides the KV streaming. `--check` lists what fits by
  VRAM.
- **The config:**
  - It uses the container's paths and `"backend": "sycl"`.
  - The VRAM reserve is 1,024 MiB up to 32K, and 2,048 MiB with 4,096-token prompt chunks above that.
  - KV streaming (`--kv-resident 32768`) is on from 64K up when the RAM holds the KV.
  - A `model_switcher` or `sampling` block from an earlier config is kept.
  - `run-<model>.sh` starts `sycl/serve/server_intel.py`.

If a future `setup.py` drops a step this relies on, it stops with a message instead of writing a wrong config.
The models, packs and checkout must sit under the folder `strata-sycl.sh` mounts at `/work`. That is the folder
above the checkout, or `STRATA_SYCL_ROOT`.

Then `run-<model>.sh` (or `sycl/setup_intel.py` again) starts the model. `--port N` and `--host 0.0.0.0` work as
in upstream's setup.

**Several cards.** The image pins `ONEAPI_DEVICE_SELECTOR=level_zero:0`. `serve/strata-sycl.sh` forwards the
host's value when it is set. Without one, a `--layer-split` defaults to `level_zero:gpu`, so the engine sees every
card.

## Things that matter on this GPU

These apply to running llama.cpp by hand on the card.

- **`SYCL_CACHE_PERSISTENT` must be 0.** The persistent JIT cache segfaults on Xe2 during the first
  compile. The start script sets it; if you run llama-server by hand, do too.
- **The whole model goes on the card** (`--n-gpu-layers 999`), except `per_layer_token_embd.weight`, the single
  28.8 GB tensor of shard 2.
  - `--override-tensor per_layer_token_embd=CPU` keeps that tensor in host memory, mmapped and paged from the SSD
    by row, which is how the model's authors serve it.
  - Host RSS stays around 2 GB.
- **Thinking.** The model reasons before it answers. Strata's web app has the setting; for an API client, either:
  - put `{"reasoning_effort": "none"}` in `strata-<model>.shared-settings.json` next to the config, or
  - send `chat_template_kwargs: {"enable_thinking": false}` per request.

  Without one of those, a short `max_tokens` is spent entirely inside the think block and the answer looks empty.
- **`/health` says 503 while loading**; the server polls `/props` instead.

## How much context fits

This table is for llama.cpp. The architecture keeps the KV small: only every fourth layer is full attention (12 of
48). The other 36 are gated-delta-net layers, with a fixed-size recurrent state that does not grow with context.

- **What each token keeps:** the 12 attention layers keep 2 KV heads x 256 x (K + V) at q8_0, plus the
  sparse-attention indexer's keys.
- **Measured: about 20 KiB per token.** VRAM grows 0.6 GB per 32K of context with everything else unchanged.

| context | VRAM in use, model loaded | headroom on 32 GB | status |
|---|---|---|---|
| 32,768 | 28.4 GB | 3.5 GB | measured, the default |
| 65,536 | 29.0 GB | 2.9 GB | measured, loads |
| 98,304 | 29.6 GB | 2.3 GB | measured, loads |
| 131,072 | 30.3 GB | 1.5 GB | **measured, the practical ceiling**: a 104,798-token prompt read and answered |
| 163,840 | ~31.0 GB | ~0.9 GB | not attempted: under the 1.2 GB safety margin |
| 262,144 | ~32.6 GB | none | **does not fit - asking for it took the host down** |

**Do not ask for more than fits.**

- **What goes wrong:** on this driver, a GPU allocation past VRAM does not fail. The xe driver evicts buffers into
  host RAM, the kernel runs out of memory, and the machine livelocks until its hardware watchdog resets it.
- **It has happened:** a 262K request did exactly that on 2026-09-29. llama.cpp's own "failed to fit params" check
  fired too late to prevent it.
- **What to do:** compute the KV size first and leave 1.5 GB free.

## The engine itself on Intel: the SYCL port (`sycl/`)

This is Strata's own engine built for the Arc with oneAPI: the 50 CUDA kernels and the host code that drives them
(streams, events, graph capture, pinned memory). It is a migration of the tree, not a new backend: the engine has
no backend seam to slot into.

**Layout.** `sycl/src` and `sycl/include` mirror the tree and hold only the files the port changes. The SYCL build
(`sycl/CMakeLists.txt`) takes every other source from the original location, and no upstream file is edited.
`sycl/include/dpct/` is the vendored SYCLomatic helper library, so the port builds without the migration tool.

**How it was made, so it can be redone.**

1. **`sycl/tools/Dockerfile`:** the dev image. It is the llama.cpp SYCL image plus SYCLomatic (`dpct` 2025.3),
   ninja, and the CUDA 12.8 headers that `sycl/tools/get-cuda-headers.sh` pulls out of NVIDIA's pip wheels. dpct
   parses CUDA, so it needs the headers, not the toolkit.
2. **`sycl/tools/migrate.sh`:** writes a compilation database for the 86 CUDA-touching translation units and runs
   dpct over them. 85 migrate; dpct reports no line it could not migrate, and about 1,400 advisory notes.
3. **`sycl/tools/fixups.py`:** what dpct got wrong or could not do, as an idempotent script with a reason per item.
   The ones that mattered:
   - CUDA's null stream means the default stream; dpct turned it into a null `sycl::queue*`. Every stream cast now
     goes through `strata::q_of()` (`sycl/include/strata/sycl_queue.hpp`).
   - `__ldg((const float*) p)` came out as `*p`, reading one byte of a float scale (two sites, s_gemv).
   - `__fadd_rn(a, b ? c : d)` lost its parentheses.
   - The ggml lookup tables were threaded through kernel parameters with the wrong table per template. They are
     plain `static const` arrays read from device code now, as ggml-sycl does.
   - A free-memory query written as an `if` with an initializer was dropped entirely. That one is the layer
     split's `stage_room()`, and it made every later card report 0 bytes free.
   - Smaller items:
     - helper headers renamed since dpct 2025.3 (`entangle`, `chunked_partition`);
     - graph introspection and `cudaGraphUpload`, which have no SYCL equivalents;
     - `%globaltimer` (the stage profiler reads zeros).
4. **`sycl/tools/build.sh`:** configure and build with icpx inside the image. Two compiler flags are load-bearing:
   - `-fp-model=precise`: icpx defaults to a fast FP model.
   - `-cl-fp32-correctly-rounded-divide-sqrt` for the device compiler. The Arc's fp32 divide is not correctly
     rounded by default (OpenCL allows 2.5 ulp), and Strata's quantizers are byte-exact against ggml through
     `amax / 127`. Without the flag `quantize_act_parity` has 303k mismatches; with it, none.

**Parity tests.** The whole tree builds and links: the `strata` binary plus the kernel parity tests.

| parity test | result |
|---|---|
| bf16_gemv, cvec, dequant_s2, elementwise, gdn, gr, kv_q4, kv_q8, kv_stream, qsa, quantize_act, rope, router_top10, s2_gemv, s2_gemv_q8, s_gemv, s_gemv_q8k, sampler, shared_expert, iq_multi (IQ2_XS included) | pass |
| kv_hybrid_parity, qsa_prompt_attn_parity (int8 and fp16 XMX cases) | pass |
| qsa_prompt_attn_parity, Q4_0 tensor-core cases (#452) | "refused": the port does not have that mode |
| iq_parity, ple_parity, native_expert_parity | need fixtures or model files |
| s2_expert_grouped_parity | fails (the s2 path, unused here) |

**How to run it by hand.** This is a greedy test run, the way the engine numbers are measured. Run it inside the
`strata-sycl-dev` image, with the AOT build in `build-sycl-aot/`:

```
STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 \
build-sycl-aot/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> \
    --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtp rt dir> --max-context 8192 --tokens <ids> --max-new 64 --greedy
```

- **`--stream-experts`** (this port): no resident host copy of the experts. Every one of the 12,288 goes from the
  GGUF into the VRAM cache through a small staging ring (`GgufExpertSource`). Upstream needs 32 GB of RAM for this
  model; with the flag the engine runs on 23 GiB.
- **`STRATA_VERIFY_DEVICE_PLAN=1`:** the GPU plans each layer itself (upstream's E-6, off by default there).
- **`STRATA_VERIFY_NO_HOST=1`** (this port): the host waits for the whole window graph instead of per-layer
  rings. Only valid with every expert resident or in the pinned host mirror.
- **`--mtp`:** the base Qwen3.8-Flash-Next checkpoint's MTP draft layer (`tools/mtp_fetch.py fetch`,
  `mtp_pack.py --experts q2_0`, `mtp_rt.py`; 4.9 GB downloaded, 809 MiB of VRAM). It drafts for the Coder
  fine-tune with the same greedy tokens. The suffix drafter alone is rarely accepted on this card, which makes the
  draft layer the lever for decode.
- **IDs over 128 KB:** 80K-token ids exceed Linux's 128 KB single-argument limit, so use `--tokens-file`.

AOT device code is what runs: `AOT=bmg-g31 BUILD_DIR=.../build-sycl-aot`. Without AOT, the runtime JIT-compiles
every kernel on first use, which is slow the first time a process runs.
`SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` keeps the result across runs.

**What the port had to get right beyond compiling.** Each item is an entry in `sycl/tools/fixups.py` or a flag.

- **Host-mapped flags.** Strata's decode is a GPU/CPU handshake through host-mapped memory.
  - `volatile` device loads do not bypass the caches on Intel. System-scope atomics do, and they are the only thing
    measured to work (`sycl/probe/doorbell.cpp`, six variants).
  - Host-to-device visibility *during* a kernel stays unreliable on this platform. That is why the all-resident
    path waits for the window instead.
- **Bounded spins.** A device spin that never sees its flag is not a hang of one process.
  - The xe driver times the queue out and resets the GT node by node; a window graph has 2,400 nodes.
  - The card then stays wedged until a reboot (twice).
  - Every spin is capped (`kSpinMax`).
- **32-lane sub-groups.** The kernels are written for warps. dpct pinned 134 of 289 launches; the rest would run
  at Xe2's default 16 (`-fsycl-default-sub-group-size=32`).
- **Synchronous copies.** `cudaMemcpy` blocks, but dpct's default-queue `memcpy` did not wait. With the streaming
  ring, that filled the expert cache from overwritten buffers: non-deterministic residuals, NaN by layer 5. It was
  found with the per-layer residual ladder (`STRATA_VERIFY_DEBUG=1` prints R after every layer).
- **No host thread waits on a queue's event that another queue's barrier uses.**
  - Under the Level Zero v2 adapter, such an event no longer releases the other barrier, and the GPU waits forever.
  - The stager and the PLE upload mark their copies with a sequence number the copy queue writes into page-locked
    memory. The host polls that number instead.
  - See "Prompt-slot borrowing: the hang" below.
- **No non-char type punning in device code.** A `uint16_t*` read through an `int2` is not honoured by the SYCL
  device compiler; extract with shifts (see "Two model-specific bugs").
- **`native_expert_parity` was hand-ported.** The GPU native expert kernel matches ggml's float reference on real
  IQ1_M rows (rel 1.1e-2, the same class as the CPU path).

**Profiling.**

- `sycl/benchy.sh` (benchy v1) runs the standard bench (`sycl/tools/perf_matrix.py`): every model x the v1 prompt
  sizes, with each model's serve config, from a cold page cache. Its report is what INTEL_PERFORMANCE.md asks
  submitters to post.

- `sycl/tools/Dockerfile.unitrace` builds the dev image with Intel's unitrace. Run `unitrace -d` around the engine,
  then `sycl/rank_kernels.py`, for device time per kernel.
- `strata-sycl-dev:metrics`, with Intel's metrics libraries and `dev.xe.observation_paranoid=0`, gives hardware
  counters.
- `mmvq_bench`, `mmvq_sg_bench`, `q6k_align_bench`, `xmx_gemm_bench`, `xmx_int8_bench` and
  `native_expert_parity NATIVE_BENCH=1` time kernels in isolation. Warm the clocks first: a 5 ms run measures the
  ramp, not the kernel.
- `STRATA_PLE_TRACE=1` traces each PLE gather.
- `STRATA_DBG_NAN=1` reports the first non-finite values per layer, including the experts' fp16 GEMM inputs.

**Two traps worth knowing.**

- **`--prefill-until N` with a native pack** does not feed the rest of the prompt through the token loop (that
  loop is skipped for native packs). The tokens after N are dropped and the model free-runs. Compare output tokens
  between paths, never only timings.
- **Identical greedy runs can decode at two speeds.** One PLE read stall lands either in the prompt's PLE wait or
  in the first decode round. It is a once-per-process cost, not lost throughput, so compare runs on time to first
  token plus decode.

### How the prompt path uses VRAM

- **The VRAM plan.** `--expert-cache auto` fills the card down to `--vram-reserve-mib` *before* two things exist:
  the KV state and the prompt chunk buffers. With too small a reserve at long contexts, the driver starts
  migrating buffers and the run never finishes. Setup's reserves (1,024 MiB to 32K; 2,048 MiB with
  `--prefill 4096` above) leave room.
- **Streamed experts.** Experts the cache does not hold are copied for every chunk, in one of two walks:
  - **Stream-all:** every non-resident expert, layer by layer ahead of the compute. This is the default when the
    VRAM holds more than 90% of the (layer, expert) pairs.
  - **Routed-only:** only the experts the chunk routes to (`STRATA_PREFILL_RING=8`). This is the default otherwise.
    Past 90%, stream-all would copy several times the routed experts.

  `STRATA_PREFILL_STREAM_ALL=1` / `=0` force a walk. The stager threads read the blobs themselves; an early version
  held `GgufExpertSource::blob()` pointers across reads of the ring, and those blobs were overwritten before they
  were copied.
- **Prompt-slot borrowing.**
  - **Without it:** the reserve evicts experts from VRAM for good, and decode after a long prompt is slow.
  - **With it:** the prompt path borrows cache slots for its buffers and refills them in about a second afterwards.
  - **The cost:** a second on short prompts. So the port borrows by default only above a 32K context;
    `--prefill-borrow` / `--no-prefill-borrow` decide explicitly.
- **KV streaming** (`--kv-resident`) keeps the whole KV in pinned host memory and only the attended window in VRAM,
  so the KV pushes no experts out. Setup turns it on from 64K up and keeps INT8, the faster KV format at every size
  measured.
  - `--kv k8v4` does not support streaming yet. Setup keeps its KV in VRAM, which pushes more experts out to the
    host mirror.
  - k8v4 reads prompts fastest, because it needs no streaming copies.
- **The pinned host mirror.** At start-up, every expert without a VRAM slot is read into pinned host memory
  (`STRATA_MIRROR_MIB`; by default free RAM less 4 GiB). The device-built verify plan points the expert kernels
  straight at it over PCIe. That is how a model bigger than VRAM runs: the original IQ2_XS keeps about a quarter of
  its experts there.

### XMX (Intel's matrix engine)

oneMKL's FP16 GEMMs already run on the XMX units, so the prompt path is bound by the dequant that feeds them, not by
the products. Every hand-written joint_matrix kernel so far is correct but loses to the existing paths on this card
(numbers in [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md), "XMX experiments").

- **`xmx_gemm_iq`:** a fused dequant + FP16 GEMM straight from the quantized rows.
- **`qsa_prompt_attn_xmx` v1 and v2:** the port of the mma.sync prompt attention, opt-in
  (`STRATA_PROMPT_ATTN_XMX=1` for 64-cell chunks with 120 KB of local memory, or `=32`).
  - This attention is gather-bound: each query position selects its own ~2,000 cells, so the K/V fetch dominates.
  - Only 12 of the 16 matrix rows are real heads.
  - Grouping neighbouring positions would cut the gather but multiply the arithmetic, because their selections
    overlap little. It was not built.
- **Expert dot products on int8 DPAS for decode:** opt-in `STRATA_EXPERT_XMX=1`; do not enable. At 1-6 rows, the
  grid decode and the packed-B layout cost more than the DPAS saves, and one version hung the GPU.
- **An int8 DPAS GEMM straight from IQ4_NL for prompts** (`xmx_int8_bench`, standalone).
  - One 32-element block per DPAS, rescaled by d_x * d_w after each.
  - The per-block rescale keeps the matrix engine waiting.
  - The dequant it would save is small at expert size.
  - Not ported.

### Serving the port

`serve/server.py --engine strata` runs the SYCL engine unchanged through `sycl/serve/strata-sycl.sh`. That script
is an `exe` that starts the binary inside the oneAPI runtime image, with the serve pipes attached. Paths in the
config's `args` are the container's, with the data root mounted at `/work`. A config:

```json
{"engine": "strata", "exe": "<repo>/sycl/serve/strata-sycl.sh",
 "args": ["--pack", "/work/pack", "--native", "<shard 1>", "--ple-gguf", "<shard 2>",
          "--expert-profile", "data/expert-profile-coder.bin", "--expert-cache", "auto", "--stream-experts",
          "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/work/mtp/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}, ...}
```

**The reserve matters.** With `--stream-experts` there is no host copy of the experts beyond the pinned mirror. Any
expert left out of both is read from the SSD and computed on the CPU whenever it is routed, and decode collapses.
1,024 MiB fits all 12,288 Coder experts at 32K.

**Start time.** Starting a model is mostly the expert cache's fill from the GGUF. The fill is a pipeline:

- the slots are admitted in profile order first (the same placement);
- the reads go in file order, by up to 8 threads, into page-locked batches of 64;
- a batch's copies run while the next batch is read.

`STRATA_FILL_SERIAL=1` is the old fill: three slices read per expert, then a copy, then a wait.

**Logprobs.** `/v1/chat/completions` takes OpenAI's `"logprobs": true` and `"top_logprobs": 0..20`, streamed or
not, through `sycl/serve/server_intel.py`.

- **The reply** carries `choices[0].logprobs.content[]` (`token`, `logprob`, `bytes`, `top_logprobs`). With thinking
  on, the thinking tokens go under `reasoning_content`.
- **The engine** (the port only) gets `logprobs=K` on its GEN line. It writes an `LP logprob id:logprob ...` line
  after each `T` line, from the verify window's head logits. That is the distribution the token was taken from,
  before temperature, penalties and sampling.
- **Compatibility:** upstream's server ignores the lines, and an engine that is never asked writes none.
- **Cost:** each token costs one 1 MB logits row copy when asked.
- **Not yet:** `/v1/completions` and the Anthropic endpoint.

**The Monitor tab on Intel.** `sycl/serve/server_intel.py` is `serve/server.py` with two additions made at run
time:

- **GPU readings.** When NVML has no card, it plugs `sycl/serve/xe_telemetry.py` into `serve/telemetry.py`'s
  `gpu_reader`.
  - From sysfs: temperature, power and its cap, and the PCIe link the card trained at.
  - Load and VRAM come from `/run/gpustat.json`. xe reports them per client in `/proc/*/fdinfo`, which only root
    can read, so a small root sampler writes them (`sycl/tools/gpustat.py`; install it with its `gpustat.service`
    as its header says).
- **A Model menu.** A run config may name a `model_switcher`, an RPC taking `{"mode": m}`, for a host that swaps
  models on one card. The web app's header then gets a Model menu (`sycl/serve/web/switcher.js`, injected into the
  page).

**The engine exits directly** once its requests are done. It used to abort at exit in serve mode: a queue wait in a
destructor ran after the runtime's teardown began.

### Status of the planned work

1. **Expert dot products on XMX in integer mode** for decode: parked (see XMX above). Today it is scalar dp4a,
   ALU-bound.
2. **Experts missing from VRAM read from pinned host memory over PCIe instead of the SSD:** done (the host mirror).
3. **KV streaming from 64K up:** done.
4. **QSA block selection on XMX:** every query against every pooled block, a dense product that grows with the
   context. Open.
5. **The hot decode kernels re-tuned for Xe2's native 16-wide sub-groups:** tried, no gain in the engine
   (`STRATA_MMVQ_SG`: 16 all, 1 IQ4_XS only, 2 short outputs only; default 32). The real headroom was misaligned
   loads, since fixed:
   - A Q6_K block is 210 bytes, so every block's `ql`/`qh` runs start only 2-byte aligned. The B70 splits a
     misaligned 16-byte load into pieces.
   - The wide Q6_K kernel now does two aligned 16-byte loads and a shift (`load16_a2`; both stay inside the block),
     and takes every Q6_K shape and window width. `STRATA_MMVQ_A2=0` restores the old path.
   - The same treatment followed for Q4_K, Q5_K, IQ4_XS, Q8_0 and IQ4_NL ("Decode round 2" in
     INTEL_PERFORMANCE.md). The aligned-load helper only loads its second chunk when the address is unaligned, so it
     never reads a 16-byte chunk without a needed byte and cannot cross a page at the end of an allocation.
   - Switches back to the old paths:

     | switch | restores |
     |---|---|
     | `STRATA_MMVQ_WIDE_32=0` | the old Q8_0/IQ4_NL kernels |
     | `STRATA_GR_DOWN_SLICED=0` | the direct GR down kernel |
     | `STRATA_PLAN_PARALLEL=0` | the serial resident plan |
     | `STRATA_GR_DOWN_DIRECT=0` | GR down staging its activations in local memory |
     | `STRATA_MMVQ_WIDE_K=0` | the old K-quant kernels |

   - Some of these change the summation order. A long greedy continuation can then flip at a near-tie.
6. **INT8 prompt GEMMs:** experts dequantized to INT8, run as oneMKL/oneDNN INT8 on XMX. Open. A fused int8 kernel
   was tried (`xmx_int8_bench`) and lost.
7. **Fewer graph nodes per decode round** (~2,500): norm+rope, scores+top-k, gate+quantize fused. Open.
8. **Wider speculation** (two draft branches per verify window): the kernels are latency-bound, so it is nearly
   free. Open.
9. **A model bigger than VRAM:** done.
   - The original Qwen3.8-Flash-Next IQ2_XS runs on the B70 with 23 GB of RAM.
     - Shard 2 is byte-identical to the Coder's, so it is a hard link.
     - A native pack (`tools/iq_pack.py`).
     - The original model's expert profile and the same MTP draft layer.
   - Swift 1.5 runs the same way.
   - IQ3_XXS/IQ3_S (43-50 GB of experts) would need 19-26 GB mirrored, more than 23 GB of RAM allows on one card.
     Two cards hold them (see INTEL_PERFORMANCE.md, "Other people's cards").

## Keeping up with upstream

A merge of upstream `main` into `b70` leaves the copies in `sycl/` behind wherever upstream touched a file they
mirror. They are refreshed by re-migration, not by hand:

1. **Merge upstream into `b70`.** No shared file should conflict: the Intel code is all in `sycl/`. Then check that
   `sycl/setup_intel.py --check` and `sycl/serve/server_intel.py --engine mock` still run against the new setup.py
   and server.py.
2. **Migrate both trees.** The *old* upstream tree (a `git archive` of the pre-merge commit) goes through
   `migrate.sh` into `sycl-base`, and the merged tree into `sycl-new`. Each takes about 3 minutes in the dev image.
3. **Normalize both:** `sycl/tools/normalize.sh <dir> <commit>`. It covers dpct's file names, the unchanged files
   and the message serials, then runs `fixups.py`. Two runs of dpct on the same source now differ only in the
   kernel name hashes it generates.
4. **Merge:** `sycl/tools/merge_upstream.py BASE_OUT NEW_OUT OLD_REV NEW_REV` 3-way merges only the files upstream
   changed.
   - It canonicalizes dpct's kernel-name hashes to the port's first, and resolves hash-only hunks.
   - Copies dpct never produced take upstream's diff by hand: `verify.cpp`, `mtp.cpp`, `ple_reader.cpp` and
     `native_expert_parity.cpp`, which include no CUDA header directly.
   - New parity tests are copied in and listed in `sycl/CMakeLists.txt`.
   - `git merge-file --diff-algorithm=histogram` aligns big restructures better.
   - To port open upstream PRs ahead of upstream: migrate main plus the PRs, and merge into only the files they
     touch. When upstream later merges them, that "main + PRs" output is the merge base.
5. **Fix what the fixups missed.** A fixup whose pattern upstream changed shows up as a compile error; extend the
   fixup, re-run it, rebuild.
6. **Audit symbols.** Count the port's feature identifiers before and after. It has caught a dropped mirror hook and
   doorbell waits that had lost their spin bound.
7. **Check outputs.** Compare greedy output tokens against the previous build: Coder 19 / 2,184-token prompts,
   IQ2_XS, and a 40K prompt.

What each merge needed:

- **0.1.25-0.1.27 (2026-09-30):** the first re-migration. The draft layer's prompt pass became upstream's batched
  one.
- **0.1.31 (2026-10-01, 132 commits):** 45 migrated files changed for real, 38 conflicts.
  - **The expert kernels.** Upstream sends the i-quant experts to new multi kernels (one warp per row, 8 rows a
    group). The port's grid is sized for its own kernels (8 lanes a row, 32 rows a group).
    - With upstream's kernels on the port's grid, only a quarter of each expert's rows were written, and the output
      was end-of-text tokens.
    - The port's kernels stay the default; they also serve upstream's new Q4_K/Q5_K/Q5_1/Q8_0 experts.
    - `STRATA_EXPERT_SPLIT=1` runs upstream's kernels with their grid; `STRATA_GR_V3=1` runs upstream's GR read.
      Both are slower here and stay opt-in.
  - **Lanes per row** of the port's expert kernels are tunable at run time (`STRATA_GU_LANES` /
    `STRATA_DOWN_LANES`, 4/8/16/32). The default (8 / 8) stays.
  - **Three dpct mistranslations from the first migration,** found by upstream's new tests. dpct writes
    `__fadd_rn(a, b)` as `a + b` without parentheses, so `__fadd_rn(sum, c ? x : y)` became `sum + c ? x : y`.
    The three places:
    - the native QSA indexer's per-token pooled key: blocks completed while generating got wrong pooled keys;
    - the native QSA scores, an opt-in path;
    - the s2 activation rounding: every value was 0, in the Q2_0 s2 pack, which is unused here.

    `fixups.py` parenthesises now.
- **0.1.32 (2026-10-01, 89 commits):** hash-only differences are canonicalized before the merge, which left 10 real
  conflicts.
  - **Async commit:** upstream's `set_commit_async` / `wait_commit` maps onto the port's deferred commit
    (`commit(n, err, false)` + `commit_finish`).
  - **New hyper-connection read variants:** upstream's are not used. The port keeps its sliced down / split norm
    read, renamed `gr_norm_split_port_kernel`.
  - **The read-variant self-test** segfaulted on the B70, so on SYCL it runs only with `STRATA_HC_CHECK=1`.
  - **Kernel names:** two collided after hash canonicalization (renamed).
  - **fused_gr's** per-block shared-memory query is a fixup now.
- **0.1.33 (2026-10-01, 17 commits):** two conflicts.
  - `--resident-cpu-experts` (upstream's new `resident_cpu_explicit`) beside the port's `--stream-experts`.
  - The prompt attention's compute-capability check: the port keeps its XMX dispatch.
- **0.1.35 + seven open PRs (2026-10-02):** ported ahead of upstream.
  - **#374 needed two port fixes:**
    - Its next-chunk read assumed every chunk is the full chunk length. The port's first chunk is 256 tokens, so the
      second chunk's rows were read from the wrong place.
    - Its host wait on the PLE upload's event deadlocked past ~4K tokens (the Level Zero v2 bug above).
  - **#413's gate** is an NVIDIA SM-count rule, and its parity test an NVIDIA bench (not built).
  - **The PRs:**

    | PR | what | in the port |
    |---|---|---|
    | #385 (sergqwer) | stager: a buffer's first job of a generation waits for the previous DMA from it | race fix |
    | #463 (constantindjonkam) | decode waits for an adaptive swap before reading the residency table | determinism fix |
    | #453 (architectds) | the drafter's batched K/V on the KV-streaming ring | on |
    | #374 (sergqwer) | the first chunk's PLE rows read beside layer 0 | on, with two port fixes |
    | #363 (BlueKingMuch) | the PCIe expert call: group stride, fused SwiGLU + q8_1 | re-done on the port's lane kernels |
    | #407 (sergqwer) | `--adapt-tuned` | opt-in |
    | #413 (BlueKingMuch) | DeltaNet recurrence per key head | opt-in (`STRATA_GDN_KEYHEAD=1`): slower here |
- **0.1.38 (2026-10-03, 83 commits):** upstream merged the seven PRs, so their header forks in `sycl/include` are
  gone.
  - **New in the port with it:**
    - the DeltaNet output norm without its dead FP32 store;
    - two prompt-path fixes: the fused layout's buffers, and the streamed walk's resident lookup;
    - `STRATA_GR_DOWN_MAX4=1`, opt-in and neutral here.
  - **Stubbed:**
    - `--peer-device`, a second GPU as an expert-cache tier. Upstream's code is CUDA calls; `open` refuses with a
      message.
    - The fused int8 prompt kernels (#136), part of the MMQ library the port does not build.
  - **Off on SYCL:** the sm_90 thread-block-cluster greedy sampler and QSA top-k. They have no SYCL counterpart,
    and the callers take the plain kernels.

## Bugs worth remembering

**Prompt-slot borrowing: the hang (fixed 2026-10-01).** From 0.1.31 on, a prompt that borrowed cache slots stopped
in its first full chunk, with the GPU at 100% and the host waiting on the compute queue.

- **Not a borrowing bug.** Borrowing is just the only way the Coder streams experts during a prompt (every expert is
  resident otherwise), and the streamed path's stager deadlocked.
- **How the stager deadlocked:**
  - Its threads read experts into a ring of 16 page-locked buffers.
  - To reuse a buffer, a thread waited on the copy queue's event for the DMA that last read it. That is the CUDA
    form, `cudaEventSynchronize`.
  - Under the Level Zero v2 adapter, once a host thread had waited on that event, it no longer released the other
    queue's barrier that also listed it. The GPU waited forever.
  - It only happened once a layer streamed more than 16 experts, when the ring wrapped.
- **What ran fine:** the v1 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`), a 256-buffer ring, and a sync after every phase
  (`STRATA_PREFILL_SYNC=1`, kept as a debug switch).
- **The fix:** the copy queue writes a sequence number into page-locked host memory after each DMA, and the stager
  polls it. Output is bit-identical to the run without borrowing.

**Two model-specific bugs found with Swift 1.5 (fixed 2026-10-01).** UkisAI's Swift 1.5 IQ2_XS (setup's `swift`
family, the same GSQ-RCO layout) decoded token 0 forever: NaN logits from layer 13 on.

- **`--stream-experts` read the wrong shard.**
  - Since upstream 0.1.31, `ExpertLayout::gguf_file` is per layer AND role (`3 * layer + role`), because a shard
    boundary can fall inside a layer.
  - The port's `GgufExpertSource` still indexed it by layer. So a layer in shard 2 was read from shard 1 at
    shard 2's offsets: garbage IQ1_M scales, then infinities after the fp16 dequantization.
  - The original model keeps every expert in shard 1 and never noticed; Swift's layers 13-47 are in shard 2.
- **IQ2_XS products were garbage.**
  - ggml's `vec_dot_iq2_xs` reads its four 16-bit codes through a `uint16_t*` to an `int2`. The SYCL device
    compiler does not honour that type punning.
  - The dequantizer, which reads the codes directly, was exact.
  - The codes are extracted by shifts now (the dot and the expert kernels' `Split<17>`).
  - `iq_multi_parity` checks against a CPU decode from ggml's tables too, and passes.

**The layer split's free memory (fixed 2026-10-03).** A layer split across two B70s crashed at the second card's
expert cache ("no room").

- Upstream reads free memory in an `if` with an initializer, and dpct dropped the call itself. So every later stage
  saw 0 bytes free.
- The call is restored, and `tools/fixups.py` re-applies it after a re-migration.
- Found and tested on 2x B70 by tmking01 in the upstream PR review.

## Not done

- **Images,** on both Intel engines. Strata's vision path encodes with `strata-vision` into embeddings the CUDA
  engine reads; neither the SYCL port nor the llama.cpp config wires it yet.
- **Three kernels with inline PTX** (`mma.sync` tensor-core matrix ops, `ldmatrix`, `cp.async`): `qsa_prompt_attn`,
  `qsa_select`'s block scores and `native_qsa_score`. The SYCL build takes the "older card" fallback the CUDA build
  uses below sm_80.
- **The ggml MMQ prefill path** (`moe_mmq.cu`) needs llama.cpp's ggml-cuda sources; not built.
- **`--peer-device`:** stubbed.
- **Speculative decoding on the llama.cpp path:** the GGUF carries no draft layer llama.cpp can use. The SYCL port
  has it (`--mtp`).

## Arc A770 (DG2, 16 GB), 2026-10-03

The SYCL port builds and generates correct text on an Arc A770 (Xe-HPG, 16 GB, xe driver), not only the Arc Pro B70 it was
written for. Measured on a Xeon E5 v4 host (22 cores, 121 GB RAM, PCIe 3.0 x16), kernel 7.0 with the xe driver, the Coder
IQ1_M, a 19-token prompt, 128 greedy tokens, 8,192 context. The output matches the sample in "The engine end to end".

**How to build and run it.**

```
cmake -S sycl -B build-sycl-aot -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_AOT=dg2-g10 \
      -DSTRATA_SYCL_NO_XMX=ON -DSTRATA_SYCL_LARGE_BUFFERS=ON
IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1 AOT=dg2-g10 sycl/tools/build.sh strata   # full build ~25-40 min

UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 ZES_ENABLE_SYSMAN=1 \
build-sycl-aot/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> --expert-profile data/expert-profile-coder.bin \
    --stream-experts --expert-cache auto --vram-reserve-mib 1024 --prefill auto --spec 2 --suffix-draft 0 ...
```

This is the host-plan path: the CPU computes part of the experts that miss VRAM beside the GPU (8.0-8.7 tok/s). For a GPU-only
run add `STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1`: 9.75 tok/s, the faster of the two (see "Why the GPU-only path was
slow").

**What differs from the B70, and what each difference needed.**

| the A770 | what went wrong | the change |
|---|---|---|
| no fp64 (the engine accumulates some norms in `double` on purpose) | the build failed with "Double type is not supported" | build with IGC's fp64 emulation (`IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1`) |
| IGC (dg2) crashes on `joint_matrix` | `ocloc` died with a floating point exception on `xmx_gemm_iq` and `qsa_prompt_attn_xmx` | `-DSTRATA_SYCL_NO_XMX=ON` (both are opt-in and slower than the default paths) |
| the oneAPI Level Zero adapter does not answer free-memory queries on xe | `--expert-cache auto` saw 0 GiB free and made 0 slots | `dpct::get_memory_info` asks Level Zero Sysman directly when the SYCL aspect is missing |
| one allocation is capped at 4,095 MiB, host and device | the 9 GiB expert cache and the 14 GiB pinned mirror failed to allocate | `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` for the device; the mirror in chunks of at most 3 GiB |
| IGC addresses a buffer with 32-bit offsets unless told the buffer may exceed 4 GiB | a resident expert 5.96 GiB into the cache dequantised to NaN while the same expert in a small buffer was fine; the prompt path then poisoned the residual from layer 1 on and the model produced only token 0 | `-DSTRATA_SYCL_LARGE_BUFFERS=ON` adds `-ze-opt-greater-than-4GB-buffer-required` to the AOT options |
| a work-group barrier after some work-items have returned never completes | the native router (`route<NE>`) hung the first decode window, GPU busy, until the xe driver reset the compute engine | the barrier is removed (it protected nothing); `native_router_parity` is new |
| `DPCT_CHECK_ERROR(queue.ext_oneapi_empty())` returns 0, not the queue's answer | "layer N never rang (graph finished)" after 2 ms on any slow layer | the queue's answer is used |
| a kernel polling `sycl::malloc_host` memory never sees a host store made after it started | the GPU's bounded wait for each flag the host raises gave up, on every layer, although the host had raised all of them; the window went on with garbage | the flag words are allocated through Level Zero with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` (`sycl_host_mem.hpp`); a wait that still gives up now stops the engine and names the flag |

The A770 has one compute queue and one copy queue (the B70 has more), 512 EUs, 64 KiB of local memory per work-group.

**Speed.** Decode is 10.4 tok/s GPU-only and 8.0-8.7 on the host-plan path (llama.cpp's SYCL backend, same GGUF, same card:
9.4-9.8), prompt reading 11-23 tok/s on a 19-token prompt. 4,836 of the 12,288 experts fit in VRAM; the other 7,452 are mirrored in pinned host memory and read by the
GPU over PCIe.

| setting | decode |
|---|---|
| `--spec 4` (default drafts, 8% accepted) | 5.88 tok/s |
| `--spec 2` | 7.67 |
| `--spec 2 --suffix-draft 0` (no drafts) | 7.11 |
| `--vram-reserve-mib 512` (5,102 slots) | 7.33 (against 7.11 at 1,024) |
| `--expert-cache-per-layer` | 6.34 |
| host handshake path, `--adapt-every 0` (CPU pool takes misses) | 7.06 |
| (measured before the handshake fix below, the GPU's waits then gave up) host path with adaptive swaps, 128 tokens | 9.80, 9.77, 6.73 (three identical runs) |
| (before the fix) the same, 256 tokens / `STRATA_ADAPT_TUNED=1` / `--adapt-every 1` / `--spec 3` with drafts | 7.66 / 7.02 / 6.53 / 7.63 |
| GPU-only (`STRATA_VERIFY_NO_HOST=1`), two runs | 7.14, 7.14 |

The three "before the fix" rows were fast when they happened to work because the device plan then never had to wait on the host.
The measurements after the fix are in the second table below.

A round costs about 140 ms whether it carries one token or four, so the decode is not bound by compute. In eager mode with
`STRATA_VERIFY_PROFILE=1` the 36 GDN layers spend 47 ms in the expert stage (VRAM hits and mirror reads), 17 ms in the first
hyper-connection read and 14 ms in the second read plus the router, against 0.27 ms per layer all-in on the B70. The card
reads host memory at 9.4 GB/s from a kernel and 9.4 GB/s by DMA on this host (a 2 GiB copy), so PCIe 3.0 is a ceiling but
not the whole cost.

**The host-to-GPU handshake (fixed).** The engine's design is that the host raises flags (the plan, the copies, the CPU's
results) that a kernel on the GPU polls. On the A770 a polling kernel never saw a host store to `sycl::malloc_host` memory: 60
million system-scope atomic loads (and every other load variant) saw nothing, and a wait gave up after its bound however long
the bound was (checked at 100x: 2 million polls, about two seconds). Every layer then paid the whole bound and went on with
the data the host was supposed to write still missing: garbage, which the recurrent GDN state keeps for good. Memory from
`zeMemAllocHost` with the uncached or write-combined flag was seen within microseconds in every trial; the same call with no
flags was not reliable. The flag words are now allocated that way. Everything else keeps the adapter's allocation: making the plan
and result rows uncached cost 42 ms per window, and with `STRATA_VERIFY_NO_HOST` (nothing waits on the host) even the three flag
words did, so that mode keeps the plain allocation. A wait that still gives up now stops the engine with the flag's name
(`STRATA_WAIT_TIMEOUT=warn` only reports; `STRATA_SPIN_MAX` sets the bound, default 20,000 polls, about 22 ms here; do not set it
to hundreds of millions: a spin that long can trip the xe driver's hang detection).

With the handshake working, the host-plan path is the fast one: 8.0-8.6 tok/s over 128 tokens, five runs, correct output (it
produced zeros from the first token before). The CPU computes about 4-5 experts per layer beside the GPU.

| setting (128 tokens, `--spec 2 --suffix-draft 0`) | decode |
|---|---|
| host plan, adaptive swaps (the default), resident experts mirrored too | 8.05, 8.64, 8.26, 7.82, 8.14, 8.03, 8.36 |
| the same with `STRATA_MIRROR_ALL=0` | 7.31, 7.45 |
| device plan + host handshake, `STRATA_MIRROR_ALL=0` | 7.28 |
| GPU-only (`STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1`) | 7.14 |
| host plan, `--adapt-every 0` (64 tokens) | 5.65, 5.97 (and one run that produced wrong tokens, see below) |

**Keeping the experts in use in VRAM.** The static profile (`data/expert-profile-coder.bin`) puts 4,836 experts in VRAM and with
swaps off the CPU computes about 10 of the 16 experts a two-token window routes per layer (a VRAM hit rate of about a third). The
adaptive swaps move the experts the prompt actually routes to into VRAM and bring that down to about 2 (a hit rate of about 87%).
Their settings were swept on the host-plan path, 256 tokens, `--spec 2 --suffix-draft 0`:

| setting | CPU experts per layer | swapped | decode |
|---|---|---|---|
| default (`--adapt-every 4 --adapt-swaps 96 --adapt-decay 0.7`, 4,836 slots) | 2.1 | 2,621 | 8.69 tok/s |
| `--adapt-swaps 192` | 2.6 | 4,340 | 8.18 |
| `--adapt-decay 0.92` | 3.3 | 4,474 | 8.13 |
| `STRATA_ADAPT_TUNED=1` (every 2, 192 swaps, decay 0.92) | 3.2 | 8,859 | 8.34 |
| `--vram-reserve-mib 512` (5,102 slots) | 3.8 | 5,630 | 7.86 |
| tuned + 5,102 slots | 2.1 | 5,198 | 8.27 |

Nothing beat the default, and the differences are within run-to-run noise (about 0.4 tok/s). More slots or more aggressive swapping
does not help because a round takes about 100 ms of GPU time whether the CPU computes 2 experts per layer or 4: the limit is now
the GPU's kernels, not where the experts live.

**Why the GPU-only path was slow, found with unitrace.** `unitrace -d` (build it from `intel/pti-gpu`, tools/unitrace, as
`sycl/tools/Dockerfile.unitrace` does) prints the device time per kernel; the decode rounds are the difference between a short
and a long run. The GPU was 88% busy: 141 ms of kernel time in each 161 ms round, of which 78 ms (55%) was the routed-expert
kernels (`native_gu_port`, `native_down_port`), 0.85 ms per layer-token for the 17 MB of expert weights it touches (from VRAM
that is 0.03 ms): they were reading the mirrored experts over PCIe at about 9 GB/s. With `STRATA_VERIFY_NO_HOST` nothing on the
host sees the routing, so the adaptive tier (which lifts the hit rate from a third to 87%) had nothing to rank. The plan kernel
now counts the routed experts on the GPU (`resident_plan_set_usage`) and the engine feeds the counts to the adaptive tier:

| GPU-only, per decode round | before | after |
|---|---|---|
| kernel time | 141 ms | 100 ms |
| expert kernels (gu_port + down_port) | 78 ms | 31 ms |
| `native_mmvq_q6k_wide` (129 calls) | 17.7 ms | 17.7 ms |
| `fused_gr_read_multi` (hyper-connection reads) | 24 ms | 24 ms |
| `gdn_step_norm_multi` | 4.2 ms | 6.7 ms |
| decode, 128 tokens | 7.14 tok/s | 9.76, 9.74 tok/s (9.77 at 256 tokens) |

What is left is spread out: the Q6_K projections run at about a quarter of the card's memory bandwidth (137 us per call for
about 21 MB), `gr_up_multi` takes 159 us for a small projection, and the GPU idles for about 16% of the round between kernels.

**Kernel work on the GPU-only path: what was tried (A770, 2026-10-04).** After the expert kernels, the biggest item in the 100 ms
round is `native_mmvq_q6k_wide` (the Q6_K projections and the LM head: 129 tensors, about 1.9 GB per round, 17.7 ms, about 106
GB/s against a measured 426 GB/s streaming ceiling with 16-byte loads). Measured with `mmvq_bench` on the model's real shapes
(2560x10240, 2560x6144, 6144x2560, 2560x12288, and the 2560x248320 head; 2 columns, GB/s of weights):

| variant | 10240 | 6144 | 6144x2560 | 12288 | head |
|---|---|---|---|---|---|
| default (aligned-load `a2` kernel), no large-buffer option | 164 | 134 | 154 | 171 | 243 |
| the same linked with `-ze-opt-greater-than-4GB-buffer-required` (`STRATA_SYCL_LARGE_BUFFERS=ON`) | 142 | 113 | 130 | 149 | 220 |
| misaligned loads (`STRATA_MMVQ_A2=0`) | 117 | 103 | 61 | 123 | 146 |
| weight loads issued ahead, `STRATA_MMVQ_PF=2 / 3 / 4` | 129 / 105 / 64 | 113 / 95 / 59 | 127 / 110 / 70 | 136 / 110 / 65 | 167 / 132 / 76 |
| 2 / 3 / 4 rows per sub-group, aligned loads | 51 / 41 / 33 | 60 / 49 / 41 | 99 / 77 / 54 | 53 / 39 / 30 | 53 / 43 / 34 |
| the shared kernel (`STRATA_MMVQ_WIDE=0`) | 62 | 60 | 61 | 63 | 65 |
| blocks repacked at a 224-byte stride (`q6k_align_bench`) | 186 vs 162 | | 142 vs 102 | 162 vs 101 | 267 vs 135 |

- Nothing existing beats the default. **The repacked 224-byte stride is now done** (type tag 114, below).
- `strata::dp4a` (byte unpack and multiply-add, which the source says IGC turns into the DP4A instruction) runs at 730 G dp4a/s
  in a microbenchmark against 2,654 G/s for `__builtin_IB_dp4a_ss`; but the microbenchmark is not a clean isolation (see below),
  and swapping the builtin into the kernel changed nothing (identical GB/s to the digit in the same build). The Q6_K kernel is
  not limited by dot-product throughput: going from 1 to 2 columns raises its time 1.6x for the same weight bytes, but a
  4x faster dp4a does not help, so it is bound by load latency, issue or occupancy. The builtin differs from the emulation only
  where the int32 accumulator saturates (12 of 4 million random inputs, all near +-2^31).
- The 14% above is the AOT option that `STRATA_SYCL_LARGE_BUFFERS=ON` adds (stateless addressing for every kernel). Over a whole
  decode it costs only 1.6% (6.84 vs 6.73 tok/s at a cache under 4 GiB, identical output), because the Q6_K kernel is about a
  sixth of the round. Splitting the expert cache into arenas under 4 GiB to drop the option is not worth it.
- An outside review of the dp4a numbers: the microbenchmark (`dp4a.cpp`) lets the emulated form share byte extraction because
  its operands are correlated and only the sum escapes, and an opaque builtin call blocks that, so 3.6x is not a trustworthy rate;
  the instruction-level cause of the kernel's ceiling has not been found (the missing measurement is the ISA: count dp4a, sends
  and spills from `ocloc disasm` for both forms). The Q6_K dequant (`dpct::vectorized_binary<char4>(.., sub_sat)`) has an exact
  cheaper form (values are 0..63 minus 32, so `((x | 0x80808080) - 0x20202020) ^ 0x80808080`); not tried.
- The matrix engine (XMX, up to 256 int8 ops per clock per unit, against DP4A on the vector ALUs): not useful for decode here,
  since nothing in decode is arithmetic-bound, and the fork's own notes record three int8 `joint_matrix` expert variants being
  1.4-3x slower than dp4a (one hung the GPU). It could matter for the prompt path (dequant to int8 plus an int8 GEMM, if int8 is
  at least 1.5x faster than the fp16 GEMM on the real shapes) but IGC crashes on the fp16 `joint_matrix` kernels for dg2 and
  nobody has probed int8. The Intel blog post that prompted this (community.intel.com, "Engineering Arc 3/30/2022") could not be
  retrieved (Cloudflare challenge), so its claims are unverified here.

**Q6_K served from a repacked layout (done).** Every Q6_K block is 210 bytes, so its 16-byte loads start at 2-byte-aligned
addresses and the kernel needs a two-load-and-shift workaround (`load16_a2`). The loaders
(`native_dense.cpp`, `native_head.cpp`) now pad each block to 224 bytes on the host before the upload, under a new
type tag 114 (`native_mmvq_serving_type`, `native_mmvq_pack`; `STRATA_Q6K_STRIDE224=0` keeps the GGUF layout). The decode kernel
(`native_q6_k_mmvq_s224`, the wide kernel with aligned weight loads and scalar activation loads, at most 4 columns a launch:
5 and 6 columns in one launch spill registers, 0.14x) and the prefill dequantiser (`dequant_bf16.dp.cpp`) know the tag; any
other consumer fails loudly on an unknown type instead of misreading. Results are bit-identical to the 210-byte kernel.
`mmvq_bench` against the old kernel in the same build, 2 columns: 1.20x (2560x10240), 1.26x (2560x6144), 1.32x (6144x2560),
1.20x (2560x12288), 1.12x (the 248,320-row head), about 1.3x at 4 columns, 0.93x at 1. Whole GPU-only decode, 128 tokens, same
tokens: **10.37 and 10.36 tok/s against 9.77 and 9.76** (+6.2%); load time unchanged (25-26 s of wall time either way; the
repack of about 1.9 GB is a fraction of a second); the weights grow from 2,019 to 2,105 MiB.

**The hyper-connection read (`fused_gr_read_multi`), looked at and left alone.** It is about 24 ms of the 100 ms round (norm,
down, reduce and up kernels back to back; `gr_bench` on the A770: 153-220 us for 1-6 tokens, the B70's notes say 76-108 us at 6).
`unitrace` on `gr_bench` puts the up kernel at the top, but its token-mixed average overstates it. An ablation of the up
kernel at 2 tokens (total 166 us): skipping its weight loads saves 35 us (that is its memory work, about 190 GB/s), skipping the
prologue, the epilogue loads or the stores about 4-5 us each, skipping all three loads leaves 95 us, which is the other three
kernels and the gaps between the launches. The tile width (`UPM_COLS`) gives 166 / 304 / 237 / 152 us at 16 / 8 / 4 / 2 (bit
identical); `STRATA_GR_V3=1` has no effect on the port's path, `STRATA_GR_DOWN_SLICED=0` is 1.5x slower, `STRATA_GR_DOWN_DIRECT=0`
is the same. The sliced down kernel spills about 38 registers (the link log says so for SIMD32 at 128 registers) and
`gdn_step_norm_multi` also warns. A real gain would come from merging the norm and down stages and shortening the chain of four
launches, worth about 6 ms of the round; not done.

**Adaptive swaps (fixed).** The default `--adapt-every 4` used to zero the output after the first swap. The swap itself was
sound: the device residency table matched the host's and the swapped-in slots were byte-identical to the GGUF. An evicted expert
had never been mirrored in pinned host memory, so the GPU could not read it and the device plan waited for the host, whose flags
the GPU could not see (above). Now the waits work, and mirroring the resident experts too (after the real misses, within the RAM
cap; `STRATA_MIRROR_ALL=0` turns it off) is a speed matter: the swap costs 0.3 ms per round with it and 6.6 without, and the
decode is 8.0-8.6 against 7.3-7.5. It costs about 9 GiB more pinned memory on the 16 GB card. Separately,
`GgufExpertSource::blob()` published a ring slot before its read had finished; that race is fixed too but was not the cause.

**Open.**
- Host plan with `--adapt-every 0` produced wrong tokens once (from the second token, the CPU computing 10 experts per layer) in
  three runs; two repeats were correct. Not explained.
- The three flag words in uncached memory cost 42 ms per window on the GPU-only path, which never reads them. Not explained;
  that mode keeps the plain allocation.
- Per-kernel timestamps read zeros (`%globaltimer`), so the profile above is host-timed and per stage.
- The prompt path needs `iq_dequant_f16` and the oneMKL GEMMs; `conversation_snapshot_test` segfaults (not investigated).
- `kv_hybrid_parity` and `qsa_prompt_attn_parity` fail with `STRATA_SYCL_NO_XMX=ON` because they require the XMX kernel.
