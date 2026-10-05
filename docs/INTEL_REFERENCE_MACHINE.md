# The reference machine: an Arc A770 on the `xe` driver

Every A770 number in `INTEL.md` and the homelab benchmark doc was measured on one machine (`arc01`). This is its exact
software and hardware, as read from the machine on 2026-10-05, so a result can be repeated or a difference explained.

## Hardware

| | |
|---|---|
| GPU | Intel Arc A770 16 GB, Alchemist (DG2-G10), PCI id `8086:56a0`, AOT target `dg2-g10` |
| VRAM aperture | BAR2 is 16 GiB: Resizable BAR is on (BAR0 16 MiB) |
| PCIe | The card's own bridges report `2.5 GT/s x1` (they idle in a low link state; ignore them). The root port reports `8.0 GT/s x16`, so the card runs at **PCIe 3.0 x16**, about 15.75 GB/s raw, although it can train at 16 GT/s. The host is the limit, and expert streaming from host RAM is bound by this link. |
| CPU | Xeon E5-2696 v4 (Broadwell: AVX2, no AVX-512), 22 cores / 44 threads at 2.2 GHz |
| RAM | 121.5 GiB; transparent huge pages `madvise` |
| Model storage | one 936 GB LVM volume (also the OS disk) |

The CPU matters for builds: the container image is built with `-DSTRATA_GGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF`,
because a binary tuned on a CI runner with AVX-512 dies with "Illegal instruction" here.

## Operating system and kernel

| | |
|---|---|
| OS | Ubuntu 26.04.1 LTS ("resolute") |
| Kernel | `7.0.0-34-generic` (`linux-image-generic 7.0.0-34.34`) |
| Firmware | `linux-firmware 20260319.git217ca6e4` |
| Orchestration | RKE2 Kubernetes v1.34.6, containerd 2.2.2; Intel GPU device plugin 0.37.0 |

## The `xe` kernel driver (not `i915`)

Intel's newer `xe` driver does not claim the A770 by default (its PCI table lacks `56a0`), and `i915` takes it first.
Both are steered with kernel parameters:

```
# /etc/default/grub.d/xe.cfg
GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT xe.force_probe=56a0 i915.force_probe=!56a0"
```

```
sudo update-grub && sudo reboot
# check:
readlink /sys/class/drm/card0/device/driver        # .../xe
cat /sys/module/xe/parameters/force_probe          # 56a0
cat /proc/cmdline                                  # ... xe.force_probe=56a0 i915.force_probe=!56a0
```

Why: on the same llama.cpp SYCL build (b11223) decoding on `xe` was 40-48 tok/s against 27-32 on `i915` (a 9B Q6_K model with MTP);
prompt processing was about 3% faster and VRAM use identical. Strata has only been run on `xe` here, so there is no `i915` comparison for it.

Things that behave differently on `xe`:

- **Kubernetes resource name:** the Intel device plugin advertises `gpu.intel.com/xe`, and `gpu.intel.com/i915` drops to 0.
  A pod that asks for `i915` stays unschedulable.
- **Monitoring:** `intel_gpu_top` does not work (it needs the `i915` PMU). Use sysfs: `/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq`,
  `gtidle/idle_status`, hwmon energy and temperature, and per-process `drm-resident-vram0` in `/proc/<pid>/fdinfo` (root only).
- **Boot:** if the parameters are only on a one-shot GRUB entry, a power cut brings the machine back on `i915` and the GPU pods
  will not start. Make the parameters the default, as above.

## Userspace stack (inside the container image)

The engine runs in `ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable` (the base of `sycl/tools/Dockerfile.serve`).
The compute stack in the container is independent of the host's packages.

| | Container (used) | Host (not used by the containers) |
|---|---|---|
| Intel oneAPI DPC++/C++ compiler | 2026.1.0-235 | |
| oneMKL | 2026.1.0-236 | |
| Compute runtime (NEO): `intel-opencl-icd`, `libze-intel-gpu1`, `intel-ocloc` | 26.35.39758.10 | 26.05.37020.3 |
| Graphics compiler (IGC) | 2.41.5 | libigc1 1.0.17791.18 |
| Level Zero loader (`libze1`) | 1.32.0 | 1.28.2 |
| gmmlib (`libigdgmm12`) | 22.10.0 | 22.9.0 |

Runtime environment the engine is started with:

| Variable | Value | Why |
|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:0` | one card, Level Zero (not OpenCL) |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | `1` | allocations over 4 GiB (the expert cache is about 9 GiB) |
| `ZES_ENABLE_SYSMAN` | `1` | free-VRAM queries |
| `IGC_EnableDPEmulation`, `OverrideDefaultFP64Settings` | `1` | the AOT compile needs fp64 emulation, and so does running |
| `SYCL_CACHE_PERSISTENT` | `0` | its persistent cache segfaults in this runtime |

The container needs the card's device nodes (`/dev/dri`): under Kubernetes by requesting `gpu.intel.com/xe: 1`, under Docker with
`--device /dev/dri`. Do not share the card with another GPU process: oversubscribed VRAM is evicted silently, not refused.

## Memory the deployment needs

The IQ3_XXS model keeps every expert in a pinned host-RAM mirror (about 40 GiB) and a 9.2 GiB cache in VRAM; the pod limit is
100 GiB of RAM. The Coder IQ1_M needs about 23 GiB for its mirror.

## To repeat a result

1. Ubuntu with a kernel that has `xe` (7.0 here), the two kernel parameters above, `update-grub`, reboot, check the driver.
2. Enable Resizable BAR in the firmware (the 16 GiB BAR2 above); without it VRAM access is slow.
3. Run the engine in the image above (or one with the same oneAPI 2026.1, NEO 26.35 and IGC 2.41 versions) with the environment above.
4. Keep the host CPU, RAM and PCIe generation in mind when comparing: PCIe 3.0 x16 and an AVX2-only Xeon are slower than a current
   host, and most of the decode time here is the experts' trip over PCIe.
