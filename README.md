# llama.cpp — multi-fork integration build

> This repository is a personal integration fork of the upstream
> [llama.cpp](https://github.com/ggml-org/llama.cpp). It is **not** an official
> llama.cpp release. On top of the upstream `master` it layers performance and
> feature optimizations taken from several community forks. Every integration
> step is recorded in the per-fork `MERGE_*.md` notes (and independently reviewed
> in the `AUDIT_*.md` notes) listed below.

The integrated work lands on the branch **`turbo-kvstream-merge`**, which is
published as `origin/master`. Build instructions, supported backends, license,
and the rest of the upstream project description are preserved verbatim further
down under [Upstream llama.cpp README](#upstream-llama-cpp-readme).

## Merged open-source forks

| Repository | Origin / focus | What was integrated | Notes |
|---|---|---|---|
| [`AtomicBot-ai/atomic-llama-cpp-turboquant`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) | TurboQuant KV cache + turbo weight quantization | TurboQuant KV cache types (`turbo2/3/4`) and `TQ3_1S`/`TQ4_1S` weight quantization (subset only) | [MERGE_TURBO_KVSTREAM.md](MERGE_TURBO_KVSTREAM.md) |
| [`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) | Adaptive KV cache streaming for the CUDA `llama-server` | Host-pinned KV with a bounded CUDA arena, phase multiplexing, and prefetch scheduling (the full KV-streaming branch) | [MERGE_TURBO_KVSTREAM.md](MERGE_TURBO_KVSTREAM.md) |
| [`anyei/llamacpp-v100`](https://github.com/anyei/llamacpp-v100) | Tesla V100 (Volta) CUDA tuning | sm70 `MMVQ_PARAMETERS_VOLTA` table + one-shot P2P NVLink AllReduce for 2-GPU tensor mode (4 files) | [MERGE_V100.md](MERGE_V100.md) · [AUDIT_V100_MERGE.md](AUDIT_V100_MERGE.md) |
| [`jackjusko/jusko-llama-volta-qwen3flash`](https://github.com/jackjusko/jusko-llama-volta-qwen3flash) | Volta / SM70 Qwen3.8-Flash optimizations | sm70 q8_0 tensor-core FA, 256×256 compact FA specialization, Volta GEMV variants (Q5_K×4, Q6_K W4R4), Q6_K MMQ Pascal-DP4A routing, GDN 128×4 Volta kernel, `--prefill-reuse`, and `--checkpoint-recurrent-prev` recurrent-state replay | [MERGE_JUSKO_VOLTA.md](MERGE_JUSKO_VOLTA.md) · [AUDIT_JUSKO_MERGE.md](AUDIT_JUSKO_MERGE.md) |
| [`jackinthebox52/qwen38-v100-serve`](https://github.com/jackinthebox52/qwen38-v100-serve) | Qwen3.8-27B V100 serving | `flash_attn_ext_vec` `ncols2 = 3` GQA head packing for GQA ratios with factor 3 (e.g. Qwen3.8-27B 24/4) on Volta (1 file) | [MERGE_QWEN38_V100.md](MERGE_QWEN38_V100.md) |

Each fork is a full distribution; this tree takes only the parts relevant to its
targets (CUDA + Volta / V100, and a backend-agnostic recurrent-checkpoint feature),
scoped file by file. See each merge note for exactly what was taken, what was
deliberately **excluded**, the integration decisions, and the verification results.

## Merge & audit documents in this repository

- **[`MERGE_TURBO_KVSTREAM.md`](MERGE_TURBO_KVSTREAM.md)** — TurboQuant KV cache + adaptive KV streaming (two forks merged together).
- **[`AUDIT_TURBO_PORT.md`](AUDIT_TURBO_PORT.md)** — independent audit of the TurboQuant (+ adaptive-KV-streaming) port (verdict: algorithm ported correctly, no numeric drift; issues were in test coverage, the template-instance list, and excluded-arch leftovers — all fixed).
- **[`MERGE_V100.md`](MERGE_V100.md)** — Tesla V100 (Volta) CUDA port from `llamacpp-v100`.
- **[`AUDIT_V100_MERGE.md`](AUDIT_V100_MERGE.md)** — independent audit of the `llamacpp-v100` port (verdict: complete & correct).
- **[`MERGE_JUSKO_VOLTA.md`](MERGE_JUSKO_VOLTA.md)** — Volta/SM70 pieces + recurrent checkpoint tail replay from `jusko-llama-volta-qwen3flash`.
- **[`AUDIT_JUSKO_MERGE.md`](AUDIT_JUSKO_MERGE.md)** — independent audit of the `jusko` port (verdict: faithful, complete, doc/code consistent).
- **[`MERGE_QWEN38_V100.md`](MERGE_QWEN38_V100.md)** — Qwen3.8-27B GQA packing from `qwen38-v100-serve`.

## Caveats (read before using)

- **Compile-verified only.** The maintainer's build machine has no NVIDIA GPU, so every CUDA/Volta feature is validated by compilation (incl. `sm_70` PTX checks) but **not** by end-to-end execution. The throughput/latency numbers in the merge notes are the upstream forks' own measurements, not re-measured here.
- **Hardware-specific.** The adaptive-KV-streaming branch targets an RTX 5070 Ti 16 GB; the V100/Volta ports target Tesla V100 (`sm_70`). Build with `GGML_CUDA=ON` and `CMAKE_CUDA_ARCHITECTURES` including `70` for the Volta features to be active.
- **Model-specific restrictions.** Adaptive KV streaming is restricted to Qwen3.5 by the upstream fork; TurboQuant KV cache types are CUDA-only (Metal/Vulkan reject them); several Volta switches are opt-in and some are GQA-shape-gated. See each merge note.
- This is an integration sandbox, not a supported distribution. Prefer the upstream llama.cpp for general use.

---

## Upstream llama.cpp README

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
