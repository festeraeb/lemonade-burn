# Nauti Ternary

Ternary LLM inference in Rust: Microsoft's **BitNet b1.58 2B4T** (weights in {-1, 0, +1}) on a
packed 2-bit kernel, served as an OpenAI-compatible server that plugs into
[Lemonade](https://github.com/lemonade-sdk/lemonade). It runs on an ordinary x86 CPU in about
2 GB of RAM, no GPU needed.

Nauti Ternary is the product name; the code, crates and Lemonade recipe are still called
`lemonade-burn` (Burn is the Rust ML framework it is built on). Dual-licensed MIT OR Apache-2.0.

## Runs under Lemonade

- **One click, today:** the [Lemonade Engines](https://github.com/festeraeb/lemonade-engines) page
  (v0.3.0) downloads the server and the model, starts it on 127.0.0.1 and registers it with
  Lemonade. Tested 2026-10-28 through the page with Playwright on Linux and Windows 11 with
  Lemonade 2026.43.3: installed, registered, and answered through Lemonade.
- **Native backend (work in progress):** a `lemonade-burn` recipe for our Lemonade fork is
  generated from `engines/lemonade-burn.json` in lemonade-engine-adapters. It builds; it has not
  been run live yet.

## Measured (nothing here is estimated; full tables in [docs/BENCHMARKS.md](docs/BENCHMARKS.md))

- **Correctness** vs Hugging Face transformers (4 prompts, 146 teacher-forced positions): top-1
  agrees with HF float64 on 142/146; HF's own float32 run agrees on 140/146. The tokenizer matches
  HF on all prompts.
- **Speed**, Xeon E3-1265L v3 (4 cores, AVX2), 64-token prompt: about 40-48 tok/s prefill and
  7.7 tok/s decode, ~1.9 GB steady RSS. The perf work on this branch (AVX-512 tiled prefill,
  per-socket pools) is measured in BENCHMARKS.md.

## Crates

| Crate | What |
|---|---|
| `ternary-kernels` | packed 2-bit ternary x int8 GEMV/GEMM: scalar reference, AVX2, AVX-512BW, AVX-512 VNNI |
| `lb-ternary` | packed weights, int8 activation quantization, CPU kernels, NUMA pools |
| `lb-model` | the BitNet b1.58 decoder on Burn (Flex CPU backend) |
| `lb-convert` | bf16 safetensors -> packed ternary safetensors |
| `lb-server` | OpenAI-compatible HTTP server (`/v1/models`, `/v1/chat/completions`, `/health`) |
| `lb-ternary-gpu` | ternary GEMM kernel in CubeCL (Vulkan/CUDA). Matches the CPU kernel exactly on a GTX 1070 and a Tesla P100; **not yet used for decoding** |

## Quick start

```bash
cargo build --release -p lb-convert -p lb-server
target/release/lb-convert --src <bitnet-b1.58-2B-4T bf16 dir> --dst bitnet-lbt
target/release/lemonade-burn-server --model bitnet-lbt --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bitnet","messages":[{"role":"user","content":"Hello"}]}'
```

Prebuilt binaries for Linux, Windows and macOS: [releases](https://github.com/festeraeb/lemonade-burn/releases).
