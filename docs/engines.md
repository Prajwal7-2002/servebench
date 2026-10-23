# Serving Engine Configuration & Tuning Guide

Servebench automatically identifies and benchmarks all major LLM serving engines through standard OpenAI-compatible `/v1` endpoints. This guide details optimal configurations, internal timing support, and Prometheus metrics for each engine.

---

## 1. Engine Compatibility Matrix

| Engine | Auto-Detect Port | Fingerprint API | Server Timing Breakdown (P11) | Prometheus /metrics | Recommended Flags for Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ollama** | `11434` | `/api/version` | ✓ (`prompt_eval_duration`) | - | `OLLAMA_NUM_PARALLEL=4` |
| **llama.cpp** | `8080` | `/props` | ✓ (`timings.prompt_ms`) | ✓ (`/metrics`) | `--parallel 4 -fa --cont-batching` |
| **vLLM** | `8000` | `/version` | Via Prometheus | ✓ (`/metrics`) | `--max-num-seqs 64 --enable-prefix-caching` |
| **mini-infer** | `8000` | `/health` (`engine: mini-infer`) | ✓ | - | `mini-infer serve --port 8000` |
| **SGLang** | `30000` | `/get_model_info` | Via Prometheus | ✓ (`/metrics`) | `--mem-fraction-static 0.85` |
| **LM Studio** | `1234` | `/api/v0/models` | - | - | Enable local server in Developer tab |

---

## 2. Engine Tuning & Quirks

### Ollama
- **Default Concurrency Bottleneck**: By default, Ollama configures single-stream execution. To test multi-user batching, set:
  ```bash
  export OLLAMA_NUM_PARALLEL=4
  export OLLAMA_FLASH_ATTENTION=1
  ollama serve
  ```
- **Localhost IPv6 Quirk on Windows**: On Windows, connecting to `http://localhost:11434` may attempt IPv6 `::1` before falling back to IPv4 `127.0.0.1`, injecting ~300 ms of connection delay into TTFT. Always use `http://127.0.0.1:11434/v1`.

### llama.cpp Server (`llama-server`)
- **Enabling Server Metrics**: Run `llama-server` with `--metrics` to expose Prometheus metrics at `/metrics`:
  ```bash
  llama-server -m model.gguf -c 4096 --parallel 4 -fa --cont-batching --metrics
  ```
- **Streaming Timings**: llama.cpp natively returns `timings: {prompt_ms, predicted_ms}` in the final stream chunk. Servebench extracts this automatically to isolate server prefill vs decode durations.

### vLLM
- **Prometheus Scraper**: vLLM exposes `/metrics` by default on its serving port. Use `--scrape-metrics auto` in Servebench to track:
  - `vllm:num_requests_waiting`: Pending FIFO queues.
  - `vllm:num_requests_running`: Active batched slots.
  - `vllm:gpu_cache_usage_factor`: KV cache VRAM pressure.
  - `vllm:prefix_cache_hit_rate`: Prefix caching effectiveness.

### SGLang
- **RadixAttention & Prefix Reuse**: Benchmark SGLang's RadixAttention using Servebench's `--shared-prefix-tokens 512` flag to observe TTFT reduction from KV cache reuse.

### mini-infer
- **Zero-Dependency Continuous Serving**: Pure Python/PyTorch inference engine featuring Paged KV-Cache, chunked prefill, and speculative decoding.
- **Auto-Detection**: Servebench recognizes `mini-infer` via its `/health` status and parses `/v1/chat/completions` usage chunks.
- **Launch Command**:
  ```bash
  mini-infer serve --port 8000
  ```
