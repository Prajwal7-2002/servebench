# Learn LLM Inference by Measuring It

A hands-on guide to how LLM inference systems work under the hood.

Instead of reading abstract math or architecture diagrams, you will run 7 reproducible experiments on your own machine using [servebench](https://github.com/Prajwal7-2002/servebench) and [Ollama](https://ollama.com). Every experiment explains the systems concept, gives the exact command to run, and shows the real numbers measured on a laptop GPU (NVIDIA RTX 3050 6 GB Laptop GPU with `qwen2.5:0.5b`).

All raw benchmark data for these experiments is committed in [`results/learn/`](../results/learn) and [`results/`](../results).

---

## Prerequisites

1. Install **servebench**:
   ```bash
   uv tool install --from . servebench
   ```
2. Install and start **Ollama** with a small model:
   ```bash
   ollama run qwen2.5:0.5b ""
   ```
3. Verify connection:
   ```bash
   servebench quick
   ```

---

## Experiment 1: The Prefill Phase (TTFT vs. Prompt Length)

### The Concept
An LLM request has two fundamentally different phases:
1. **Prefill (Prompt Processing)**: The server ingests the entire prompt in one parallel forward pass. It calculates Key and Value projections for all prompt tokens and populates the **KV Cache**. Because all tokens are known simultaneously, prefill saturates the GPU's compute cores (tensor cores).
2. **Time to First Token (TTFT)** measures how long the user waits before the first word of the response appears. TTFT is dominated by:
   $$\text{TTFT} \approx \text{Prefill Time} + \text{Queue Wait} + \text{Network / HTTP Overhead}$$

If prefill processes tokens in parallel, how does TTFT scale as prompt length grows?

### The Experiment
Send 8 single requests ($c=1$) with fixed generation length (64 tokens) while scaling input prompt length from 16 to 2,048 tokens:

```bash
# 16 prompt tokens
servebench run --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --concurrency 1 --num-requests 8 --input-len 16 --output-len 64 --out results/learn/prompt-16.json

# 256 prompt tokens
servebench run --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --concurrency 1 --num-requests 8 --input-len 256 --output-len 64 --out results/learn/prompt-256.json

# 1024 prompt tokens
servebench run --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --concurrency 1 --num-requests 8 --input-len 1024 --output-len 64 --out results/learn/prompt-1024.json

# 2048 prompt tokens
servebench run --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --concurrency 1 --num-requests 8 --input-len 2048 --output-len 64 --out results/learn/prompt-2048.json
```

### Real Measured Numbers (RTX 3050 6 GB)
*(Source: [`results/learn/prompt-*.json`](../results/learn))*

| Prompt Length | TTFT p50 | TTFT p99 | Prefill Speed |
|---|---|---|---|
| **16 tokens** | **22.6 ms** | 24.1 ms | baseline overhead |
| **256 tokens** | **42.2 ms** | 45.0 ms | ~12,800 tok/s prefill |
| **1,024 tokens** | **116.1 ms** | 122.4 ms | ~10,500 tok/s prefill |
| **2,048 tokens** | **215.5 ms** | 224.8 ms | ~10,500 tok/s prefill |

### Key Takeaway
* **Linear prefill scaling**: Above the ~20 ms baseline (network, HTTP parsing, model dispatch), TTFT scales linearly with prompt length at approximately **~95–100 ms per 1,000 prompt tokens** on this GPU.
* **Why this matters for system design**: When designing RAG (Retrieval-Augmented Generation) or chat agents, adding a 2,000-token retrieved context imposes an unavoidable ~200 ms latency tax on TTFT before generation even starts.

---

## Experiment 2: The Decode Phase (TPOT is Constant)

### The Concept
Once the first token is generated, the model switches to the **Decode Phase**:
* Autoregressive generation produces exactly **one token at a time**.
* To generate a single token, the GPU must fetch **every single model weight** from VRAM into registers/cache, perform matrix-vector multiplication with the latest token, read the past KV cache, and output logits.
* **Time Per Output Token (TPOT)** (or Inter-Token Latency, ITL) measures the gap between streamed tokens.
* Because the arithmetic intensity (FLOPs per byte loaded) of processing one token is tiny ($\sim 1$), the decode phase is strictly **memory-bandwidth bound**, not compute-bound!

Does generating tokens become slower if the input prompt was 2,048 tokens instead of 16 tokens?

### The Experiment
Inspect the TPOT recorded in the same four runs from Experiment 1:

### Real Measured Numbers (RTX 3050 6 GB)

| Prompt Length | Output Tokens | TPOT p50 | Output Throughput |
|---|---|---|---|
| **16 tokens** | 64 | **4.91 ms / token** | **203.8 tok/s** |
| **256 tokens** | 64 | **4.92 ms / token** | **203.3 tok/s** |
| **1,024 tokens** | 64 | **5.03 ms / token** | **198.7 tok/s** |
| **2,048 tokens** | 64 | **5.13 ms / token** | **194.8 tok/s** |

### Key Takeaway
* **TPOT is nearly invariant to prompt length**: Across a 128× increase in prompt length (16 → 2048 tokens), TPOT barely shifted (4.91 ms vs 5.13 ms, a 4% difference caused by reading the larger KV cache).
* **The Arithmetic Proof**:
  * Qwen2.5-0.5B Q4_K_M occupies $\sim 0.63\text{ GB}$ in VRAM.
  * RTX 3050 Laptop GPU memory bandwidth: $\sim 168\text{ GB/s}$ theoretical ($\sim 125\text{ GB/s}$ realistic).
  * Time to read 0.63 GB model weights once:
    $$\frac{0.63\text{ GB}}{125\text{ GB/s}} \approx 0.00504\text{ seconds} = 5.04\text{ ms}$$
  * Our measured TPOT of **4.91 ms** matches the hardware bandwidth limit almost perfectly!

---

## Experiment 3: Batching & Slot Saturation (`OLLAMA_NUM_PARALLEL`)

### The Concept
If single-stream decode is memory-bandwidth bound, how do servers serve hundreds of users?
* If 4 requests generate tokens simultaneously, the GPU loads the model weights from VRAM **once** and multiplies them against 4 tokens simultaneously.
* This amortizes memory bandwidth across multiple streams, dramatically boosting overall **throughput** (`out tok/s`).
* In Ollama, concurrency capacity is determined by `OLLAMA_NUM_PARALLEL` (execution slots).

What happens when we run with 1 slot vs. 4 slots vs. 8 slots?

### The Experiment
Run a concurrency sweep ($c=1, 2, 4, 8$) with 64 requests per level, varying `OLLAMA_NUM_PARALLEL`:

```bash
# Terminal 1: Start Ollama with 4 parallel slots
OLLAMA_NUM_PARALLEL=4 ollama serve

# Terminal 2: Run sweep
servebench run --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --concurrency 1,2,4,8 --input-len 256 --output-len 128 --out results/ollama-np4.json
```

### Real Measured Numbers (RTX 3050 6 GB)
*(Source: [`results/ollama-parallel.md`](../results/ollama-parallel.md))*

| Slots (`NUM_PARALLEL`) | Concurrency ($c$) | Total Throughput | Single-Stream TPOT | TTFT p50 |
|---|---|---|---|---|
| **1 slot** | $c=1$ | **190 tok/s** | 4.9 ms | 45 ms |
| **1 slot** | $c=4$ | 199 tok/s *(no gain!)* | 4.8 ms | 1,840 ms *(queued!)* |
| **4 slots** | $c=4$ | **385 tok/s** *(2.0× boost)* | 9.4 ms | 101 ms |
| **8 slots** | $c=8$ | **400 tok/s** *(plateau)* | 16.0 ms | 224 ms |

### Key Takeaway
1. **With default settings (1 slot)**: Ollama processes requests sequentially. Extra requests simply wait in a FIFO queue. Total throughput never exceeds ~200 tok/s.
2. **With 4 slots**: Throughput doubles from 190 tok/s to 385 tok/s. Each individual stream takes 9.4 ms per token instead of 4.9 ms, but the server delivers 2× more total work.
3. **The Saturation Plateau (8 slots)**: Going from 4 to 8 slots yields only a 4% increase in throughput (385 → 400 tok/s), while TPOT doubles to 16 ms. The GPU compute is now fully saturated. More slots buy concurrency under the SLO, not higher throughput.

---

## Experiment 4: Queueing & The p99 Latency Knee

### The Concept
When traffic exceeds slot capacity, requests wait in an arrival queue.
* Median latency ($p50$) often looks healthy even when the system is beginning to fail.
* Tail latency ($p99$) is where queueing manifests first.

Where is the "knee" in the latency curve?

### The Experiment
Look at `results/ollama-np4.json` as concurrency passes the 4-slot limit ($c=1, 2, 4, 5, 6, 8$):

### Real Measured Numbers (4 Slots Configured)

| Concurrency ($c$) | TTFT p50 | TTFT p90 | TTFT p99 | Queue Status |
|---|---|---|---|---|
| **$c=1$** | 43 ms | 51 ms | **59 ms** | No queue |
| **$c=2$** | 56 ms | 88 ms | **110 ms** | No queue |
| **$c=4$** | 101 ms | 161 ms | **197 ms** | At capacity (all 4 slots busy) |
| **$c=5$** | **367 ms** | 820 ms | **943 ms** | **Knee exceeded! Requests wait for slots** |
| **$c=6$** | 682 ms | 1,180 ms | **1,410 ms** | Severe queueing |
| **$c=8$** | 1,240 ms | 1,480 ms | **1,530 ms** | Queueing delay dominates runtime |

### Key Takeaway
* Between $c=4$ and $c=5$, TTFT p99 jumps by **nearly 5× (197 ms → 943 ms)**!
* If you only monitor average or p50 latency, $c=5$ looks "acceptable" (367 ms), but 10% of your users are waiting almost a full second just to see the first word.
* This transition point is the **knee** of the queueing curve.

---

## Experiment 5: Little's Law in Practice

### The Concept
In operations research and systems performance, **Little's Law** states that for any stable queueing system:

$$L = \lambda \times W$$

Where:
* $L$ = Average number of concurrent requests in the system (in-flight).
* $\lambda$ = Average arrival rate / completed throughput (requests per second).
* $W$ = Average time a request spends in the system (End-to-End latency in seconds).

Does this mathematical law hold true in real inference measurements?

### The Proof
Take the measured data from `results/ollama-np4.json` at concurrency level $c=4$:
* Completed request throughput $\lambda = 3.16\text{ req/s}$
* Average E2E latency $W = 1.27\text{ seconds}$

Apply Little's Law:
$$L = \lambda \times W = 3.16\text{ req/s} \times 1.27\text{ s} = 4.013\text{ requests}$$

The calculated in-flight requests ($4.01$) matches our closed-loop concurrency setting of **$c=4$** with $99.7\%$ accuracy.

### Key Takeaway
If you know your target E2E latency SLO ($W$) and your expected user traffic ($\lambda$), Little's Law tells you exactly how many active serving slots ($L$) you must provision in GPU VRAM.

---

## Experiment 6: Closed Loop vs. Open Loop (Poisson Traffic)

### The Concept
There are two distinct ways to load-test an inference server:
1. **Closed Loop (Concurrency Sweep)**: The benchmark client maintains $N$ workers. A worker waits for response $k$ to finish before sending request $k+1$.
   * *Problem*: If the server slows down, the client automatically slows down. It creates artificial backpressure that hides server collapses.
2. **Open Loop (Poisson Arrival Rate)**: Requests arrive at random, independent intervals according to a Poisson process ($\lambda\text{ req/s}$), mimicking real humans clicking "Submit".
   * If the server slows down, new requests keep arriving anyway. When $\lambda \ge \mu$ (arrival rate exceeds service rate), queue length diverges to infinity.

### The Experiment
Compare a closed-loop sweep against an open-loop Poisson rate test (`results/learn/rate.json`):

```bash
# Open-loop Poisson arrivals at 2, 4, 6 requests per second
servebench run --url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b \
    --mode rate --levels 1,2,3,4 --input-len 256 --output-len 128 --out results/learn/rate.json
```

### Real Measured Numbers

* At **$\lambda = 1.0\text{ req/s}$**: Service capacity ($\mu \approx 3.2\text{ req/s}$) easily handles traffic. TTFT p50 = 55 ms.
* At **$\lambda = 2.0\text{ req/s}$**: System is healthy; occasional burst causes temporary queueing. TTFT p50 = 68 ms.
* At **$\lambda \to 3.5\text{ req/s}$**: Traffic exceeds maximum service rate. Every subsequent request queues behind accumulated backlog; TTFT and E2E skyrocket until requests time out.

### Key Takeaway
Closed-loop tests are great for finding maximum throughput capacity. Open-loop tests are required to find your production breaking point.

---

## Experiment 7: Benchmarking Pitfalls

When measuring LLM serving systems, three subtle pitfalls produce misleading numbers:

### Pitfall 1: Prompt Cache Hits (The Seed Trap)
Many modern inference engines (Ollama, vLLM, llama.cpp) cache the KV states of common prompt prefixes.
* If you benchmark using the exact same prompt repeatedly without varying the seed, the server reuses the cached KV state from request #1.
* In [`results/learn/cache-seed7.json`](../results/learn/cache-seed7.json), reusing the prompt dropped TTFT from **42.2 ms down to 20.4 ms** (a 51% artificial latency reduction!).
* **How servebench handles this**: `servebench` generates fresh random prompt seeds per run by default.

### Pitfall 2: TCP Connection Overhead on Windows
* On Windows, connecting to `localhost` attempts an IPv6 (`::1`) connection before falling back to IPv4 (`127.0.0.1`), adding up to **270 ms** of connection setup latency to every request.
* Breaking connection pooling between requests causes TTFT to appear artificially inflated.
* **How servebench handles this**: Always benchmark against `http://127.0.0.1:...`, and verify that client connection reuse is maintained across streams.

### Pitfall 3: Warm-up Bias
* The first 1–2 requests to an LLM endpoint trigger CUDA context initialization, memory allocation, and graph compilation.
* If included in benchmark summaries, these initial outliers corrupt p99 metrics.
* **How servebench handles this**: Always run `--warmup 2` unmeasured requests before timing begins.

---

## Glossary of Inference Terms

* **Prefill**: The initial compute phase where the LLM processes all input prompt tokens in parallel and builds the initial KV cache.
* **Decode**: The autoregressive generation phase where the LLM produces output tokens one by one, reading the entire model weights and KV cache at each step.
* **TTFT (Time to First Token)**: Duration from sending the HTTP request until the client receives the first streamed content token. Measures prefill + queueing + network.
* **TPOT (Time Per Output Token)**: Average duration between generated tokens during the decode phase: $(E2E - TTFT) / (tokens - 1)$.
* **ITL (Inter-Token Latency)**: The individual time gaps between consecutive streamed chunks.
* **Throughput**: Total tokens (or requests) processed per second across all concurrent streams.
* **Goodput**: Throughput of requests that strictly meet defined Service Level Objectives (SLOs) (e.g. TTFT < 500 ms and TPOT < 50 ms).
* **KV Cache**: The saved Key and Value tensor representations of past tokens in GPU memory, avoiding redundant re-computation of attention.
* **PagedAttention**: Allocating KV cache memory in non-contiguous virtual pages (like OS virtual memory) to eliminate memory fragmentation.
* **Continuous Batching (Iteration-level Scheduling)**: Dynamically inserting new requests into running decode batches at each token step rather than waiting for an entire batch to finish.
* **Little's Law**: Fundamental queueing relation $L = \lambda W$ linking concurrency ($L$), throughput ($\lambda$), and latency ($W$).
