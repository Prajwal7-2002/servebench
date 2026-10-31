# Metrics Glossary & Mathematical Formulations

This document provides rigorous mathematical definitions, explanations, and industry significance for all metrics measured by Servebench.

---

## 1. Core Latency Metrics

### Time to First Token (TTFT)
The duration from the moment the client dispatches the HTTP request to the moment the first non-empty content chunk is parsed.

$$\text{TTFT} = t_{\text{first\_content\_chunk}} - t_{\text{send}}$$

- **Units**: seconds / milliseconds.
- **Components**: Client network transit + Server queue wait + Engine prompt prefill compute.
- **Significance**: Determines the perceived responsiveness of interactive chat applications.

### Time Per Output Token (TPOT)
The average duration required to generate each token after the first token arrives.

$$\text{TPOT} = \frac{\text{E2E Latency} - \text{TTFT}}{\text{Output Tokens} - 1}$$

- **Units**: milliseconds / seconds per token.
- **Significance**: Determines reading speed. Human reading speed is approximately 20–30 ms/token (30–50 tokens/s).

### Inter-Token Latency (ITL)
The sequence of elapsed durations between consecutive content chunks arriving over the streaming Server-Sent Events (SSE) connection:

$$\text{ITL}_i = t_{i} - t_{i-1}$$

- **Significance**: While TPOT measures overall generation speed, ITL percentiles (p50, p90, p99) measure smooth token pacing and detect generation stuttering.

---

## 2. Server-Side Timing Breakdown (P11)

When using engines like llama.cpp or Ollama that report internal phase durations, Servebench breaks TTFT into:
1. **Server Prefill**: Time the GPU spent evaluating input prompt tokens.
2. **Server Decode**: Time the GPU spent generating output tokens.
3. **Queue & Network Overhead**:
   $$\text{Queue/Network Overhead} = \text{Client TTFT} - \text{Server Prefill}$$

- **Significance**: If client TTFT is 200 ms and Server Prefill is only 25 ms, requests are spending 175 ms waiting in line for an available KV cache slot.

---

## 3. Throughput & Goodput

### Output Token Throughput
$$\text{Output Throughput} = \frac{\sum_{i=1}^N \text{Output Tokens}_i}{\text{Wall Duration}}$$

- **Units**: tokens per second (`tok/s`).

### Steady-State Throughput
Throughput calculated over a stable time window that excludes the initial connection warm-up ramp-up and late draining tail.

### Goodput
The rate of completed tokens delivered by requests that satisfied **both** target latency Service Level Objectives (SLOs):

$$\text{Goodput} = \frac{\sum_{i \in \text{In-SLO}} \text{Output Tokens}_i}{\text{Wall Duration}}$$

where request $i$ is in-SLO if:
$$\text{TTFT}_i \le \text{SLO}_{\text{TTFT}} \quad \text{and} \quad \text{TPOT}_i \le \text{SLO}_{\text{TPOT}}$$

---

## 4. Hardware Telemetry & Energy Efficiency

### Tokens per Joule (Efficiency)
$$\text{Tokens per Joule} = \frac{\text{Output Throughput (tok/s)}}{\text{Average GPU Power (W)}} = \frac{\text{Total Output Tokens}}{\text{Total Energy Consumed (Joules)}}$$

- **Units**: tokens per Joule.
- **Significance**: Crucial for comparing models across quantization levels (e.g. FP16 vs INT4) and assessing datacenter operational costs.

---

## 5. Statistical Aggregations

### Order Statistics & Percentiles
For any timing distribution, percentiles (p50, p90, p99) are computed via nearest-rank interpolation.

> **Statistical Warning:** When sample size $N < 100$, computing p99 is mathematically bounded by the sample maximum. Servebench reports sample count $N$ alongside all percentiles and issues a warning if $N < 100$.

### Multi-Trial Repeat Confidence Intervals
When `--repeat N` is specified, each load level is executed across $N$ independent runs with fresh seeds. Servebench computes sample mean $\bar{x}$, sample standard deviation $s$, and two-tailed Student's t 95% confidence intervals:

$$\text{CI}_{95\%} = \bar{x} \pm t_{0.025, N-1} \times \frac{s}{\sqrt{N}}$$

---

## 6. Hardware Roofline & Utilization Math

### Operational / Arithmetic Intensity ($I$)
The ratio of floating point operations performed per byte of memory transferred from DRAM/HBM:

$$I = \frac{\text{Floating Point Operations (FLOP)}}{\text{DRAM Memory Transferred (Bytes)}} \quad [\text{FLOP/Byte}]$$

- **Decode Phase**: Each autoregressively generated token accesses all model weights once ($2P$ FLOPs per parameter, reading $P \times \text{Bytes/Param}$). At batch size $c=1$, intensity is $\approx 1.0\text{ FLOP/Byte}$ (FP16) or $\approx 4.0\text{ FLOP/Byte}$ (INT4).
- **Batching Amortization**: As batch size $c$ increases, weights are reused across $c$ tokens, scaling arithmetic intensity $I \approx c \times I_1$.

### Ridge Point ($I_{\text{ridge}}$)
The critical arithmetic intensity where hardware performance transitions from being memory-bandwidth-bound to compute-bound:

$$I_{\text{ridge}} = \frac{\text{Peak Compute Ceiling (FLOP/s)}}{\text{Peak Memory Bandwidth (Bytes/s)}} \quad [\text{FLOP/Byte}]$$

- If operational intensity $I < I_{\text{ridge}}$, the workload is **Memory-Bandwidth Bound**.
- If operational intensity $I \ge I_{\text{ridge}}$, the workload is **Compute Bound**.

### Memory Bandwidth Utilization (MBU %)
The fraction of theoretical memory bus throughput achieved by weight streaming during decode:

$$\text{MBU} = \frac{\text{Achieved Memory Bandwidth}}{\text{Peak Hardware Memory Bandwidth}} \times 100\%$$

where:
$$\text{Achieved Memory Bandwidth (GB/s)} = \frac{\text{Decode Tokens/s}}{c} \times \text{Model Size (GB)}$$

### Model FLOPs Utilization (MFU %)
The fraction of theoretical peak matrix tensor compute utilized by GEMM operations:

$$\text{MFU} = \frac{\text{Achieved TFLOPs}}{\text{Peak Hardware TFLOPs}} \times 100\%$$

where achieved decode TFLOPs is:
$$\text{Achieved TFLOPs} = \frac{\text{Tokens/s} \times 2 \times \text{Param Count}}{10^{12}}$$

