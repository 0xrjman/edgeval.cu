# Optimization Journey

How edgeval.cu went from 5.7s/image to 0.47s/image — a 12× within-GPU speedup.

All benchmarks on RTX 4090 (sm_89), AMD Ryzen 9 7950X, BSDS500 test set (200 images, 99 thresholds × ~5 GT annotations).

## Timeline

| Stage | Time/img | 200 imgs | vs CPU | What changed |
|-------|----------|----------|--------|--------------|
| 0. CPU CSA baseline | ~6s | 20 min | 1× | MATLAB reference pipeline |
| 1. Extended graph (GPU) | 5.7s | — | — | kOfN + matchable filtering on GPU |
| 2. Simple bipartite graph | 1.17s | 3.9 min | 5× | Drop kOfN, skip matchable — faster graph |
| 3. Fused CUDA edge builder | 1.00s | 3.3 min | 6× | `edge_builder.cu`: single kernel replaces cdist+mask+nonzero |
| 4. GPU batched thinning | 1.00s | 3.3 min | 6× | Zhang-Suen via PyTorch conv2d, 99 masks in one batch |
| 5. Consecutive stall detection | 0.77s | 2.6 min | 7.7× | Stall after N consecutive no-change rounds (not 1) |
| 6. GPU annotator split | 0.60s | 2.0 min | 10× | Compound sort by (annotator, person, object) on GPU |
| 7. GPU nonzero | 0.58s | 1.9 min | 10.3× | torch.nonzero on GPU, eliminate mask download/reupload |
| 8. Tuned ITERS_EPS0 | **0.47s** | **1.6 min** | **12.8×** | ITERS_EPS0 5000→500, STALL_EPS_0 50→200 |

## Key Optimizations Explained

### 1. Simple vs Extended Graph

The original CPU CSA solver uses an "extended" n×n bipartite graph with kOfN random outlier edges. This replicates MATLAB exactly but the kOfN overhead (degree-6 random edges per pixel) inflates the edge count 5-6× and adds RNG complexity. We found that a simple bipartite graph (real edges only, no kOfN) produces the same ODS within ±0.003 — the solver bias dominates the graph structure difference.

### 2. Fused CUDA Edge Builder (`edge_builder.cu`)

Replaces three separate operations: `torch.cdist` (distance matrix) → `mask` (threshold) → `torch.nonzero` (extract edges). The fused kernel does all three in a single grid-stride loop over (pred, gt) pairs, using `atomicAdd` to write edges to a flat buffer. Result: 0.02s for all 99 thresholds (was 0.67s, 32× faster).

### 3. Consecutive Stall Detection

The original kernel exited epsilon=0 if ANY single round had no assignment changes. This was too aggressive — unassigned persons keep bidding without winning (price still below threshold), triggering a false stall. The fix: count CONSECUTIVE no-change rounds. Only exit after N consecutive rounds. Found optimal: STALL_EPS_0=200 for eps=0.

### 4. ITERS_EPS0 Tuning

Systematic sweep of 10 configurations:

| ITERS_EPS0 | STALL_EPS_0 | Time | ΔODS |
|------------|-------------|------|------|
| 5000 | 50 | 0.570s | +0.0031 |
| 3000 | 50 | 0.525s | +0.0030 |
| 2000 | 50 | 0.503s | +0.0027 |
| 1000 | 50 | 0.489s | +0.0027 |
| **500** | **200** | **0.468s** | **+0.0026** |
| 250 | 200 | 0.470s | +0.0023 |

Key insight: the Auction algorithm converges within 200-300 iterations for virtually all BSDS500 problems. Running 5000 iterations wastes 90% of the time. Lower ITERS + higher STALL is the winning combo — give every chance to converge, but don't force unnecessary iterations.
