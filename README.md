<p align="center">
  <strong>edgeval.cu</strong><br>
  <em>GPU-accelerated edge detection evaluation — ODS · OIS · AP · R50</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia" alt="CUDA">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
</p>

---

## Overview

Standard edge detection evaluation (ODS/OIS/AP/R50) requires solving ~99,000 independent assignment problems — one for every (threshold × ground-truth-annotation) pair across a benchmark dataset. The CPU CSA pipeline takes ~20 minutes for BSDS500.

**edgeval.cu** replaces this with a fused GPU pipeline that achieves **0.47s per image, 1.6 minutes for BSDS500** — a **12.8× speedup** over CPU on an RTX 4090.

---

## How It Works

### 1. Edge Matching as Assignment Problem

Predicted edge pixels must be matched to ground truth edge pixels within a spatial tolerance (`max_dist`, typically 0.75% of image diagonal). Each match incurs a cost proportional to Euclidean distance. Unmatched pixels pay an outlier penalty.

This is a **minimum-cost bipartite assignment problem**: persons = predicted pixels, objects = ground truth pixels, costs = distances.

### 2. Auction Algorithm (Bertsekas, 1979)

The Auction algorithm solves assignment problems through iterative bidding:

```
For each unassigned person:
  1. Find best object (lowest cost + current price)
  2. Compute bid increment from gap to second-best
  3. Place bid via atomicMax (race-free on GPU)

For each object:
  4. Accept highest bidder, evict previous owner
  5. Update price to new bid

Repeat until all assigned or stalled.
```

**Epsilon-scaling** (ε = 8 → 4 → 2 → 1 → 0) guarantees optimality: coarse ε finds a rough solution quickly, fine ε refines it exactly.

### 3. GPU Pipeline

```
Input Edge Map
    │
    ▼
┌─────────────────────┐
│ Batched Thinning    │  Zhang-Suen via PyTorch conv2d
│ 99 thresholds × GPU │  0.12s
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ CUDA Edge Builder   │  Single kernel: distance + threshold
│ Fused cdist+mask    │  0.02s
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ GPU Sort + Split    │  Compound key: (annotator, person, object)
│ Bucketize by GT     │  0.08s
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Auction Solver      │  485 problems, one CUDA block each
│ ε-scaling + stall   │  0.14s
└─────────┬───────────┘
          ▼
     ODS / OIS / AP / R50
```

### 4. Stall Detection

A critical optimization: instead of exiting when a single round produces no new assignments (which happens when bids haven't accumulated enough to overcome prices), we count **consecutive** no-change rounds. eps=0 waits 200 consecutive rounds before giving up — ensuring true convergence without wasting iterations.

### 5. Two Graph Modes

| Mode | Graph | Speed | ΔODS |
|------|-------|-------|------|
| `simple` | Bipartite: real edges only | 0.47s | +0.003 |
| `extended` | n×n: kOfN + diagonal overlay | ~5.7s | <0.001 |

The `simple` graph skips kOfN random outlier edges. The result is 12× faster with a stable +0.003 ODS bias — perfectly adequate for training monitoring. Use `extended` mode or CPU CSA for final paper evaluation.

---

## Installation

```bash
git clone https://github.com/0xrjman/edgeval.cu.git
cd edgeval.cu
pip install -e .
cd edgeval_cu/cuda && make
```

**Requirements**: Python 3.8+, NumPy, SciPy >= 1.6.0, PyTorch, CUDA 12.x, OpenCV, tqdm.

---

## Quick Start

```bash
# GPU evaluation
edgeval eval results --gpu --dataset BSDS

# Python API
from edgeval_cu.eval import gpu_edges_eval_img
info, _ = gpu_edges_eval_img(edge_map, "GT/100007.mat", thrs=99, mode='simple')
```

See [docs/benchmarks.md](docs/benchmarks.md) for detailed numbers and [docs/optimization.md](docs/optimization.md) for the optimization journey.

---

## Performance

| Scenario | CPU CSA | GPU simple | Speedup |
|----------|---------|------------|---------|
| 1 image | ~6s | **0.47s** | **12.8×** |
| 200 images (BSDS500) | ~20 min | **1.6 min** | **12.8×** |

ΔODS vs reference: **+0.003** (stable, systematic).

Full pipeline breakdown and configuration sweep: [docs/benchmarks.md](docs/benchmarks.md).

---

## Project Structure

```
edgeval.cu/
├── edgeval_cu/                  # Package
│   ├── eval.py                  # Main pipeline
│   ├── auction.py               # GPU Auction wrapper
│   ├── thin.py                  # GPU thinning (inline)
│   ├── metrics.py               # ODS/OIS/AP/R50 computation
│   ├── csa.py                   # CPU CSA solver (reference)
│   ├── nms_thin.py              # Zhang-Suen LUTs + CPU fallback
│   ├── cli.py, show.py          # CLI interface
│   └── cuda/                    # CUDA source
│       ├── auction_kernel.cu    # Auction solver (ε-scaling)
│       ├── edge_builder.cu      # Fused edge builder
│       └── Makefile
├── docs/
│   ├── benchmarks.md            # Detailed benchmarks
│   └── optimization.md          # Optimization journey
└── README.md
```

---

## References

- [Bertsekas, "Auction Algorithms" (1979)](https://web.mit.edu/dimitrib/www/Auction_Encycl.pdf)
- [Guo & Hall, "Parallel Thinning" (1989)](https://gist.github.com/joefutrelle/562f25bbcf20691217b8)
- [HED evaluation (MATLAB)](https://github.com/s9xie/hed_release-deprecated)
- [extended-berkeley-segmentation-benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark)
- [edge-eval-python](https://github.com/Walstruzz/edge_eval_python) — Python CSA port
