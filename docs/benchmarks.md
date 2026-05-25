# Benchmarks

All measurements on RTX 4090 (CUDA 12.x, sm_89), AMD Ryzen 9 7950X, BSDS500 test set.

## Overall Performance

| Mode | Time/img | 200 imgs | vs CPU (20 min) | ΔODS vs CSA |
|------|----------|----------|-----------------|-------------|
| CPU CSA (MATLAB ref) | ~6s | ~20 min | 1× | 0 (reference) |
| GPU simple (fast) | **0.47s** | **1.6 min** | **12.8×** | +0.003 |
| GPU extended (precise) | ~5.7s | ~19 min | ~1× | <0.001 |

## Pipeline Breakdown (GPU simple mode)

Per-image timing for a typical BSDS500 image (321×481), 99 thresholds, ~5 GT annotations:

| Component | Time | % | Implementation |
|-----------|------|----|---------------|
| GPU batched thinning | 0.12s | 25% | PyTorch conv2d, 99 masks in batch |
| Fused CUDA edge builder | 0.02s | 4% | Single kernel launch, 485 problems |
| GPU sort + annotator split | 0.08s | 17% | Compound key sort + bucketize |
| Download + problem build | 0.04s | 9% | Single download, boundary split |
| GPU Auction solve | **0.14s** | **30%** | 485 problems, eps-scaling |
| Overhead (I/O, upload) | 0.07s | 15% | GT loading, pred stack, numpy ops |
| **Total** | **0.47s** | 100% | |

## Auction Solver Configurations

Systematic sweep of ITERS_EPS0 and STALL_EPS_0 on 5 representative images:

| ITERS_EPS0 | STALL_EPS_0 | Time/img | ODS (GPU) | ODS (CSA) | ΔODS |
|------------|-------------|----------|-----------|-----------|------|
| 5000 | 50 | 0.570s | 0.7939 | 0.7908 | +0.0031 |
| 5000 | 100 | 0.566s | 0.7939 | 0.7908 | +0.0031 |
| 5000 | 200 | 0.568s | 0.7939 | 0.7908 | +0.0031 |
| 3000 | 50 | 0.525s | 0.7938 | 0.7908 | +0.0030 |
| 3000 | 100 | 0.524s | 0.7938 | 0.7908 | +0.0030 |
| 2000 | 50 | 0.503s | 0.7936 | 0.7908 | +0.0027 |
| 2000 | 100 | 0.511s | 0.7936 | 0.7908 | +0.0027 |
| 1000 | 50 | 0.489s | 0.7935 | 0.7908 | +0.0027 |
| 1000 | 200 | 0.492s | 0.7935 | 0.7908 | +0.0027 |
| **500** | **200** | **0.468s** | **0.7934** | **0.7908** | **+0.0026** |
| 250 | 200 | 0.470s | 0.7931 | 0.7908 | +0.0023 |

**Optimal default**: ITERS_EPS0=500, STALL_EPS_0=200.

## Per-Image Results (GPU simple vs CPU CSA)

| Image | GPU ODS | CSA ODS | ΔODS | Time |
|-------|---------|---------|------|------|
| 100007 | 0.8601 | 0.8570 | +0.0031 | 0.85s |
| 100039 | 0.7358 | 0.7347 | +0.0011 | 0.61s |
| 100099 | 0.8091 | 0.8056 | +0.0035 | 0.40s |
| 10081 | 0.7655 | 0.7631 | +0.0024 | 0.61s |
| 101027 | 0.8749 | 0.8724 | +0.0026 | 0.42s |

Time variation reflects different edge pixel counts per image.

## Accuracy Notes

- ΔODS ≈ +0.003 is **systematic** — GPU Auction `atomicMax` tie-breaking consistently favors slightly more matches than CSA's deterministic cost-scaling.
- This bias is **stable** across images and iterations — the ranking/trend of different models is preserved.
- For training-time monitoring: use GPU simple mode (fast, Δ~0.003).
- For final paper evaluation: use CPU CSA (exact, ~20 min).
