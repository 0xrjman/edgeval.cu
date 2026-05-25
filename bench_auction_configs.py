#!/usr/bin/env python3
"""Benchmark different ITERS_EPS0 and STALL_EPS_0 configurations.

Modifies auction_kernel.cu #defines, recompiles, runs 5-image eval,
and records ODS, time, and ΔODS for each configuration.
"""
import subprocess, os, time, sys, re
import numpy as np
import cv2

# Configure paths via environment variables or defaults.
REPO = os.environ.get('BENCH_REPO', os.path.dirname(os.path.abspath(__file__)))
CUDA_DIR = f'{REPO}/edgeval_cu/cuda'
KERNEL_FILE = f'{CUDA_DIR}/auction_kernel.cu'

PRED_DIR = os.environ.get('BENCH_PRED_DIR',
    os.path.join(REPO, '..', 'edge_eval_python', 'examples', 'eval-result', 'hed'))
GT_DIR = os.environ.get('BENCH_GT_DIR',
    os.path.expanduser('~/data/BSR/BSDS500/data/groundTruth/test'))
EVAL_DIR = os.environ.get('BENCH_EVAL_DIR',
    os.path.join(REPO, '..', 'edge_eval_python', 'examples', 'eval-result', 'hed-eval'))
IMG_IDS = ['100007', '100039', '100099', '10081', '101027']


def set_defines(iters_eps0, stall_eps0):
    """Modify the #define values in auction_kernel.cu."""
    content = open(KERNEL_FILE).read()
    content = re.sub(r'#define DEFAULT_ITERS_EPS0\s+\d+',
                     f'#define DEFAULT_ITERS_EPS0  {iters_eps0}', content)
    content = re.sub(r'#define DEFAULT_STALL_EPS_0\s+\d+',
                     f'#define DEFAULT_STALL_EPS_0   {stall_eps0}', content)
    open(KERNEL_FILE, 'w').write(content)


def compile_and_bench(iters_eps0, stall_eps0):
    """Recompile, then run 5-image benchmark."""
    set_defines(iters_eps0, stall_eps0)

    # Recompile
    subprocess.run(['make', '-C', CUDA_DIR, 'auction_cuda.so'],
                   capture_output=True, check=True)

    # Reload module (Python caches ctypes CDLL — need fresh import)
    # Use subprocess to avoid caching issues
    cmd = f"""
import sys; sys.path.insert(0,'.')
import numpy as np, cv2, time
from edgeval_cu.eval import gpu_edges_eval_img
from edgeval_cu.metrics import compute_rpf, find_best_rpf

ids = {IMG_IDS}
thrs=99; t=np.linspace(1/100,1-1/100,99)
cg=np.zeros((thrs,4),dtype=np.int64); cc=np.zeros((thrs,4),dtype=np.int64); times=[]

for iid in ids:
    edge=cv2.imread('{PRED_DIR}/' + iid + '.png', cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
    t0=time.time()
    info,_=gpu_edges_eval_img(edge, '{GT_DIR}/' + iid + '.mat', thrs=thrs, mode='simple')
    elapsed=time.time()-t0
    times.append(elapsed)
    cg+=info[:,1:5].astype(np.int64)
    import os
    et='{EVAL_DIR}/' + iid + '_ev1.txt'
    if os.path.exists(et):
        d=np.loadtxt(et); cc+=d[:,1:5].astype(np.int64)

rg,pg,fg=compute_rpf(cg); _,_,ods_g,_=find_best_rpf(t,rg,pg)
rc,pc,fc=compute_rpf(cc); _,_,ods_c,_=find_best_rpf(t,rc,pc)
print(f'{{np.mean(times):.4f}} {{ods_g:.4f}} {{ods_c:.4f}} {{ods_g-ods_c:+.4f}}')
"""
    result = subprocess.run(['python3', '-c', cmd],
                           capture_output=True, text=True, timeout=120,
                           cwd=REPO)
    if result.returncode != 0:
        return None, result.stderr
    parts = result.stdout.strip().split()
    if len(parts) >= 4:
        return {
            'time': float(parts[0]),
            'ods_gpu': float(parts[1]),
            'ods_csa': float(parts[2]),
            'delta': float(parts[3]),
            'iters': iters_eps0,
            'stall': stall_eps0,
        }, None
    return None, result.stdout


# ── Run benchmark ────────────────────────────────────────────────────

configs = [
    # (ITERS_EPS0, STALL_EPS_0)
    (5000, 50),    # current baseline
    (5000, 100),
    (5000, 200),
    (3000, 50),
    (3000, 100),
    (2000, 50),
    (2000, 100),
    (1000, 50),
    (1000, 200),
    (500, 200),
]

results = []
for iters, stall in configs:
    print(f'\nTesting ITERS_EPS0={iters}, STALL_EPS_0={stall} ...', flush=True)
    result, err = compile_and_bench(iters, stall)
    if result:
        results.append(result)
        print(f"  time={result['time']:.3f}s  ODS={result['ods_gpu']:.4f}  "
              f"Δ={result['delta']:+.4f}")
    else:
        print(f"  FAILED: {err[:200]}")

# ── Summary ──────────────────────────────────────────────────────────
print('\n' + '='*70)
print(f'{"ITERS":>6} {"STALL":>6} {"Time":>8} {"ODS_gpu":>9} {"ODS_csa":>9} {"ΔODS":>8}')
print('-'*50)
baseline_delta = None
for r in sorted(results, key=lambda x: x['time']):
    if r['iters'] == 5000 and r['stall'] == 50:
        baseline_delta = r['delta']
    marker = ''
    if abs(r['delta']) <= 0.004:
        marker = ' ✓'
    print(f"{r['iters']:>6} {r['stall']:>6} {r['time']:>7.3f}s {r['ods_gpu']:>9.4f} "
          f"{r['ods_csa']:>9.4f} {r['delta']:>+8.4f}{marker}")

print(f'\nΔODS threshold: |Δ| ≤ 0.004 (acceptable for training)')
if baseline_delta:
    print(f'Baseline (5000/50): Δ={baseline_delta:+.4f}')

# Restore baseline
set_defines(5000, 50)
subprocess.run(['make', '-C', CUDA_DIR, 'auction_cuda.so'],
               capture_output=True)
print('\nRestored baseline config.')
