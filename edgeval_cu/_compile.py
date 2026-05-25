"""Compile CUDA kernels on first import.

This module handles lazy compilation of .cu source files to .so shared
libraries.  Compilation happens once — subsequent imports skip it if the
.so files already exist and are newer than their .cu sources.
"""
import os
import subprocess


# ── Kernel definitions ──────────────────────────────────────────────

CUDA_KERNELS = [
    ("auction_kernel.cu", "auction_cuda.so"),
    ("edge_builder.cu", "edge_builder.so"),
]

# Directory containing the .cu source files (edgeval_cu/cuda/)
_CU_DIR = os.path.join(os.path.dirname(__file__), "cuda")


# ── CUDA detection ──────────────────────────────────────────────────

def _get_cuda_arch():
    """Detect GPU compute capability for nvcc -arch flag."""
    try:
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return f"sm_{capability[0]}{capability[1]}"
    except Exception:
        pass
    # Fallback: try nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader"],
            timeout=5, text=True,
        ).strip().split(",")[0].strip()
        return f"sm_{out.replace('.', '')}"
    except Exception:
        pass
    # Default to sm_80 (Turing) — broad compatibility
    return "sm_80"


def _find_nvcc():
    """Find nvcc binary path."""
    candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    try:
        return subprocess.check_output(
            ["which", "nvcc"], timeout=5, text=True
        ).strip()
    except Exception:
        return None


# ── Compilation ─────────────────────────────────────────────────────

def _need_compile():
    """Check if any CUDA kernel needs compilation."""
    for src_name, so_name in CUDA_KERNELS:
        src = os.path.join(_CU_DIR, src_name)
        dst = os.path.join(_CU_DIR, so_name)
        if not os.path.isfile(dst):
            return True
        if os.path.getmtime(src) > os.path.getmtime(dst):
            return True
    return False


def _compile():
    """Compile all CUDA kernels in-place next to .cu sources."""
    arch = _get_cuda_arch()
    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError(
            "edgeval-cu: nvcc not found. "
            "Please install CUDA toolkit "
            "(e.g. sudo apt install nvidia-cuda-toolkit)."
        )

    print(f"edgeval-cu: compiling CUDA kernels (arch={arch})")

    for src_name, so_name in CUDA_KERNELS:
        src = os.path.join(_CU_DIR, src_name)
        dst = os.path.join(_CU_DIR, so_name)
        if os.path.isfile(dst) and os.path.getmtime(src) <= os.path.getmtime(dst):
            print(f"  skipping {so_name} (up to date)")
            continue

        cmd = [
            nvcc, "-shared", f"-arch={arch}", "-O3",
            "-Xcompiler", "-fPIC", "-Xcompiler", "-Wall",
            "-o", dst, src,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"edgeval-cu: failed to compile {src_name}.\n"
                f"  stderr: {result.stderr}"
            )
        print(f"  built {so_name}")


# ── Public API ──────────────────────────────────────────────────────

def ensure_compiled():
    """Ensure CUDA kernels are compiled.  Call this before loading .so files."""
    if _need_compile():
        _compile()
