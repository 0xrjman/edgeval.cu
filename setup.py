"""Build script for edgeval — compiles CUDA kernels at pip install time."""
import os
import sys
import subprocess
import shutil

if sys.version_info < (3, 8):
    sys.exit("Python >= 3.8 required")

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Dummy extension to force build_ext to run (triggers CUDA compilation).
# The .c file is minimal — it produces no real code, just satisfies setuptools.
_DUMMY_EXT = Extension(
    "edgeval_cu._dummy",
    sources=["edgeval_cu/_dummy.c"],
)


def _detect_nvcc():
    """Find nvcc binary path."""
    for path in shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc":
        if path and os.path.isfile(path):
            return path
    raise RuntimeError(
        "nvcc not found. Install CUDA toolkit:\n"
        "  sudo apt install nvidia-cuda-toolkit   # Debian/Ubuntu\n"
        "  conda install cuda-toolkit             # conda"
    )


def _detect_arch():
    """Detect GPU compute capability via PyTorch or nvidia-smi."""
    # Try PyTorch first (available at install time since it's a dependency)
    try:
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return f"sm_{capability[0]}{capability[1]}"
    except Exception:
        pass
    # Fallback: nvidia-smi + hardcoded map
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10
        ).decode().strip().split()[0]
        return f"sm_{out.replace('.', '')}"
    except Exception:
        pass
    # Default to sm_80 (Turing) — broad compatibility
    print("WARNING: could not detect GPU arch, defaulting to sm_80")
    return "sm_80"


def _compile_cuda(nvcc, arch, src, out):
    """Compile a single .cu file to .so using nvcc."""
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    cmd = [
        nvcc,
        "-shared", "-O3", "--expt-relaxed-constexpr",
        "-Xcompiler", "-fPIC",
        "-arch", arch,
        "-I", os.path.join(cuda_home, "include"),
        "-I", os.path.join(cuda_home, "targets",
                           f"x86_64-{os.uname().sysname.lower()}", "include"),
        "-L", os.path.join(cuda_home, "lib64"),
        "-lcuda", "-lcudart",
        "-o", out, src,
    ]
    print(f"  nvcc: {os.path.basename(src)} -> {os.path.basename(out)}")
    subprocess.check_call(cmd)


class CudaBuildExt(build_ext):
    """Custom build_ext that compiles CUDA kernels during pip install."""

    def run(self):
        # Compile CUDA kernels before standard build
        self._compile_cuda_kernels()
        super().run()

    def _compile_cuda_kernels(self):
        """Compile .cu -> .so into the package's cuda/ directory."""
        nvcc = _detect_nvcc()
        arch = _detect_arch()
        print(f"edgeval: compiling CUDA kernels (nvcc, arch={arch})")

        # Source is in the package directory (edgeval_cu/cuda/)
        pkg_dir = os.path.join(self.build_lib, "edgeval_cu")
        cu_dir = os.path.join(pkg_dir, "cuda")
        os.makedirs(cu_dir, exist_ok=True)

        kernels = {
            "auction_kernel.cu": "auction_cuda.so",
            "edge_builder.cu":   "edge_builder.so",
        }
        # Find source .cu files — they should be in the build tree already
        for cu_name, so_name in kernels.items():
            src = os.path.join(cu_dir, cu_name)
            dst = os.path.join(cu_dir, so_name)
            if os.path.isfile(src):
                _compile_cuda(nvcc, arch, src, dst)
            else:
                raise RuntimeError(
                    f"Source file not found: {src}\n"
                    f"This should not happen — .cu files should be in the build tree."
                )
        print("  CUDA compilation complete")


setup(
    name="edgeval",
    version="0.1.1",
    description="GPU-accelerated edge detection evaluation (ODS/OIS/AP/R50)",
    long_description=open("README.md").read() if os.path.isfile("README.md") else "",
    long_description_content_type="text/markdown",
    author="0xrjman & Contributors",
    license="Apache-2.0",
    url="https://github.com/0xrjman/edgeval.cu",
    keywords="edge-detection evaluation cuda gpu ods ois bsds500",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy>=1.6.0",
        "opencv-python",
        "tqdm",
        "click",
        "torch>=2.0",
    ],
    package_dir={"": "."},
    packages=find_packages(where=".", include=["edgeval_cu", "edgeval_cu.*", "cxx"]),
    package_data={
        "edgeval_cu": [
            "cuda/*.cu",
            "cuda/*.so",
        ],
        "cxx": [
            "lib/*.so",
        ],
    },
    include_package_data=True,
    ext_modules=[_DUMMY_EXT],
    cmdclass={
        "build_ext": CudaBuildExt,
    },
    entry_points={
        "console_scripts": [
            "edgeval=edgeval_cu.cli:cli",
        ],
    },
    zip_safe=False,
)
