"""Build script for edgeval-cu — compiles CUDA kernels at first import."""
import os
import sys

if sys.version_info < (3, 8):
    sys.exit("Python >= 3.8 required")

from setuptools import setup, find_packages

setup(
    name="edgeval-cu",
    version="0.1.0",
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
        "Programming Language :: CUDA",
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
        ],
        "cxx": [
            "lib/*.so",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "edgeval=edgeval_cu.cli:cli",
        ],
    },
    zip_safe=False,
)
