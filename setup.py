import sys
if sys.version_info < (3, 8):
    sys.exit("Python >= 3.8 required")

from setuptools import setup, find_packages

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["edgeval_cu", "edgeval_cu.*"]),
    package_data={
        "edgeval_cu": [
            "cxx/lib/solve_csa.so",
            "gpu_eval/auction_cuda.so",
        ],
    },
    zip_safe=False,
)
