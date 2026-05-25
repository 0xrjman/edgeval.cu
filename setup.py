import sys
if sys.version_info < (3, 8):
    sys.exit("Python >= 3.8 required")

from setuptools import setup, find_packages

setup(
    package_dir={"": "."},
    packages=find_packages(where=".", include=["edgeval_cu", "edgeval_cu.*"]),
    package_data={
        "edgeval_cu": [
            "cxx/lib/solve_csa.so",
            "cuda/auction_cuda.so",
            "cuda/edge_builder.so",
        ],
    },
    zip_safe=False,
)
