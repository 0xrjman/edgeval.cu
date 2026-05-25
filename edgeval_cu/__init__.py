"""
edgeval.cu — GPU-accelerated edge detection evaluation.

A Python + CUDA implementation of standard edge detection evaluation metrics
(ODS, OIS, AP, R50) with a GPU-accelerated Auction Algorithm solver that
achieves ~10× speedup over CPU CSA solvers.
"""

__version__ = "0.1.0"
__author__ = "0xrjman & Contributors"

# Core modules (flattened from _impl/)
from .nms_thin import bwmorph_thin
from .toolbox import conv_tri, grad2
from .csa import correspond_pixels
from .metrics import edges_eval_dir, compute_rpf, find_best_rpf
from .nms_process import nms_process, nms_process_one_image
from .eval_component import eval_one_epoch

# GPU acceleration (optional)
try:
    from .auction import batch_solve
    from .eval import gpu_edges_eval_img, gpu_edges_eval_dir
    _cuda_available = True
except Exception:
    _cuda_available = False


def cuda_available():
    """Check if GPU acceleration is available."""
    return _cuda_available
