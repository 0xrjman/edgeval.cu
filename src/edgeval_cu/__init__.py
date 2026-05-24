"""
edgeval.cu — GPU-accelerated edge detection evaluation.

A Python + CUDA implementation of standard edge detection evaluation metrics
(ODS, OIS, AP, R50) with a GPU-accelerated Auction Algorithm solver that
achieves ~7.4× speedup over CPU CSA solvers while maintaining exact metric
consistency.
"""

__version__ = "0.1.0"
__author__ = "0xrjman & Contributors"

from ._impl import (
    bwmorph_thin, conv_tri, grad2,
    correspond_pixels, edges_eval_dir, compute_rpf, find_best_rpf,
)
from .nms_process import nms_process, nms_process_one_image
from .eval_component import eval_one_epoch

try:
    from .gpu_eval import batch_solve, gpu_edges_eval_img, gpu_edges_eval_dir
    _cuda_available = True
except Exception:
    _cuda_available = False

def cuda_available():
    """Check if GPU acceleration is available."""
    return _cuda_available
