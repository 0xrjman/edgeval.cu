"""Evaluation component — orchestrates NMS + eval for a result directory."""
import os
from .nms_process import nms_process
from ._impl.edges_eval_dir import edges_eval_dir


def eval_one_epoch(root, dataset, full=False, key="img", file_format=".mat"):
    """Run full NMS + evaluation for one result directory."""
    print(root)
    result_dir = os.path.join(root, "mat")
    nms_dir = os.path.join(root, "nms")
    datasets = {
        "BSDS": "GT/BSDS",
        "BIPED": "GT/BIPED",
        "NYUD": "GT/NYUD",
        "UDED": "GT/UDED",
    }
    gt_dir = datasets[dataset]
    thrs = 99 if full else 9
    nms_process(result_dir, nms_dir, key, file_format)
    edges_eval_dir(nms_dir, gt_dir, thrs=thrs, thin=1, max_dist=0.0075, workers=-1)
