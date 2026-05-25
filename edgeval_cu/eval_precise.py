"""gpu_eval_precise.py — Precise mode: multi-process CSA via fast_match_edge_maps.

Uses the exact same C++ CSA solver as the reference MATLAB implementation.
Multi-process parallelism across thresholds for speed.

~1.5s/image (8 workers), ODS identical to CPU CSA reference.
"""
import numpy as np
import multiprocessing as mp
from scipy.io import loadmat

from .nms_thin import bwmorph_thin
from .metrics import eps as EVAL_EPS


def _csa_worker(edge, gt_raw, thrs_arr, indices, H, W, max_dist_px, oc_val, queue):
    """Worker process: evaluate assigned thresholds using CSA."""
    from .csa import fast_match_edge_maps

    all_g_sum = np.zeros((H, W), dtype=np.int64)
    for g in gt_raw:
        all_g_sum += g.astype(np.int64)

    for k in indices:
        t = thrs_arr[k]
        e1 = edge >= max(EVAL_EPS, t)
        e1 = bwmorph_thin(e1)

        pb = np.zeros((H, W), dtype=bool)
        pp = np.stack(np.where(e1), axis=1)
        pb[pp[:, 0], pp[:, 1]] = True

        me_agg = np.zeros((H, W), dtype=bool)
        mg_agg = np.zeros((H, W), dtype=np.int32)

        for g in gt_raw:
            gb = np.zeros((H, W), dtype=bool)
            gb[g] = True
            m1, m2, _ = fast_match_edge_maps(pb, gb, max_dist_px, oc_val)
            me_agg = np.logical_or(me_agg, m1 > 0)
            mg_agg = mg_agg + (m2 > 0)

        queue.put((k, int(mg_agg.sum()), int(all_g_sum.sum()),
                    int(me_agg.sum()), int(e1.sum())))


def gpu_eval_img_precise(edge_prob, gt_path, thrs=99, max_dist=0.0075,
                         thin=True, workers=8):
    """Precise mode: exact CSA solver with multi-process parallelism.

    Args:
        edge_prob: 2D edge probability map (H, W), values in [0, 1]
        gt_path: path to .mat ground truth file
        thrs: number of thresholds (int) or array
        max_dist: max matching distance ratio
        thin: apply thinning
        workers: number of parallel processes (default 8)

    Returns:
        info: (k, 5) array [threshold, matched_gt, total_gt, matched_pred, total_pred]
    """
    if edge_prob.ndim != 2:
        raise ValueError("edge_prob must be 2D")

    try:
        gt_raw = [g.item()[1] for g in loadmat(gt_path)["groundTruth"][0]]
    except Exception:
        gt_raw = [g.item()[0] for g in loadmat(gt_path)["groundTruth"][0]]

    if isinstance(thrs, int):
        k = thrs
        thrs_vals = np.linspace(1 / (k + 1), 1 - 1 / (k + 1), k)
    else:
        k = len(thrs)
        thrs_vals = np.asarray(thrs)

    H, W = edge_prob.shape
    idiag = np.sqrt(H * H + W * W)
    max_dist_px = max_dist * idiag
    oc_val = 100.0 * max_dist_px

    workers = min(workers, k)
    if workers <= 0:
        workers = 1

    cnt_sum = np.zeros((k, 4), dtype=np.int64)

    # Import here so workers=1 doesn't need multiprocessing
    from .csa import fast_match_edge_maps

    if workers == 1:
        # Serial mode
        all_g_sum = np.zeros((H, W), dtype=np.int64)
        for g in gt_raw:
            all_g_sum += g.astype(np.int64)

        for k_idx in range(k):
            t = thrs_vals[k_idx]
            e1 = edge_prob >= max(EVAL_EPS, t)
            if thin:
                e1 = bwmorph_thin(e1)
            pb = np.zeros((H, W), dtype=bool)
            pp = np.stack(np.where(e1), axis=1)
            pb[pp[:, 0], pp[:, 1]] = True
            me_agg = np.zeros((H, W), dtype=bool)
            mg_agg = np.zeros((H, W), dtype=np.int32)
            for g in gt_raw:
                gb = np.zeros((H, W), dtype=bool)
                gb[g] = True
                m1, m2, _ = fast_match_edge_maps(pb, gb, max_dist_px, oc_val)
                me_agg = np.logical_or(me_agg, m1 > 0)
                mg_agg = mg_agg + (m2 > 0)
            cnt_sum[k_idx] = [int(mg_agg.sum()), int(all_g_sum.sum()),
                              int(me_agg.sum()), int(e1.sum())]
    else:
        # Multi-process mode
        queue = mp.SimpleQueue()
        chunks = np.array_split(range(k), workers)
        procs = []
        for chunk in chunks:
            p = mp.Process(target=_csa_worker,
                           args=(edge_prob, gt_raw, thrs_vals, chunk,
                                 H, W, max_dist_px, oc_val, queue))
            p.start()
            procs.append(p)

        done = 0
        while done < k:
            k_idx, a, b, c, d = queue.get()
            cnt_sum[k_idx] = [a, b, c, d]
            done += 1

        for p in procs:
            p.join()

    info = np.concatenate([thrs_vals[:, None], cnt_sum], axis=1)
    return info