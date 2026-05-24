"""gpu_eval.py — GPU-accelerated edge evaluation pipeline.

Batches ALL (threshold, GT) pairs of one image into a single
auction kernel launch on GPU.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat
from scipy.interpolate import interp1d
from .._impl.bwmorph_thin import bwmorph_thin
from .._impl.edges_eval_dir import compute_rpf, find_best_rpf, eps as EVAL_EPS
from .gpu_auction import batch_solve
import os
import glob
from tqdm import tqdm


def build_problem(e1_binary, g_binary, max_dist, outlier_cost):
    """Build one assignment problem for GPU auction."""
    pred_pixels = np.stack(np.where(e1_binary), axis=1).astype(np.float64)
    gt_pixels = np.stack(np.where(g_binary), axis=1).astype(np.float64)
    n1, n2 = len(pred_pixels), len(gt_pixels)
    if n1 == 0 or n2 == 0:
        return None
    gt_tree = cKDTree(gt_pixels)
    pairs = cKDTree(pred_pixels).sparse_distance_matrix(
        gt_tree, max_dist, output_type='coo_matrix')
    if len(pairs.data) == 0:
        return None
    mult = 100
    oc_int = int(np.ceil(outlier_cost * mult))
    edges = np.zeros((len(pairs.data), 3), dtype=np.int32)
    edges[:, 0] = pairs.row.astype(np.int32)
    edges[:, 1] = pairs.col.astype(np.int32)
    edges[:, 2] = np.rint(pairs.data * mult).astype(np.int32)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return {
        'n_persons': n1, 'n_objects': n2,
        'edges': edges, 'outlier_cost': oc_int,
    }


def gpu_edges_eval_img(edge_prob, gt_path, thrs=99, max_dist=0.0075,
                       thin=True, need_v=False):
    """GPU-accelerated edge evaluation for one image."""
    H, W = edge_prob.shape
    idiag = np.sqrt(H * H + W * W)
    max_dist_px = max_dist * idiag
    oc = 100 * max_dist_px

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

    all_binary = []
    for t in thrs_vals:
        e1 = edge_prob >= max(EVAL_EPS, t)
        if thin:
            e1 = bwmorph_thin(e1)
        all_binary.append(e1)

    problems = []
    prob_map = []
    for k_idx, e1 in enumerate(all_binary):
        for g_idx, g in enumerate(gt_raw):
            prob = build_problem(e1, g, max_dist_px, oc)
            if prob is None:
                prob_map.append((k_idx, g_idx, -1))
            else:
                prob_idx = len(problems)
                problems.append(prob)
                prob_map.append((k_idx, g_idx, prob_idx))

    assignments = batch_solve(problems) if problems else []

    cnt_sum_r_p = np.zeros((k, 4), dtype=np.int64)
    # matched_pred_union: track which pred pixel indices are matched across
    # all GT annotations for each threshold (OR semantics, like CPU version's
    # np.logical_or(match_e, match_e1 > 0) across GTs)
    matched_pred_union = [set() for _ in range(k)]

    for item_idx, (k_idx, g_idx, prob_idx) in enumerate(prob_map):
        g = gt_raw[g_idx]
        cnt_sum_r_p[k_idx, 1] += g.sum()  # total_GT: sum across GTs
        if prob_idx < 0:
            continue
        assign = assignments[prob_idx]
        n2 = problems[prob_idx]['n_objects']
        # matched GT pixels (unique per GT annotation, summed across GTs)
        unique_gt = len(np.unique(assign[assign < n2]))
        cnt_sum_r_p[k_idx, 0] += unique_gt
        # Track which pred pixel indices are matched (OR across GTs)
        matched_idx = np.where(assign < n2)[0]
        matched_pred_union[k_idx].update(matched_idx.tolist())

    # Populate total_pred and matched_pred once per threshold (not per GT)
    for k_idx in range(k):
        e1 = all_binary[k_idx]
        cnt_sum_r_p[k_idx, 2] = len(matched_pred_union[k_idx])
        cnt_sum_r_p[k_idx, 3] = e1.sum()

    info = np.concatenate([thrs_vals[:, None], cnt_sum_r_p], axis=1)
    return info, None


def gpu_edges_eval_dir(res_dir, gt_dir, thrs=99, max_dist=0.0075,
                       thin=True, cleanup=0):
    """GPU-accelerated directory-level edge evaluation."""
    import cv2
    eval_dir = res_dir + "-eval-gpu"
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    ids = [os.path.splitext(os.path.basename(f))[0]
           for f in glob.glob(os.path.join(gt_dir, "*.mat"))]

    t = np.linspace(1 / (thrs + 1), 1 - 1 / (thrs + 1), thrs)
    cnt_sum = np.zeros((thrs, 4), dtype=np.int64)
    ois_sum = np.zeros(4, dtype=np.int64)
    scores = np.zeros((len(ids), 5), dtype=np.float32)

    for ci, name in enumerate(tqdm(ids)):
        res_path = os.path.join(res_dir, name + ".png")
        gt_path = os.path.join(gt_dir, name + ".mat")
        out_path = os.path.join(eval_dir, name + "_ev1.txt")

        if os.path.isfile(out_path):
            data = np.loadtxt(out_path, dtype=np.float32)
            img_cnt = data[:, 1:5].astype(np.int64)
        else:
            edge = cv2.imread(res_path, cv2.IMREAD_UNCHANGED) / 255.0
            if edge.ndim != 2:
                continue
            info_v, _ = gpu_edges_eval_img(edge, gt_path, thrs=thrs,
                                           max_dist=max_dist, thin=thin)
            img_cnt = info_v[:, 1:5].astype(np.int64)
            np.savetxt(out_path, info_v, fmt="%10g")

        cnt_sum += img_cnt
        r, p, f = compute_rpf(img_cnt)
        k_best = f.argmax()
        ois_r1, ois_p1, ois_f1, ois_t1 = find_best_rpf(t, r, p)
        scores[ci, :] = [ci + 1, ois_t1, ois_r1, ois_p1, ois_f1]
        ois_sum += img_cnt[k_best, :]

    r, p, f = compute_rpf(cnt_sum)
    ods_r, ods_p, ods_f, ods_t = find_best_rpf(t, r, p)
    ois_r, ois_p, ois_f = compute_rpf(ois_sum[None, :])

    k = np.unique(r, return_index=True)[1][::-1]
    rr, pp = r[k], p[k]
    ap = 0.0
    if len(rr) > 1:
        ap = interp1d(rr, pp, bounds_error=False, fill_value=0)(np.linspace(0, 1, 101))
        ap = np.sum(ap) / 100.0

    r50 = np.nan
    _, o = np.unique(pp, return_index=True)
    if len(o) > 1 and np.max(pp[o]) >= 0.5:
        r50 = interp1d(pp[o], rr[o], bounds_error=False, fill_value=np.nan)(0.5)

    bdry = np.array([[ods_t, ods_r, ods_p, ods_f,
                      ois_r.item(), ois_p.item(), ois_f.item(), ap]])

    np.savetxt(os.path.join(eval_dir, "eval_bdry_img.txt"), scores, fmt="%.6f")
    np.savetxt(os.path.join(eval_dir, "eval_bdry_thr.txt"),
               np.stack([t, r, p, f], axis=1), fmt="%.6f")
    np.savetxt(os.path.join(eval_dir, "eval_bdry.txt"), bdry, fmt="%.6f")

    print(f"ODS: {ods_f:.4f}    OIS: {ois_f.item():.4f}")

    if cleanup:
        for f in glob.glob(os.path.join(eval_dir, "*_ev1.txt")):
            os.remove(f)

    return {'ods_f': ods_f, 'ois_f': ois_f.item(), 'ap': ap, 'r50': r50}
