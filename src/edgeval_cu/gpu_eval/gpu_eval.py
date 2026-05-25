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
from .gpu_auction import batch_solve, build_extended_problem
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
    prob_meta = []  # (k_idx, g_idx, prob_idx, n1_orig, n2_orig)
    for k_idx, e1 in enumerate(all_binary):
        for g_idx, g in enumerate(gt_raw):
            prob = build_problem(e1, g, max_dist_px, oc)
            if prob is None:
                prob_meta.append((k_idx, g_idx, -1, 0, 0))
            else:
                n1_orig = prob['n_persons']
                n2_orig = prob['n_objects']
                ext = build_extended_problem(n1_orig, n2_orig,
                                             prob['edges'], prob['outlier_cost'])
                if ext is None:
                    prob_meta.append((k_idx, g_idx, -1, 0, 0))
                else:
                    prob_idx = len(problems)
                    problems.append(ext)
                    prob_meta.append((k_idx, g_idx, prob_idx,
                                      n1_orig, n2_orig))

    assignments = batch_solve(problems) if problems else []

    cnt_sum_r_p = np.zeros((k, 4), dtype=np.int64)
    matched_pred_union = [set() for _ in range(k)]

    for item_idx, (k_idx, g_idx, prob_idx, n1_orig, n2_orig) in enumerate(prob_meta):
        g = gt_raw[g_idx]
        cnt_sum_r_p[k_idx, 1] += g.sum()
        if prob_idx < 0:
            continue
        assign = assignments[prob_idx]
        # Extended graph: persons 0..n1_orig-1 are real, rest are virtual
        # assign[i] < n2_orig means real-to-real match
        real_assign = assign[:n1_orig]
        real_matches = real_assign[real_assign < n2_orig]
        unique_gt = len(np.unique(real_matches))
        cnt_sum_r_p[k_idx, 0] += unique_gt
        matched_idx = np.where(real_assign < n2_orig)[0]
        matched_pred_union[k_idx].update(matched_idx.tolist())

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

    # AP and R50 — match original edges_eval_plot.py behavior exactly
    mask = r >= 1e-3
    rr_full, pp_full = r[mask], p[mask]
    ap = 0.0
    if len(rr_full) > 1:
        kk = np.unique(rr_full, return_index=True)[1][::-1]
        rr_uniq, pp_uniq = rr_full[kk], pp_full[kk]
        ap = interp1d(rr_uniq, pp_uniq, bounds_error=False, fill_value=0)(np.linspace(0, 1, 101))
        ap = np.sum(ap) / 100.0

    # R50: match edges_eval_plot.py exactly
    r50 = np.nan
    _, o = np.unique(pp_full, return_index=True)
    if len(o) > 1:
        r50 = interp1d(pp_full[o], rr_full[o], bounds_error=False, fill_value=np.nan)(np.maximum(pp_full[o[0]], 0.5))

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
