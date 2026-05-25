"""gpu_eval.py — GPU-accelerated edge evaluation pipeline.

Fused CUDA kernel for graph construction + GPU sort + GPU Auction batch solve.
Single-image 99-threshold evaluation in <1s.

Two modes:
  - 'simple': Simple bipartite graph + GPU Auction (fast, ~0.8s/img, ΔODS≈+0.002)
  - 'extended': CSA-compatible extended graph with kOfN (slower, precise)
"""
import numpy as np
import torch
import torch.nn.functional as F
import ctypes
from scipy.io import loadmat
from scipy.interpolate import interp1d
from .._impl.bwmorph_thin import G123_LUT, G123P_LUT
from .._impl.edges_eval_dir import compute_rpf, find_best_rpf, eps as EVAL_EPS
from .gpu_auction import batch_solve, build_extended_problem, reseed_kofn
import os
import glob
from tqdm import tqdm


# ── GPU thinning LUTs (loaded once) ──────────────────────────────────

_G123_T = None
_G123P_T = None
_THIN_KERNEL = None

def _get_thin_luts():
    global _G123_T, _G123P_T, _THIN_KERNEL
    if _G123_T is None:
        _G123_T = torch.tensor(G123_LUT.astype(np.float32), device='cuda')
        _G123P_T = torch.tensor(G123P_LUT.astype(np.float32), device='cuda')
        _THIN_KERNEL = torch.tensor(
            [[8, 4, 2], [16, 0, 1], [32, 64, 128]],
            dtype=torch.float32, device='cuda'
        ).view(1, 1, 3, 3)
    return _G123_T, _G123P_T, _THIN_KERNEL


def _gpu_thin_batch(masks_t, max_iter=20):
    """Batch GPU morphological thinning for (B, H, W) bool tensor on GPU."""
    g123_t, g123p_t, kernel = _get_thin_luts()
    B = masks_t.shape[0]
    skels = masks_t.float().unsqueeze(1)  # (B, 1, H, W)
    converged = torch.zeros(B, dtype=torch.bool, device='cuda')

    for _ in range(max_iter):
        before = skels.sum(dim=(1, 2, 3))
        for lut_t in [g123_t, g123p_t]:
            N = F.conv2d(skels, kernel, padding=1).long().squeeze(1)
            N = N.clamp(0, 255)
            D = lut_t[N]
            skels = skels * (1 - D).unsqueeze(1)
        after = skels.sum(dim=(1, 2, 3))
        converged = converged | (before == after)
        if converged.all():
            break

    return skels.squeeze(1).bool()  # (B, H, W)


# ── Fused CUDA edge builder ─────────────────────────────────────────

_lib_edge = None

def _get_edge_lib():
    global _lib_edge
    if _lib_edge is not None:
        return _lib_edge
    lib_path = os.path.join(os.path.dirname(__file__), 'edge_builder.so')
    if not os.path.exists(lib_path):
        raise RuntimeError("edge_builder.so not found. Run 'make -C gpu_eval' first.")
    _lib_edge = ctypes.CDLL(lib_path)
    _lib_edge.launch_build_edges.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int,
    ]
    _lib_edge.launch_build_edges.restype = ctypes.c_int
    return _lib_edge


def _build_edges_fused(pred_t, gt_all_t, max_dist_px, scale=100):
    """Fused CUDA kernel: cdist + mask + nonzero in one launch. Returns GPU tensor."""
    lib = _get_edge_lib()
    n_pred, n_gt_total = pred_t.shape[0], gt_all_t.shape[0]
    max_edges = max(1, int(n_pred * n_gt_total * 0.01))
    count_gpu = torch.zeros(1, dtype=torch.int32, device='cuda')
    edges_gpu = torch.zeros((max_edges, 3), dtype=torch.int32, device='cuda')
    ret = lib.launch_build_edges(
        pred_t.data_ptr(), gt_all_t.data_ptr(),
        n_pred, n_gt_total, max_dist_px, scale,
        count_gpu.data_ptr(), edges_gpu.data_ptr(), max_edges,
    )
    if ret != 0:
        raise RuntimeError(f"launch_build_edges failed: {ret}")
    count = count_gpu.item()
    if count == 0:
        return None
    if count > max_edges:
        count = max_edges
    return edges_gpu[:count]



# ── Main evaluation ──────────────────────────────────────────────────

def gpu_edges_eval_img(edge_prob, gt_path, thrs=99, max_dist=0.0075,
                       thin=True, mode='simple'):
    """GPU-accelerated edge evaluation for one image.

    Args:
        edge_prob: 2D float32 edge probability map, values in [0, 1].
        gt_path: path to .mat ground truth file.
        thrs: number of thresholds (int) or array of threshold values.
        max_dist: max matching distance as fraction of image diagonal.
        thin: apply morphological thinning.
        mode: 'simple' (fast, ~0.8s) or 'extended' (CSA-compatible, slower).

    Returns:
        info: (k, 5) array [threshold, matched_gt, total_gt, matched_pred, total_pred].
    """
    H, W = edge_prob.shape
    idiag = np.sqrt(H * H + W * W)
    max_dist_px = float(max_dist * idiag)
    oc = 100.0 * max_dist_px
    oc_int = int(np.ceil(oc * 100))

    # Load GT
    mat = loadmat(gt_path)
    try:
        gt_raw = [g.item()[1] for g in mat["groundTruth"][0]]
    except Exception:
        gt_raw = [g.item()[0] for g in mat["groundTruth"][0]]

    # Thresholds
    if isinstance(thrs, int):
        k = thrs
        thrs_vals = np.linspace(1 / (k + 1), 1 - 1 / (k + 1), k)
    else:
        k = len(thrs)
        thrs_vals = np.asarray(thrs)

    # Pre-build pred coords per threshold (batched GPU thinning)
    pred_tensors = []
    pred_binary = []

    # Stack all binary masks, thin on GPU in one batch
    binary_masks = []
    for t in thrs_vals:
        e1 = edge_prob >= max(EVAL_EPS, t)
        binary_masks.append(e1.astype(np.uint8))
    masks_t = torch.from_numpy(np.stack(binary_masks)).cuda()

    if thin:
        masks_t = _gpu_thin_batch(masks_t)

    # Extract per-threshold results
    for k_idx in range(k):
        e1 = masks_t[k_idx].cpu().numpy()
        pred_binary.append(e1)
        coords = np.column_stack(np.where(e1)).astype(np.float32)
        if len(coords) > 0:
            pred_tensors.append(torch.from_numpy(coords).cuda())
        else:
            pred_tensors.append(None)

    # Pre-build GT data
    gt_list = []
    for g in gt_raw:
        coords = np.column_stack(np.where(g)).astype(np.float32)
        gt_list.append(coords)

    gt_all_np = np.concatenate(gt_list, axis=0) if gt_list else np.zeros((0, 2), dtype=np.float32)
    gt_all_t = torch.from_numpy(gt_all_np).cuda()
    gt_n = np.array([len(g) for g in gt_list], dtype=np.int32)
    gt_offsets = np.array([0] + list(np.cumsum(gt_n)), dtype=np.int32)
    gt_offsets_t = torch.from_numpy(gt_offsets).cuda()
    n_gt_pixels = int(gt_all_np.shape[0])  # total GT pixels
    n_annos = len(gt_n)

    # Total GT pixel count
    all_g_sum = np.zeros((H, W), dtype=np.int64)
    for g in gt_raw:
        all_g_sum += g.astype(np.int64)
    total_gt_val = int(all_g_sum.sum())

    # Build all problems
    problems = []
    prob_meta = []  # (k_idx, g_idx)

    if mode == 'simple':
        for k_idx, pred_t in enumerate(pred_tensors):
            if pred_t is None:
                continue
            n_pred = pred_t.shape[0]

            # Fused kernel: build edges
            edges_t = _build_edges_fused(pred_t, gt_all_t, max_dist_px)
            if edges_t is None:
                continue

            # GPU split: sort by (annotator, person, gt_global)
            # Compound key = anno * SP + person * SG + gt_global
            SP = n_pred * n_gt_pixels + 1  # scale for annotator
            SG = n_gt_pixels                # scale for person
            ai = torch.bucketize(edges_t[:, 1].contiguous().to(torch.int32),
                                 gt_offsets_t, right=True) - 1
            ai = ai.clamp(0, n_annos - 1)
            compound = (ai.to(torch.int64) * SP +
                        edges_t[:, 0].to(torch.int64) * SG +
                        edges_t[:, 1].to(torch.int64))
            _, indices = torch.sort(compound)
            edges_sorted = edges_t[indices]
            ai_sorted = ai[indices]

            # Find annotator boundaries
            boundaries = torch.cat([
                torch.tensor([0], device='cuda'),
                torch.where(ai_sorted[1:] != ai_sorted[:-1])[0] + 1,
                torch.tensor([len(ai_sorted)], device='cuda')
            ])

            # Download once and split by boundaries
            edges_np = edges_sorted.cpu().numpy()
            bounds_np = boundaries.cpu().numpy()
            for b in range(len(bounds_np) - 1):
                lo, hi = bounds_np[b], bounds_np[b + 1]
                if lo == hi:
                    continue
                g_idx = int(ai_sorted[lo].item())
                eg = edges_np[lo:hi].copy()
                eg[:, 1] -= gt_offsets[g_idx]
                problems.append({
                    'n_persons': n_pred, 'n_objects': int(gt_n[g_idx]),
                    'edges': eg, 'outlier_cost': oc_int,
                })
                prob_meta.append((k_idx, g_idx))

    elif mode == 'extended':
        from scipy.spatial import cKDTree
        KOFN_SEED = 42
        for k_idx, pred_t in enumerate(pred_tensors):
            if pred_t is None:
                continue
            pred_np = pred_t.cpu().numpy()
            n_pred = len(pred_np)
            for g_idx, gt_coords in enumerate(gt_list):
                n_gt = len(gt_coords)
                if n_gt == 0:
                    continue
                # Matchable filtering + real edges (CPU cKDTree)
                pred_tree = cKDTree(pred_np)
                gt_tree = cKDTree(gt_coords)
                pairs = pred_tree.sparse_distance_matrix(gt_tree, max_dist_px, output_type='coo_matrix')
                if len(pairs.data) == 0:
                    continue
                edges = np.zeros((len(pairs.data), 3), dtype=np.int32)
                edges[:, 0] = pairs.row.astype(np.int32)
                edges[:, 1] = pairs.col.astype(np.int32)
                edges[:, 2] = np.rint(pairs.data * 100).astype(np.int32)
                edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
                reseed_kofn(KOFN_SEED)
                ext = build_extended_problem(n_pred, n_gt, edges, oc_int)
                if ext is not None:
                    problems.append(ext)
                    prob_meta.append((k_idx, g_idx))

    # GPU batch solve
    assignments = batch_solve(problems) if problems else []

    # Aggregate results
    cnt_sum = np.zeros((k, 4), dtype=np.int64)
    cnt_sum[:, 1] = total_gt_val
    for k_idx, e1 in enumerate(pred_binary):
        cnt_sum[k_idx, 3] = int(e1.sum())

    matched_pred_sets = {}
    for p_idx, (k_idx, g_idx) in enumerate(prob_meta):
        assign = assignments[p_idx]
        n_objects = problems[p_idx]['n_objects']
        gt_matched = {int(a) for a in assign if 0 <= int(a) < n_objects}
        cnt_sum[k_idx, 0] += len(gt_matched)
        if k_idx not in matched_pred_sets:
            matched_pred_sets[k_idx] = set()
        for pi, a in enumerate(assign):
            if 0 <= int(a) < n_objects:
                matched_pred_sets[k_idx].add(pi)

    for k_idx in range(k):
        cnt_sum[k_idx, 2] = len(matched_pred_sets.get(k_idx, set()))

    info = np.concatenate([thrs_vals[:, None], cnt_sum], axis=1)
    return info, None


# ── Directory-level evaluation ───────────────────────────────────────

def gpu_edges_eval_dir(res_dir, gt_dir, thrs=99, max_dist=0.0075,
                       thin=True, cleanup=0, mode='simple'):
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
                                           max_dist=max_dist, thin=thin,
                                           mode=mode)
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

    mask = r >= 1e-3
    rr_full, pp_full = r[mask], p[mask]
    ap = 0.0
    if len(rr_full) > 1:
        kk = np.unique(rr_full, return_index=True)[1][::-1]
        rr_uniq, pp_uniq = rr_full[kk], pp_full[kk]
        ap = interp1d(rr_uniq, pp_uniq, bounds_error=False,
                      fill_value=0)(np.linspace(0, 1, 101))
        ap = np.sum(ap) / 100.0

    r50 = np.nan
    _, o = np.unique(pp_full, return_index=True)
    if len(o) > 1:
        r50 = interp1d(pp_full[o], rr_full[o], bounds_error=False,
                       fill_value=np.nan)(np.maximum(pp_full[o[0]], 0.5))

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
