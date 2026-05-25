"""cpu_auction.py — Pure-Python Auction Algorithm for assignment problems.

Matches the GPU auction kernel (auction_kernel.cu) exactly: minimizes
cost + price, uses the same ε-scaling (8→4→2→1→0), same outlier handling,
and same stall detection.

Used as a deterministic CPU fallback when the GPU is not available.
"""
import numpy as np
from scipy.spatial import cKDTree


def build_problem(e1_binary, g_binary, max_dist, outlier_cost):
    """Build one assignment problem — identical to gpu_eval.build_problem."""
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


def _auction_solve_one(problem):
    """Solve one assignment problem, matching the GPU kernel exactly.

    Algorithm (matches auction_kernel.cu):
    - Minimize cost + price (lower val = better)
    - ε-scaling: 8→4→2→1→0
    - Stall detection: if no changes in a round, leftover → outlier
    """
    n1 = problem['n_persons']
    n2 = problem['n_objects']
    edges = problem['edges']
    oc = problem['outlier_cost']

    if n1 == 0:
        return np.array([], dtype=np.int32)

    # Build per-person edge ranges (pe_start)
    pe_start = np.zeros(n1 + 1, dtype=np.int32)
    e_idx = 0
    for p in range(n1):
        pe_start[p] = e_idx
        while e_idx < len(edges) and edges[e_idx, 0] == p:
            e_idx += 1
    pe_start[n1] = e_idx

    # Per-object edge lists (for assignment phase)
    obj_edges = [[] for _ in range(n2)]
    for ei in range(len(edges)):
        obj_edges[edges[ei, 1]].append((edges[ei, 0], edges[ei, 2]))

    # State
    assign = np.full(n1, -1, dtype=np.int32)   # -1 = unassigned, n2 = outlier
    prices = np.zeros(n2, dtype=np.int64)
    owners = np.full(n2, -1, dtype=np.int32)

    # ε-scaling: same as GPU kernel
    eps_values = [8, 4, 2, 1, 0]
    eps_iters = [100, 200, 400, 500, 5000]

    for eidx in range(5):
        eps = eps_values[eidx]
        limit = eps_iters[eidx]

        for _ in range(limit):
            any_change = False

            # ---- BIDDING PHASE ----
            # Track best bid per object: packed (price << 32) | bidder
            # Matches GPU atomicMax on uint64_t exactly.
            best_bid_price = np.full(n2, 0, dtype=np.int64)
            best_bidder = np.full(n2, -1, dtype=np.int32)

            for p in range(n1):
                if assign[p] >= 0:
                    continue  # already assigned

                es = pe_start[p]
                ee = pe_start[p + 1]

                if es == ee:
                    # No edges at all
                    assign[p] = n2
                    continue

                best_val = 10**9
                best_j = -1
                second = 10**9

                for ei in range(es, ee):
                    j = edges[ei, 1]
                    val = int(edges[ei, 2]) + prices[j]
                    if val < best_val:
                        second = best_val
                        best_val = val
                        best_j = j
                    elif val < second:
                        second = val

                # Compare with outlier
                if oc < best_val:
                    second = best_val
                    best_val = oc
                    best_j = n2
                elif oc < second:
                    second = oc

                if best_j == -1:
                    continue
                if second == 10**9:
                    second = best_val + (eps if eps > 0 else 1)

                inc = (second - best_val) + (eps if eps > 0 else 0)
                if inc < 1:
                    inc = 1

                if best_j < n2:
                    bid_v = prices[best_j] + inc
                    # Packed comparison matching GPU atomicMax: (price << 32) | bidder
                    packed = (int(bid_v) << 32) | (int(p) & 0xFFFFFFFF)
                    best_packed = (int(best_bid_price[best_j]) << 32) | (int(best_bidder[best_j]) & 0xFFFFFFFF)
                    if packed > best_packed:
                        best_bid_price[best_j] = bid_v
                        best_bidder[best_j] = p
                else:
                    assign[p] = n2

            # ---- ASSIGNMENT PHASE ----
            for j in range(n2):
                b_price = best_bid_price[j]
                b_bidder = best_bidder[j]
                if b_price > prices[j] and 0 <= b_bidder < n1:
                    prev_owner = owners[j]
                    if 0 <= prev_owner < n1 and assign[prev_owner] == j:
                        assign[prev_owner] = -1
                    if assign[b_bidder] == -1:
                        assign[b_bidder] = j
                        owners[j] = b_bidder
                        prices[j] = b_price
                        any_change = True

            # Check remaining
            remaining = int((assign < 0).sum())
            if remaining == 0:
                break  # all assigned

            # Stall detection
            if not any_change:
                assign[assign < 0] = n2
                break

    # Final: any unassigned → outlier
    assign[assign < 0] = n2
    return assign


def solve_one(problem):
    """Solve one assignment problem. Returns assignment of length n1."""
    return _auction_solve_one(problem)


def solve_batch(problems):
    """Solve a batch of assignment problems on CPU."""
    return [solve_one(p) for p in problems]
