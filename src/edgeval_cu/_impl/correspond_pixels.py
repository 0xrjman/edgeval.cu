"""
Edge pixel matching via CSA (Cost Scaling Algorithm).
See https://github.com/davidstutz/extended-berkeley-segmentation-benchmark
"""

import numpy as np
from collections import namedtuple
from ctypes import *
from scipy.spatial import cKDTree
import os

# Locate the compiled CSA solver relative to this package
_cxx_dir = os.path.join(os.path.dirname(__file__), '..', 'cxx', 'lib')
_solver_path = os.path.join(_cxx_dir, 'solve_csa.so')
solver = cdll.LoadLibrary(_solver_path)
c_int_pointer = POINTER(c_int32)


def match_edge_maps(bmap1, bmap2, max_dist, outlier_cost):
    """CXX-like implementation of edge map matching."""
    Edge = namedtuple("Edge", ("i", "j", "w"))
    degree = 6
    multiplier = 100

    assert bmap1.shape == bmap2.shape
    assert max_dist >= 0
    assert outlier_cost > max_dist

    height, width = bmap1.shape
    m1, m2 = np.zeros_like(bmap1, dtype=np.int32), np.zeros_like(bmap2, dtype=np.int32)
    match1, match2 = np.ones((*bmap1.shape, 2), dtype=int) * -1, np.ones((*bmap2.shape, 2), dtype=int) * -1

    r = int(np.ceil(max_dist))

    matchable1, matchable2 = np.zeros_like(bmap1, dtype=bool), np.zeros_like(bmap2, dtype=bool)
    for y1 in range(height):
        for x1 in range(width):
            if bmap1[y1, x1] == 0:
                continue
            for v in range(-r, r + 1):
                for u in range(-r, r + 1):
                    d2 = u * u + v * v
                    if d2 > max_dist * max_dist:
                        continue
                    x2 = x1 + u
                    y2 = y1 + v
                    if x2 < 0 or x2 >= width:
                        continue
                    if y2 < 0 or y2 >= height:
                        continue
                    if bmap2[y2, x2] == 0:
                        continue
                    matchable1[y1, x1] = True
                    matchable2[y2, x2] = True

    n1, n2 = 0, 0
    node_to_pix1, node_to_pix2 = [], []
    pix_to_node1, pix_to_node2 = np.zeros_like(bmap1, dtype=int), np.zeros_like(bmap2, dtype=int)
    for x in range(width):
        for y in range(height):
            pix_to_node1[y, x] = -1
            pix_to_node2[y, x] = -1
            if matchable1[y, x]:
                pix_to_node1[y, x] = n1
                node_to_pix1.append([x, y])
                n1 += 1
            if matchable2[y, x]:
                pix_to_node2[y, x] = n2
                node_to_pix2.append([x, y])
                n2 += 1

    edges = []
    for x1 in range(width):
        for y1 in range(height):
            if matchable1[y1, x1] == 0:
                continue
            for u in range(-r, r + 1):
                for v in range(-r, r + 1):
                    d2 = u * u + v * v
                    if d2 > max_dist * max_dist:
                        continue
                    x2 = x1 + u
                    y2 = y1 + v
                    if x2 < 0 or x2 >= width:
                        continue
                    if y2 < 0 or y2 >= height:
                        continue
                    if bmap2[y2, x2] == 0:
                        continue
                    edges.append(Edge(pix_to_node1[y1, x1], pix_to_node2[y2, x2], np.sqrt(d2).item()))

    n = n1 + n2
    n_min, n_max = min(n1, n2), max(n1, n2)
    d1 = max(0, min(degree, n1 - 1))
    d2 = max(0, min(degree, n2 - 1))
    d3 = min(degree, min(n1, n2))
    dmax = max([d1, d2, 3])
    assert n1 == 0 or 0 <= d1 < n1
    assert n2 == 0 or 0 <= d2 < n2

    m = len(edges) + d1 * n1 + d2 * n2 + d3 * n_max + n
    if m == 0:
        return m1, m2, 0

    ow = int(np.ceil(outlier_cost * multiplier).item())
    outliers_buffer = [0] * dmax
    c_outliers = (c_int32 * len(outliers_buffer))(*outliers_buffer)
    igraph = np.zeros((m, 3), dtype=np.int32)
    count = 0
    for a in range(len(edges)):
        igraph[count, 0] = edges[a].i
        igraph[count, 1] = edges[a].j
        igraph[count, 2] = int(np.rint(edges[a].w * multiplier).item())
        count += 1

    for i in range(n1):
        solver.kOfN(d1, n1 - 1, c_outliers)
        outliers = np.array(c_outliers)
        for a in range(d1):
            j = outliers[a]
            if j >= i:
                j += 1
            assert i != j and 0 <= j < n1
            igraph[count, 0] = i
            igraph[count, 1] = n2 + j
            igraph[count, 2] = ow
            count += 1

    for j in range(n2):
        solver.kOfN(d2, n2 - 1, c_outliers)
        outliers = np.array(c_outliers)
        for a in range(d2):
            i = outliers[a]
            if i >= j:
                i += 1
            assert i != j and 0 <= i < n2
            igraph[count, 0] = n1 + i
            igraph[count, 1] = j
            igraph[count, 2] = ow
            count += 1

    for i in range(n_max):
        solver.kOfN(d3, n_min, c_outliers)
        outliers = np.array(c_outliers)
        for a in range(d3):
            j = outliers[a]
            assert 0 <= j < n_min
            if n1 < n2:
                igraph[count, 0] = n1 + i
                igraph[count, 1] = n2 + j
            else:
                igraph[count, 0] = n1 + j
                igraph[count, 1] = n2 + i
            igraph[count, 2] = ow
            count += 1

    for i in range(n1):
        igraph[count, 0] = i
        igraph[count, 1] = n2 + i
        igraph[count, 2] = ow * multiplier
        count += 1
    for i in range(n2):
        igraph[count, 0] = n1 + i
        igraph[count, 1] = i
        igraph[count, 2] = ow * multiplier
        count += 1
    assert count == m

    for i in range(m):
        igraph[i, 0] += 1
        igraph[i, 1] += 1 + n

    ograph = [0] * (n * 3)
    c_ograph = (c_int32 * len(ograph))(*ograph)
    igraph_list = list(igraph.flatten())
    c_igraph = (c_int32 * len(igraph_list))(*igraph_list)
    solver.solve(n, m, c_igraph, c_ograph)
    ograph = np.array(c_ograph).reshape(n, 3)

    overlay_count = 0
    for a in range(n):
        i, j, c = ograph[a, :]
        if c == ow * multiplier:
            overlay_count += 1
        if i >= n1 or j >= n2:
            continue
        pix1, pix2 = node_to_pix1[i], node_to_pix2[j]
    if overlay_count > 5:
        print(f"WARNING: The match includes {overlay_count} outlier(s) from overlay.")

    for a in range(n):
        i, j = ograph[a, :2]
        if i >= n1 or j >= n2:
            continue
        pix1, pix2 = node_to_pix1[i], node_to_pix2[j]
        match1[pix1[1], pix1[0], :] = [pix2[1], pix2[0]]
        match2[pix2[1], pix2[0], :] = [pix1[1], pix1[0]]
    for x in range(width):
        for y in range(height):
            if bmap1[y, x] != 0:
                if (match1[y, x, :] != np.array([-1, -1], dtype=np.int32)).all():
                    m1[y, x] = match1[y, x, 1] * height + match1[y, x, 0] + 1
            if bmap2[y, x] != 0:
                if (match2[y, x, :] != np.array([-1, -1], dtype=np.int32)).all():
                    m2[y, x] = match2[y, x, 1] * height + match2[y, x, 0] + 1

    cost = 0
    for x in range(width):
        for y in range(height):
            if bmap1[y, x] != 0:
                if (match1[y, x, :] == np.array([-1, -1], dtype=np.int32)).all():
                    cost += outlier_cost
                else:
                    dx = x - match1[y, x, 1]
                    dy = y - match1[y, x, 0]
                    cost += 0.5 * np.sqrt(dx * dx + dy * dy).item()
            if bmap2[y, x] != 0:
                if (match2[y, x, :] == np.array([-1, -1], dtype=np.int32)).all():
                    cost += outlier_cost
                else:
                    dx = x - match2[y, x, 1]
                    dy = y - match2[y, x, 0]
                    cost += 0.5 * np.sqrt(dx * dx + dy * dy).item()
    return m1, m2, cost


def fast_match_edge_maps(bmap1, bmap2, max_dist, outlier_cost, need_cost=False):
    """NumPy-accelerated edge map matching using KDTree."""
    degree = 6
    multiplier = 100

    assert bmap1.shape == bmap2.shape
    assert max_dist >= 0
    assert outlier_cost > max_dist

    height, width = bmap1.shape
    m1, m2 = np.zeros_like(bmap1, dtype=np.int32), np.zeros_like(bmap2, dtype=np.int32)
    match1, match2 = -np.ones((*bmap1.shape, 2), dtype=int), -np.ones((*bmap2.shape, 2), dtype=int)

    bmap1_points = np.stack(np.where(bmap1), axis=0).T
    bmap1_tree = cKDTree(bmap1_points)
    bmap2_points = np.stack(np.where(bmap2), axis=0).T
    bmap2_tree = cKDTree(bmap2_points)
    cnt_1 = bmap1_tree.query_ball_tree(bmap2_tree, r=max_dist)
    cnt_2 = bmap2_tree.query_ball_tree(bmap1_tree, r=max_dist)
    cnt_1 = np.array([len(c) for c in cnt_1])
    cnt_2 = np.array([len(c) for c in cnt_2])
    matchable1, matchable2 = np.zeros_like(bmap1, dtype=np.bool), np.zeros_like(bmap2, dtype=np.bool)
    points_1, points_2 = bmap1_points[cnt_1 > 0], bmap2_points[cnt_2 > 0]
    if len(points_1):
        matchable1[points_1[:, 0], points_1[:, 1]] = True
    if len_points_2:
        matchable2[points_2[:, 0], points_2[:, 1]] = True

    pix_to_node1 = np.cumsum(matchable1.T).reshape(width, height).T - 1
    pix_to_node2 = np.cumsum(matchable2.T).reshape(width, height).T - 1
    n1, n2 = int(pix_to_node1[-1, -1]) + 1, int(pix_to_node2[-1, -1]) + 1
    pix_to_node1[np.logical_not(matchable1)] = -1
    pix_to_node2[np.logical_not(matchable2)] = -1

    matchable1_points = np.stack(np.where(matchable1), axis=0).T
    matchable1_tree = cKDTree(matchable1_points)
    pairs = matchable1_tree.sparse_distance_matrix(bmap2_tree, max_dist, output_type='coo_matrix')
    if len(pairs.data) > 0:
        distance = np.rint(pairs.data * multiplier).astype(np.uint32)
        p_i, p_j = matchable1_points[pairs.row], bmap2_points[pairs.col]
        p_i, p_j = pix_to_node1[p_i[:, 0], p_i[:, 1]], pix_to_node2[p_j[:, 0], p_j[:, 1]]
        edges = np.stack([p_i, p_j, distance], axis=0).T
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    else:
        edges = np.zeros((0, 3), dtype=np.int32)
    n = n1 + n2
    n_min, n_max = min(n1, n2), max(n1, n2)
    d1 = max(0, min(degree, n1 - 1))
    d2 = max(0, min(degree, n2 - 1))
    d3 = min(degree, min(n1, n2))

    m = len(edges) + d1 * n1 + d2 * n2 + d3 * n_max + n
    if m == 0:
        return m1, m2, 0

    ow = int(np.ceil(outlier_cost * multiplier).item())
    igraph = np.zeros((m, 3), dtype=np.int32)
    count = len(edges)
    if count:
        igraph[:count, ...] = edges

    count_end = count + n1 * d1
    indices = np.tile(np.arange(n1)[:, None], (d1,)).flatten()
    outliers = np.zeros((n1 * d1,), dtype=np.int32)
    for i in range(n1):
        solver.kOfN(d1, n1 - 1, outliers[i * d1:(i + 1) * d1].ctypes.data_as(c_int_pointer))
    outliers[outliers >= indices] += 1
    igraph[count:count_end, 0] = indices
    igraph[count:count_end, 1] = n2 + outliers
    igraph[count:count_end, 2] = ow
    count = count_end

    count_end = count + n2 * d2
    indices = np.tile(np.arange(n2)[:, None], (d2,)).flatten().astype(np.int32)
    outliers = np.zeros((n2 * d2,), dtype=np.int32)
    for i in range(n2):
        solver.kOfN(d2, n2 - 1, outliers[i * d2:(i + 1) * d2].ctypes.data_as(c_int_pointer))
    outliers[outliers >= indices] += 1
    igraph[count:count_end, 0] = n1 + outliers
    igraph[count:count_end, 1] = indices
    igraph[count:count_end, 2] = ow
    count = count_end

    count_end = count + n_max * d3
    indices = np.tile(np.arange(n_max)[:, None], (d3,)).flatten().astype(np.int32)
    outliers = np.zeros((n_max * d3,), dtype=np.int32)
    for i in range(n_max):
        solver.kOfN(d3, n_min, outliers[i * d3:(i + 1) * d3].ctypes.data_as(c_int_pointer))
    igraph[count:count_end, 0] = n1 + (indices if n1 < n2 else outliers)
    igraph[count:count_end, 1] = n2 + (outliers if n1 < n2 else indices)
    igraph[count:count_end, 2] = ow
    count = count_end

    indices = np.arange(n1).astype(np.int32)
    igraph[count:count + n1, 0] = indices
    igraph[count:count + n1, 1] = n2 + indices
    igraph[count:count + n1, 2] = ow * multiplier
    count += n1
    indices = np.arange(n2).astype(np.int32)
    igraph[count:count + n2, 0] = n1 + indices
    igraph[count:count + n2, 1] = indices
    igraph[count:count + n2, 2] = ow * multiplier
    count += n2

    igraph[:, 0] += 1
    igraph[:, 1] += 1 + n

    ograph = np.zeros((n, 3), dtype=np.int32)
    solver.solve(n, m, igraph.ctypes.data_as(c_int_pointer), ograph.ctypes.data_as(c_int_pointer))

    overlay_count = (ograph[:, 2] == (ow * multiplier)).sum()
    if overlay_count > 5:
        print(f"WARNING: The match includes {overlay_count} outlier(s) from overlay.")

    index_i, index_j = ograph[:n, 0], ograph[:n, 1]
    mask = np.logical_and(index_i < n1, index_j < n2)
    index_i, index_j = index_i[mask], index_j[mask]
    node_to_pix1_arr = np.array(node_to_pix1)
    node_to_pix2_arr = np.array(node_to_pix2)
    pix1, pix2 = node_to_pix1_arr[index_i], node_to_pix2_arr[index_j]
    match1[pix1[:, 1], pix1[:, 0], 0] = pix2[:, 1]
    match1[pix1[:, 1], pix1[:, 0], 1] = pix2[:, 0]
    match2[pix2[:, 1], pix2[:, 0], 0] = pix1[:, 1]
    match2[pix2[:, 1], pix2[:, 0], 1] = pix1[:, 0]
    m1_mask = np.logical_and.reduce((bmap1, match1[..., 0] != -1, match1[..., 1] != -1), axis=0)
    m2_mask = np.logical_and.reduce((bmap2, match2[..., 0] != -1, match2[..., 1] != -1), axis=0)
    m1[m1_mask] = match1[m1_mask, 1] * height + match1[m1_mask, 0] + 1
    m2[m2_mask] = match2[m2_mask, 1] * height + match2[m2_mask, 0] + 1

    cost = 0
    if need_cost:
        m1_unmatched = np.logical_and(bmap1, np.logical_and(match1[..., 0] == -1, match1[..., 1] == -1))
        m2_unmatched = np.logical_and(bmap2, np.logical_and(match2[..., 0] == -1, match2[..., 1] == -1))
        cost += (m1_unmatched.sum() + m2_unmatched.sum()) * outlier_cost
        m1_matched = np.logical_and(bmap1, np.logical_not(m1_unmatched))
        m2_matched = np.logical_and(bmap2, np.logical_not(m2_unmatched))
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        dx1 = xx[m1_matched] - match1[m1_matched, 1]
        dy1 = yy[m1_matched] - match1[m1_matched, 0]
        dx2 = xx[m2_matched] - match2[m2_matched, 1]
        dy2 = yy[m2_matched] - match2[m2_matched, 0]
        cost += 0.5 * np.sqrt(dx1 ** 2 + dy1 ** 2).sum()
        cost += 0.5 * np.sqrt(dx2 ** 2 + dy2 ** 2).sum()
    return m1, m2, cost


def correspond_pixels(bmap1, bmap2, max_dist=0.0075, outliner_cost=100):
    """Correspond pixels between two binary edge maps."""
    assert bmap1.shape == bmap2.shape
    assert max_dist >= 0
    assert outliner_cost > 1

    rows, cols = bmap1.shape
    idiag = np.sqrt(rows * rows + cols * cols)
    max_dist *= idiag
    oc = outliner_cost * max_dist

    match1, match2, cost = fast_match_edge_maps(bmap1, bmap2, max_dist, oc, False)
    return match1, match2, cost, oc
