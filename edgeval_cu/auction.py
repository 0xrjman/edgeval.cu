"""gpu_auction.py — ctypes wrapper for GPU Auction Algorithm."""
import ctypes
import numpy as np
import os
import subprocess
from ctypes import *

_lib = None
_c_int_pointer = POINTER(c_int32)


# ── Vectorized extended graph builder ───────────────────────────────

def _sample_unique(d, n, cnt):
    """Generate (cnt, d) matrix of unique values from [0, n-1] per row."""
    result = np.random.randint(0, int(n), size=(cnt, d), dtype=np.int32)
    while True:
        result.sort(axis=1)
        dup = (result[:, 1:] == result[:, :-1]).any(axis=1)
        if not dup.any():
            break
        result[dup] = np.random.randint(0, int(n), size=(int(dup.sum()), d), dtype=np.int32)
    return result


def _build_extended_edges_fast(n1, n2, real_edges, oc_int):
    """Build extended graph edges using fully vectorized numpy kOfN.

    Replaces the per-call C++ kOfN (ctypes) with numpy batch operations.
    """
    degree = 6
    multiplier = 100

    n = n1 + n2
    n_min, n_max = min(n1, n2), max(n1, n2)
    d1 = max(0, min(degree, int(n1) - 1)) if n1 > 0 else 0
    d2 = max(0, min(degree, int(n2) - 1)) if n2 > 0 else 0
    d3 = min(degree, min(int(n1), int(n2))) if n1 > 0 and n2 > 0 else 0
    ow = int(oc_int)

    total_edges = len(real_edges) + d1 * n1 + d2 * n2 + d3 * n_max + n
    if total_edges == 0:
        return np.empty((0, 3), dtype=np.int32), n

    igraph = np.empty((total_edges, 3), dtype=np.int32)
    count = 0

    # 1. Real edges
    if len(real_edges) > 0:
        igraph[:len(real_edges)] = real_edges
        count += len(real_edges)

    # 2. Real person -> virtual object (skip self)
    if d1 > 0 and n1 > 0:
        choices = _sample_unique(d1, n1 - 1, n1)
        choices = choices + (choices >= np.arange(n1, dtype=np.int32)[:, None]).astype(np.int32)
        n_rows = n1 * d1
        igraph[count:count + n_rows, 0] = np.repeat(np.arange(n1, dtype=np.int32), d1)
        igraph[count:count + n_rows, 1] = n2 + choices.ravel()
        igraph[count:count + n_rows, 2] = ow
        count += n_rows

    # 3. Virtual person -> real object (skip self)
    if d2 > 0 and n2 > 0:
        choices = _sample_unique(d2, n2 - 1, n2)
        choices = choices + (choices >= np.arange(n2, dtype=np.int32)[:, None]).astype(np.int32)
        n_rows = n2 * d2
        igraph[count:count + n_rows, 0] = n1 + choices.ravel()
        igraph[count:count + n_rows, 1] = np.repeat(np.arange(n2, dtype=np.int32), d2)
        igraph[count:count + n_rows, 2] = ow
        count += n_rows

    # 4. Virtual <-> virtual
    if d3 > 0 and n_min > 0:
        choices = _sample_unique(d3, n_min, n_max)
        n_rows = n_max * d3
        if n1 < n2:
            igraph[count:count + n_rows, 0] = n1 + np.repeat(np.arange(n_max, dtype=np.int32), d3)
            igraph[count:count + n_rows, 1] = n2 + choices.ravel()
        else:
            igraph[count:count + n_rows, 0] = n1 + choices.ravel()
            igraph[count:count + n_rows, 1] = n2 + np.repeat(np.arange(n_max, dtype=np.int32), d3)
        igraph[count:count + n_rows, 2] = ow
        count += n_rows

    # 5. Diagonal perfect-match overlay
    oc_mult = ow * multiplier
    diag1 = np.zeros((n1, 3), dtype=np.int32)
    diag1[:, 0] = np.arange(n1, dtype=np.int32)
    diag1[:, 1] = n2 + np.arange(n1, dtype=np.int32)
    diag1[:, 2] = oc_mult
    igraph[count:count + n1] = diag1
    count += n1

    diag2 = np.zeros((n2, 3), dtype=np.int32)
    diag2[:, 0] = n1 + np.arange(n2, dtype=np.int32)
    diag2[:, 1] = np.arange(n2, dtype=np.int32)
    diag2[:, 2] = oc_mult
    igraph[count:count + n2] = diag2
    count += n2

    assert count == total_edges, f"Edge count mismatch: {count} vs {total_edges}"

    igraph = igraph[np.lexsort((igraph[:, 1], igraph[:, 0]))]
    return igraph, n


# ── Extended problem builder (public API) ───────────────────────────

def build_extended_problem(n1, n2, real_edges, oc_int):
    """Build extended assignment problem matching CPU CSA graph structure."""
    edges, n = _build_extended_edges_fast(n1, n2, real_edges, oc_int)
    if len(edges) == 0:
        return None
    return {
        'n_persons': n,
        'n_objects': n,
        'edges': edges,
        'outlier_cost': np.iinfo(np.int32).max // 4,
    }


# ── RNG seeding (compatibility with C++ kOfN interface) ─────────────

def reseed_kofn(seed):
    """Reset the numpy random state for reproducible kOfN sampling.
    
    In the C++ kOfN path, this called reseed_random() via ctypes.
    With the vectorized numpy kOfN, this is equivalent to np.random.seed().
    """
    np.random.seed(int(seed))


# ── GPU library loader ──────────────────────────────────────────────

def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    # Compile CUDA kernels if needed (first import after pip install)
    from ._compile import ensure_compiled
    ensure_compiled()
    lib_path = os.path.join(os.path.dirname(__file__), 'cuda', 'auction_cuda.so')
    if not os.path.exists(lib_path):
        raise RuntimeError(
            "auction_cuda.so not found after compilation. "
            "Check that nvcc is available and CUDA toolkit is installed."
        )
    _lib = ctypes.CDLL(lib_path)
    _lib.auction_solve_batch.restype = ctypes.c_int
    _lib.auction_solve_batch.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ]
    return _lib


# ── Memory query and chunking ───────────────────────────────────────

def _get_gpu_free_memory():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=5
        ).decode().strip()
        free_mb, total_mb = map(int, out.split(','))
        return free_mb * 1024 * 1024, total_mb * 1024 * 1024
    except Exception:
        return None, None


def _estimate_chunk_mem(problems):
    P = len(problems)
    total_persons = sum(p['n_persons'] for p in problems)
    total_objects = sum(p['n_objects'] for p in problems)
    total_edges = sum(len(p['edges']) for p in problems)

    mem = 0
    mem += P * 4 * 4
    mem += (P + 1) * 4 * 5
    mem += total_edges * 4 * 3
    mem += total_persons * 4
    mem += (total_persons + P) * 4
    mem += total_objects * 4
    mem += total_objects * 8
    mem += total_objects * 4
    return mem


def _build_pe_start(edge_person, n_persons_list, edge_starts):
    P = len(n_persons_list)
    pe_start = []
    for p in range(P):
        n1 = n_persons_list[p]
        e_beg = edge_starts[p]
        e_end = edge_starts[p + 1]
        if n1 == 0 or e_beg == e_end:
            pe_start.extend([e_beg] * (n1 + 1))
            continue
        person_ids = edge_person[e_beg:e_end]
        person_counts = np.bincount(person_ids, minlength=n1)
        person_edge_start = np.zeros(n1 + 1, dtype=np.int32)
        person_edge_start[1:] = np.cumsum(person_counts).astype(np.int32)
        pe_start.extend((e_beg + person_edge_start).tolist())
    pe_start = np.array(pe_start, dtype=np.int32)
    pe_offset = np.zeros(P + 1, dtype=np.int32)
    for p in range(P):
        pe_offset[p + 1] = pe_offset[p] + n_persons_list[p] + 1
    return pe_start, pe_offset


def _build_chunks(problems):
    free_mem, _ = _get_gpu_free_memory()
    if free_mem is None:
        chunk_sz = min(128, len(problems))
        return [problems[i:i + chunk_sz] for i in range(0, len(problems), chunk_sz)]

    budget = int(0.6 * free_mem)
    chunks = []
    i = 0
    while i < len(problems):
        chunk = []
        chunk_mem = 0
        while i < len(problems):
            p_mem = _estimate_chunk_mem([problems[i]])
            if chunk_mem + p_mem > budget and len(chunk) > 0:
                break
            chunk.append(problems[i])
            chunk_mem += p_mem
            i += 1
        if not chunk:
            chunk = [problems[i]]
            i += 1
        chunks.append(chunk)
    return chunks


def _solve_chunk(problems, lib, verbose):
    P = len(problems)
    if P == 0:
        return []

    h_n_persons = np.array([p['n_persons'] for p in problems], dtype=np.int32)
    h_n_objects = np.array([p['n_objects'] for p in problems], dtype=np.int32)
    h_outlier_cost = np.array([p['outlier_cost'] for p in problems], dtype=np.int32)

    edge_starts = [0]
    all_ep, all_eo, all_ec = [], [], []
    for p in problems:
        e = p['edges']
        if len(e) > 0:
            all_ep.append(e[:, 0].astype(np.int32))
            all_eo.append(e[:, 1].astype(np.int32))
            all_ec.append(e[:, 2].astype(np.int32))
        edge_starts.append(edge_starts[-1] + len(e))

    h_edge_start = np.array(edge_starts, dtype=np.int32)
    h_edge_person = np.concatenate(all_ep).astype(np.int32) if all_ep else np.array([], dtype=np.int32)
    h_edge_object = np.concatenate(all_eo).astype(np.int32) if all_eo else np.array([], dtype=np.int32)
    h_edge_cost = np.concatenate(all_ec).astype(np.int32) if all_ec else np.array([], dtype=np.int32)

    total_persons = int(h_n_persons.sum())
    h_pe_start, h_pe_offset = _build_pe_start(h_edge_person, h_n_persons.tolist(), edge_starts)
    h_assignment = np.full(total_persons, -1, dtype=np.int32)

    ret = lib.auction_solve_batch(
        ctypes.c_int(P),
        h_n_persons.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_n_objects.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_edge_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_edge_person.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_edge_object.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_edge_cost.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_outlier_cost.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_assignment.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_pe_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        h_pe_offset.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(1 if verbose else 0),
    )

    if ret != 0:
        raise RuntimeError(f"auction_solve_batch returned {ret}")

    results = []
    offset = 0
    for p in range(P):
        n1 = h_n_persons[p]
        results.append(h_assignment[offset:offset + n1].copy())
        offset += n1
    return results


# ── Public API ──────────────────────────────────────────────────────

def batch_solve(problems, verbose=False):
    lib = _get_lib()
    if not problems:
        return []
    chunks = _build_chunks(problems)
    results = []
    for chunk in chunks:
        chunk_results = _solve_chunk(chunk, lib, verbose)
        results.extend(chunk_results)
    return results
