"""gpu_auction.py — ctypes wrapper for GPU Auction Algorithm."""
import ctypes
import numpy as np
import os
import subprocess

_lib = None

def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    lib_path = os.path.join(os.path.dirname(__file__), 'auction_cuda.so')
    if not os.path.exists(lib_path):
        raise RuntimeError("auction_cuda.so not found. Run 'make -C gpu_eval' first.")
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


def _get_gpu_free_memory():
    """Query available GPU memory in bytes via nvidia-smi.

    Returns (free_bytes, total_bytes) or (None, None) on failure.
    """
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
    """Estimate GPU memory (bytes) needed for a chunk of problems.

    Mirrors the cudaMalloc calls in auction_kernel.cu:auction_solve_batch.
    """
    P = len(problems)
    total_persons = sum(p['n_persons'] for p in problems)
    total_objects = sum(p['n_objects'] for p in problems)
    total_edges = sum(len(p['edges']) for p in problems)

    mem = 0
    # Per-problem arrays (int32 = 4B)
    mem += P * 4 * 4            # d_np, d_no, d_oc, (P*4)
    mem += (P + 1) * 4 * 5      # d_es, d_po, d_oo, d_peo (+1 for sentinel)
    # Edge data (int32)
    mem += total_edges * 4 * 3  # d_ep, d_eo, d_ec
    # Person arrays (int32)
    mem += total_persons * 4    # d_as  (assignment output)
    mem += (total_persons + P) * 4  # d_pes (pe_start)
    # Object arrays
    mem += total_objects * 4    # d_pr  (prices, int32)
    mem += total_objects * 8    # d_pk  (packed bids, uint64)
    mem += total_objects * 4    # d_own (owners, int32)
    return mem

def _build_pe_start(edge_person, n_persons_list, edge_starts):
    """Build flat person_edge_start array and per-problem offsets."""
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
        # Vectorized: bincount + cumsum replaces O(n) Python loop
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
    """Partition problems into memory-safe chunks using dynamic sizing.

    Uses nvidia-smi to query available GPU memory and greedily packs
    problems into chunks that each fit within ~60% of free memory.
    Falls back to 128 problems per chunk when query fails.
    """
    free_mem, _ = _get_gpu_free_memory()
    if free_mem is None:
        # fallback: 128 per chunk (was 32 — too conservative, caused
        # 16x overhead on BSDS500 single-image eval)
        chunk_sz = min(128, len(problems))
        return [problems[i:i + chunk_sz] for i in range(0, len(problems), chunk_sz)]

    # Use 60% of free memory per chunk — safe margin avoids OOM from
    # CUDA context overhead and concurrent GPU activity.
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
        if not chunk:  # single problem over budget — force it anyway
            chunk = [problems[i]]
            i += 1
        chunks.append(chunk)
    return chunks


def _solve_chunk(problems, lib, verbose):
    """Solve a chunk of problems on GPU (fits in device memory)."""
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


def batch_solve(problems, verbose=False):
    """Solve batch of assignment problems on GPU.

    Dynamically chunks problems based on available GPU memory to
    minimise kernel launch overhead while avoiding OOM.
    """
    lib = _get_lib()
    if not problems:
        return []

    chunks = _build_chunks(problems)
    results = []
    for chunk in chunks:
        chunk_results = _solve_chunk(chunk, lib, verbose)
        results.extend(chunk_results)
    return results
