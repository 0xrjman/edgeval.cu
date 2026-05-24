/*
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2026 0xrjman & Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file auction_kernel.cu
 * @brief GPU-parallel Auction Algorithm for batched assignment problem solving.
 *
 * Implements Bertsekas'"'"' forward auction algorithm with epsilon-scaling
 * to solve multiple independent assignment (matching) problems in parallel
 * on GPU. Each CUDA block handles one problem.
 *
 * Applicable to edge detection evaluation (correspondence matching between
 * predicted edge pixels and ground-truth edge pixels) and any other
 * assignment problem formulation.
 *
 * Algorithm summary:
 *   - Persons (predicted edge pixels) bid on objects (GT edge pixels)
 *     based on cost (= distance^2) + current price.
 *   - Highest bidder wins each object via atomicMax on a 64-bit packed
 *     (bid_price << 32 | bidder_id) value -- race-free auction mechanics.
 *   - epsilon-scaling (8 -> 4 -> 2 -> 1) plus 5000-iteration refinement at epsilon=1 guarantees optimality for integer-cost assignment
 *     for integer costs at epsilon = 0.
 *
 * Reference:
 *   D. P. Bertsekas, "A distributed algorithm for the assignment problem,"
 *   1979.  GPU adaptation with epsilon-scaling and race-free atomic packing.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */
/** Maximum threads per block.  Must match GPU warp size constraints. */
#define MAX_THREADS 256

/** Maximum bidding iterations per epsilon level. */
#define ITERS_EPS8   100
#define ITERS_EPS4   200
#define ITERS_EPS2   400
#define ITERS_EPS1  5000
#define ITERS_EPS0     0

/**
 * @def CUDA_CHECK(ans)
 * @brief Macro that checks every CUDA API call and aborts on error.
 *
 * Based on the pattern used in NVIDIA CUDA Samples (helper_cuda.h).
 * Wraps both allocation and memory-transfer calls so no return value
 * is left unchecked.
 */
#define CUDA_CHECK(ans) do {                                   \
    cudaError_t _err = (ans);                                  \
    if (_err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s:%d -- %s\n",         \
                __FILE__, __LINE__, cudaGetErrorString(_err));  \
        return 1;                                              \
    }                                                          \
} while (0)

/* ------------------------------------------------------------------ */
/*  Device helpers -- packed 64-bit atomic bid                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Pack a (price, bidder_id) pair into a single uint64_t.
 *
 * The upper 32 bits store the bid price (>= 0), the lower 32 bits store
 * the bidder index.  Used with atomicMax so that both fields are updated
 * atomically -- no race between price- and bidder-write.
 *
 * @param price   Bid price (non-negative).
 * @param bidder  Index of the bidding person.
 * @return Packed uint64_t with price in high word, bidder in low word.
 */
__device__ __forceinline__ uint64_t pack_bid(int price, int bidder) {
    return ((uint64_t)(uint32_t)price << 32) | (uint32_t)bidder;
}

/** @brief Extract bid price from a packed uint64_t. */
__device__ __forceinline__ int unpack_price(uint64_t packed) {
    return (int)(packed >> 32);
}

/** @brief Extract bidder index from a packed uint64_t. */
__device__ __forceinline__ int unpack_bidder(uint64_t packed) {
    return (int)(packed & 0xFFFFFFFFULL);
}

/* ------------------------------------------------------------------ */
/*  Kernel                                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief GPU parallel Auction Algorithm kernel (epsilon-scaling).
 *
 * Each CUDA block (blockIdx.x == problem index) solves one assignment
 * problem independently.  Kernel exits when all persons are assigned or
 * stall-detection forces remaining unassigned persons to outlier.
 *
 * The auction proceeds in rounds: BID -> ASSIGN -> check-convergence.
 * Epsilon starts at 8 and decreases through 4, 2, 1, to 0 for exact
 * optimality on integer costs.
 *
 * @param[in]  n_persons   [P] Number of persons per problem.
 * @param[in]  n_objects   [P] Number of objects per problem.
 * @param[in]  edge_person [total_edges] Person index per edge.
 * @param[in]  edge_object [total_edges] Object index per edge.
 * @param[in]  edge_cost   [total_edges] Cost (int, typically distance^2).
 * @param[in]  outlier_cost [P] Cost of leaving a person unmatched.
 * @param[out] assignment  [total_persons] Assigned object index per person
 *                         (n2 means outlier/unmatched).
 * @param[out] prices      [total_objects] Current price per object.
 * @param      packed_bids [total_objects] Workspace: packed bid per object.
 * @param      owners_ws   [total_objects] Workspace: current owner per object.
 * @param[in]  po_start    [P + 1] Prefix sum of persons per problem.
 * @param[in]  oo_start    [P + 1] Prefix sum of objects per problem.
 * @param[in]  pe_start    [total_persons + 1] Edge-range per person (flat).
 * @param[in]  pe_offset   [P + 1] Section start in pe_start per problem.
 * @param[in]  P           Number of problems in this batch.
 */
__global__ void auction_kernel(
    const int  * __restrict__ n_persons,
    const int  * __restrict__ n_objects,
    const int  * __restrict__ edge_person,
    const int  * __restrict__ edge_object,
    const int  * __restrict__ edge_cost,
    const int  * __restrict__ outlier_cost,
    int        * __restrict__ assignment,
    int        * __restrict__ prices,
    uint64_t   * __restrict__ packed_bids,
    int        * __restrict__ owners_ws,
    const int  * __restrict__ po_start,
    const int  * __restrict__ oo_start,
    const int  * __restrict__ pe_start,
    const int  * __restrict__ pe_offset,
    int          P
) {
    int p = blockIdx.x;
    if (p >= P || n_persons[p] == 0) return;

    int n1   = n_persons[p];
    int n2   = n_objects[p];
    int oc   = outlier_cost[p];
    int po0  = po_start[p];
    int oo0  = oo_start[p];
    int pe0  = pe_offset[p];

    int *assign  = assignment + po0;
    int *pr      = prices      + oo0;
    uint64_t *pk = packed_bids + oo0;
    int *owners  = owners_ws   + oo0;

    int tid   = threadIdx.x;
    int n_thr = blockDim.x;

    /* ---------- Init ---------- */
    for (int j = tid; j < n2; j += n_thr) {
        pr[j] = 0;
        pk[j] = 0;
        owners[j] = -1;
    }
    for (int i = tid; i < n1; i += n_thr) {
        assign[i] = -1;
    }
    __syncthreads();

    /* epsilon-scaling: 8 -> 4 -> 2 -> 1 -> 0 */
    int eps_values[] = {8, 4, 2, 1, 0};
    int eps_iters[]  = {ITERS_EPS8, ITERS_EPS4, ITERS_EPS2,
                        ITERS_EPS1, ITERS_EPS0};
    int n_eps_levels = 5;

    for (int eidx = 0; eidx < n_eps_levels; ++eidx) {
        int eps   = eps_values[eidx];
        int limit = eps_iters[eidx];

        for (int j = tid; j < n2; j += n_thr) {
            pk[j] = 0;
        }
        __syncthreads();

        int iter = 0;
        while (iter < limit) {
            for (int j = tid; j < n2; j += n_thr) {
                pk[j] = 0;
            }
            __syncthreads();

            /* ---- BIDDING PHASE ---- */
            for (int i = tid; i < n1; i += n_thr) {
                if (assign[i] >= 0) continue;

                int es = pe_start[pe0 + i];
                int ee = pe_start[pe0 + i + 1];

                int best_j   = -1;
                int best_val = INT_MAX;
                int second   = INT_MAX;

                for (int e = es; e < ee; ++e) {
                    int j   = edge_object[e];
                    int val = edge_cost[e] + pr[j];
                    if (val < best_val) {
                        second   = best_val;
                        best_val = val;
                        best_j   = j;
                    } else if (val < second) {
                        second = val;
                    }
                }

                if (oc < best_val) {
                    second   = best_val;
                    best_val = oc;
                    best_j   = n2;
                } else if (oc < second) {
                    second = oc;
                }

                if (best_j == -1) continue;
                if (second == INT_MAX) second = best_val + (eps > 0 ? eps : 1);

                int inc = (second - best_val) + (eps > 0 ? eps : 0);
                if (inc < 1) inc = 1;

                if (best_j < n2) {
                    int bid_v = pr[best_j] + inc;
                    uint64_t new_packed = pack_bid(bid_v, i);
                    atomicMax((unsigned long long *)&pk[best_j],
                              (unsigned long long)new_packed);
                } else {
                    assign[i] = n2;
                }
            }
            __syncthreads();

            /* ---- ASSIGNMENT PHASE ---- */
            __shared__ int s_any_change;
            if (tid == 0) s_any_change = 0;
            __syncthreads();

            for (int j = tid; j < n2; j += n_thr) {
                uint64_t packed = pk[j];
                int b_price = unpack_price(packed);
                int b_bidder = unpack_bidder(packed);

                if (b_price > pr[j] && b_bidder >= 0 && b_bidder < n1) {
                    int prev_owner = owners[j];
                    if (prev_owner >= 0 && prev_owner < n1 && assign[prev_owner] == j) {
                        assign[prev_owner] = -1;
                    }
                    if (assign[b_bidder] == -1) {
                        assign[b_bidder] = j;
                        owners[j] = b_bidder;
                        pr[j] = b_price;
                        s_any_change = 1;
                    }
                }
            }
            __syncthreads();

            int remaining = 0;
            for (int i = tid; i < n1; i += n_thr) {
                if (assign[i] < 0) remaining++;
            }
            __shared__ int s_rem;
            if (tid == 0) s_rem = 0;
            __syncthreads();
            atomicAdd(&s_rem, remaining);
            __syncthreads();
            if (s_rem == 0) break;

            __shared__ int s_stall;
            if (tid == 0) s_stall = (s_any_change == 0) ? 1 : 0;
            __syncthreads();
            if (s_stall) {
                for (int i = tid; i < n1; i += n_thr)
                    if (assign[i] < 0) assign[i] = n2;
                break;
            }
            ++iter;
        }
    }

    for (int i = tid; i < n1; i += n_thr)
        if (assign[i] < 0) assign[i] = n2;
}

/* ------------------------------------------------------------------ */
/*  Host entry point                                                  */
/* ------------------------------------------------------------------ */

extern "C" {

/**
 * @brief Solve a batch of assignment problems on GPU.
 *
 * Allocates device memory, copies inputs, launches the kernel, copies
 * assignment results back, and frees device resources.  Every CUDA call
 * is wrapped with CUDA_CHECK() for immediate error detection.
 *
 * @param  P                Number of problems in the batch.
 * @param  h_n_persons      [P] Persons per problem (host).
 * @param  h_n_objects      [P] Objects per problem (host).
 * @param  h_edge_start     [P + 1] Exclusive-prefix edge index per problem.
 * @param  h_edge_person    [total_edges] Person index per edge (host).
 * @param  h_edge_object    [total_edges] Object index per edge (host).
 * @param  h_edge_cost      [total_edges] Edge cost (int, host).
 * @param  h_outlier_cost   [P] Outlier cost per problem (host).
 * @param  h_assignment     [total_persons] Output assignments (host buffer).
 * @param  h_pe_start       [total_persons + 1] Per-person edge ranges (host).
 * @param  h_pe_offset      [P + 1] Section offsets in pe_start (host).
 * @param  verbose          If nonzero, print batch summary to stdout.
 * @return 0 on success, 1 on any CUDA error.
 */
int auction_solve_batch(
    int    P,
    const int *h_n_persons,
    const int *h_n_objects,
    const int *h_edge_start,
    const int *h_edge_person,
    const int *h_edge_object,
    const int *h_edge_cost,
    const int *h_outlier_cost,
    int       *h_assignment,
    const int *h_pe_start,
    const int *h_pe_offset,
    int        verbose
) {
    int total_persons = 0, total_objects = 0;
    int total_edges = (P > 0) ? h_edge_start[P] : 0;
    for (int p = 0; p < P; ++p) {
        total_persons += h_n_persons[p];
        total_objects += h_n_objects[p];
    }

    if (verbose) {
        printf("[auction] P=%d persons=%d objects=%d edges=%d\n",
               P, total_persons, total_objects, total_edges);
    }

    int *h_po = new int[P + 1];
    int *h_oo = new int[P + 1];
    h_po[0] = h_oo[0] = 0;
    for (int p = 0; p < P; ++p) {
        h_po[p + 1] = h_po[p] + h_n_persons[p];
        h_oo[p + 1] = h_oo[p] + h_n_objects[p];
    }

    int *d_np = NULL, *d_no = NULL, *d_es = NULL;
    int *d_ep = NULL, *d_eo = NULL, *d_ec = NULL, *d_oc = NULL;
    int *d_as = NULL, *d_pr = NULL;
    uint64_t *d_pk = NULL;
    int *d_own = NULL;
    int *d_po = NULL, *d_oo = NULL;
    int *d_pes = NULL, *d_peo = NULL;
    cudaError_t err = cudaSuccess;

    /* ---------- Device allocation (all checked) ---------- */
    CUDA_CHECK(cudaMalloc(&d_np,   P * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_no,   P * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_es,  (P + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ep,   total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eo,   total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ec,   total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_oc,   P * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_as,   total_persons * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pr,   total_objects * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pk,   total_objects * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_po,  (P + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_oo,  (P + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pes, (total_persons + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_peo, (P + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_own, total_objects * sizeof(int)));

    /* ---------- Host -> Device copy (all checked) ---------- */
    CUDA_CHECK(cudaMemcpy(d_np,  h_n_persons,   P * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_no,  h_n_objects,   P * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_es,  h_edge_start,  (P + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ep,  h_edge_person, total_edges * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eo,  h_edge_object, total_edges * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ec,  h_edge_cost,   total_edges * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oc,  h_outlier_cost, P * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_po,  h_po,         (P + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_oo,  h_oo,         (P + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pes, h_pe_start,   (total_persons + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_peo, h_pe_offset,  (P + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    /* ---------- Device memory initialization (all checked) ---------- */
    CUDA_CHECK(cudaMemset(d_as,  0xFF, total_persons * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pr,  0,    total_objects * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_own, 0xFF, total_objects * sizeof(int)));

    /* ---------- Launch kernel ---------- */
    auction_kernel<<<P, MAX_THREADS>>>(
        d_np, d_no, d_ep, d_eo, d_ec, d_oc, d_as,
        d_pr, d_pk, d_own, d_po, d_oo, d_pes, d_peo, P
    );

    /* Combined kernel launch + sync error check (NVIDIA Samples style) */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error at %s:%d -- %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        goto cleanup;
    }

    /* ---------- Device -> Host copy ---------- */
    CUDA_CHECK(cudaMemcpy(h_assignment, d_as,
                          total_persons * sizeof(int),
                          cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(d_own);
    cudaFree(d_np); cudaFree(d_no); cudaFree(d_es);
    cudaFree(d_ep); cudaFree(d_eo); cudaFree(d_ec);
    cudaFree(d_oc); cudaFree(d_as); cudaFree(d_pr);
    cudaFree(d_pk);
    cudaFree(d_po); cudaFree(d_oo);
    cudaFree(d_pes); cudaFree(d_peo);
    delete[] h_po;
    delete[] h_oo;

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

} /* extern "C" */
