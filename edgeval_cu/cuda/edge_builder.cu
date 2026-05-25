/* edge_builder.cu — Fused CUDA kernel for graph construction.
 *
 * Single kernel launch replaces: torch.cdist + mask + nonzero + .cpu()
 *
 * All memory is managed by the caller (PyTorch tensors on GPU).
 * The caller passes GPU pointers directly.
 *
 * Build: nvcc -shared -arch=sm_89 -O3 -Xcompiler -fPIC -o edge_builder.so edge_builder.cu
 */

#include <cuda_runtime.h>
#include <math.h>

__global__ void build_edges_kernel(
    const float* __restrict__ pred,    // [n_pred, 2]
    const float* __restrict__ gt,      // [n_gt_total, 2]
    int n_pred,
    int n_gt_total,
    float max_dist,
    int scale,
    int* __restrict__ edge_count,       // [1] GPU counter
    int* __restrict__ edges_out,        // [max_edges, 3] GPU output
    int max_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = n_pred * n_gt_total;

    for (int i = tid; i < total; i += stride) {
        int p = i / n_gt_total;
        int g = i % n_gt_total;

        float dx = pred[p * 2] - gt[g * 2];
        float dy = pred[p * 2 + 1] - gt[g * 2 + 1];
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist <= max_dist) {
            int idx = atomicAdd(edge_count, 1);
            if (idx < max_edges) {
                edges_out[idx * 3] = p;
                edges_out[idx * 3 + 1] = g;
                edges_out[idx * 3 + 2] = (int)roundf(dist * (float)scale);
            }
        }
    }
}

// Host wrapper — caller manages all GPU memory via PyTorch tensors.
// All pointers must be GPU-allocated (cudaMalloc or torch tensor .data_ptr()).
extern "C" int launch_build_edges(
    const float* pred_gpu, const float* gt_gpu,
    int n_pred, int n_gt_total,
    float max_dist, int scale,
    int* count_gpu, int* edges_gpu, int max_edges
) {
    // Zero counter first
    cudaMemset(count_gpu, 0, sizeof(int));

    int total = n_pred * n_gt_total;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;

    build_edges_kernel<<<grid, block>>>(
        pred_gpu, gt_gpu, n_pred, n_gt_total,
        max_dist, scale,
        count_gpu, edges_gpu, max_edges
    );

    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : (int)err;
}
