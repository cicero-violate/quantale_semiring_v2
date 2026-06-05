
// ── Device helper kernels for complex IR ops ──────────────────────────────────
//
// These kernels are called directly (not via the JIT element-wise framework)
// when TypedIrOp::Reduce or TypedIrOp::TopK must be lowered to GPU code.

// Parallel warp-shuffle + shared-memory reduction (sum).
// Launch with <<<1, THREADS>>> where THREADS is a power of two <= 1024.
extern "C" __global__ void quantale_parallel_reduce(
    const float* in, float* out, int n, float init
) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    float acc = init;
    for (int i = tid; i < n; i += blockDim.x)
        acc += in[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

// Bitonic sort top-k selection: sorts the first BLOCK_SIZE elements in-place
// using shared memory and writes the top-k results to `out`.
// Assumes n <= 1024.  For production use Thrust or CUB.
extern "C" __global__ void quantale_topk_bitonic(
    float* data, float* out, int n, int k
) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? data[tid] : -1.0e30f;
    __syncthreads();

    // Bitonic sort (ascending).
    for (int size = 2; size <= blockDim.x; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int partner = tid ^ stride;
            if (partner > tid) {
                bool asc = ((tid & size) == 0);
                if (asc ? sdata[tid] > sdata[partner]
                        : sdata[tid] < sdata[partner]) {
                    float tmp = sdata[tid];
                    sdata[tid] = sdata[partner];
                    sdata[partner] = tmp;
                }
            }
            __syncthreads();
        }
    }

    // Write top-k (largest = end of ascending sort) to output.
    if (tid < k && (blockDim.x - 1 - tid) < n)
        out[tid] = sdata[blockDim.x - 1 - tid];
}
