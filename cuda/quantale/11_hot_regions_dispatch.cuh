// ── GPU region __device__ functions ──────────────────────────────────────────
//
// One __device__ function per hot region.  The dispatch kernel selects the
// right function via a switch table and calls it.  When slot_ptrs == NULL the
// function runs in receipt-only mode: it records output_flags but performs no
// element-wise work. Passing a DeviceSlotRegistry-built float** table enables
// true in-kernel computation.
//
// Slot layout per region matches regions.hot.json / operators.generated.json.

__device__ void region_vector_add(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 0);
    if (!slot_ptrs || n <= 0) return;
    float* a   = slot_ptrs[0];  // math.a
    float* b   = slot_ptrs[1];  // math.b
    float* out = slot_ptrs[2];  // math.add_out
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = a[i] + b[i];
}

__device__ void region_vector_scale(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 1);
    if (!slot_ptrs || n <= 0) return;
    float* x   = slot_ptrs[0];  // math.add_out
    float* s   = slot_ptrs[1];  // math.scale
    float* out = slot_ptrs[2];  // math.out
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = x[i] * s[i];
}

__device__ void region_fused_add_scale(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 2);
    if (!slot_ptrs || n <= 0) return;
    float* a   = slot_ptrs[0];  // math.a
    float* b   = slot_ptrs[1];  // math.b
    float* s   = slot_ptrs[2];  // math.scale
    float* out = slot_ptrs[3];  // math.out
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = (a[i] + b[i]) * s[i];
}

__device__ void region_analysis_return1(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 3);
    if (!slot_ptrs || n <= 0) return;
    float* price = slot_ptrs[0];  // market.price
    float* open  = slot_ptrs[1];  // market.open
    float* out   = slot_ptrs[2];  // analysis.return
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = (price[i] - open[i]) / (open[i] + 1e-8f);
}

__device__ void region_analysis_volatility(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 4);
    if (!slot_ptrs || n <= 0) return;
    float* price = slot_ptrs[0];  // market.price
    float* ret   = slot_ptrs[1];  // analysis.return
    float* out   = slot_ptrs[2];  // analysis.volatility
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = fabsf(price[i] - ret[i]) / (ret[i] + 1e-8f);
}

__device__ void region_analysis_signal_score(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 5);
    if (!slot_ptrs || n <= 0) return;
    float* ret = slot_ptrs[0];  // analysis.return
    float* vol = slot_ptrs[1];  // analysis.volatility
    float* out = slot_ptrs[2];  // analysis.signal_score
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = ret[i] / (1.0f + fabsf(vol[i]));
}

__device__ void region_analysis_fused_signal_score(float** slot_ptrs, int n, DeviceReceipt* r) {
    r->output_flags |= (1 << 7);
    if (!slot_ptrs || n <= 0) return;
    float* price = slot_ptrs[0];  // market.price
    float* open  = slot_ptrs[1];  // market.open
    float* out   = slot_ptrs[2];  // analysis.signal_score
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float ret = (price[i] - open[i]) / (open[i] + 1.0e-8f);
        float vol = fabsf(price[i] - ret) / (ret + 1.0e-8f);
        out[i] = ret / (1.0f + fabsf(vol));
    }
}

__device__ void region_commit_receipt(float** slot_ptrs, int n, DeviceReceipt* r) {
    (void)slot_ptrs; (void)n;
    r->output_flags |= (1 << 6);
}

// ── Generated fusion H_f handlers ─────────────────────────────────────────────
// The Rust kernel-source assembler replaces this marker with the contents of
// assets/fusion_hf.stubs.cu before PTX compilation.
// @@FUSION_HF_GENERATED_FUNCTIONS@@

// ── GPU-side region dispatch ──────────────────────────────────────────────────
//
// Selects the appropriate __device__ region function via a switch table and
// writes a DeviceReceipt to the ring — all without returning to the CPU.
//
// When slot_ptrs == NULL the region functions run in receipt-only mode; pass
// actual device slot pointer arrays plus element_count to enable true in-kernel
// computation.

extern "C" __global__ void tensor_quantale_gpu_dispatch(
    const GpuDispatchMailbox* mailbox,
    DeviceReceipt*            receipt_ring,
    int*                      ring_tail,
    int                       ring_size,
    int                       region_count,
    float**                   slot_ptrs,
    int                       element_count
) {
    int rid = mailbox->pending_region_id;
    if (rid < 0 || rid >= region_count) return;

    DeviceReceipt r;
    r.region_id    = rid;
    r.src          = mailbox->src_node;
    r.dst          = mailbox->dst_node;
    r.outcome      = mailbox->outcome;
    r.latency      = 0.0f;
    r.valid        = 1;
    r.output_flags = 0;

    switch (rid) {
        case 0: region_vector_add           (slot_ptrs, element_count, &r); break;
        case 1: region_vector_scale         (slot_ptrs, element_count, &r); break;
        case 2: region_fused_add_scale      (slot_ptrs, element_count, &r); break;
        case 3: region_analysis_return1     (slot_ptrs, element_count, &r); break;
        case 4: region_analysis_volatility  (slot_ptrs, element_count, &r); break;
        case 5: region_analysis_signal_score(slot_ptrs, element_count, &r); break;
        case 6: region_commit_receipt       (slot_ptrs, element_count, &r); break;
        case 7: region_analysis_fused_signal_score(slot_ptrs, element_count, &r); break;
        // @@FUSION_HF_GENERATED_GPU_DISPATCH_CASES@@
        default: break;
    }

    __syncthreads();

    // Only thread 0 appends to the receipt ring.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int tail = *ring_tail;
        int slot = tail % ring_size;
        receipt_ring[slot] = r;
        *ring_tail = tail + 1;
    }
}

