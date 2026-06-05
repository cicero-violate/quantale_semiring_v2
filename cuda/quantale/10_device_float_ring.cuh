// ── Device-ring push / pop ────────────────────────────────────────────────────

// Write n floats into the ring from src (single-threaded for head/tail safety).
extern "C" __global__ void device_ring_push(
    float* ring, int* tail, int capacity,
    const float* src, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int t = *tail;
    for (int i = 0; i < n; ++i) {
        ring[t % capacity] = src[i];
        ++t;
    }
    *tail = t;
}

// Read n floats from the ring into dst (single-threaded for head/tail safety).
extern "C" __global__ void device_ring_pop(
    const float* ring, int* head, int capacity,
    float* dst, int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int h = *head;
    for (int i = 0; i < n; ++i) {
        dst[i] = ring[h % capacity];
        ++h;
    }
    *head = h;
}

