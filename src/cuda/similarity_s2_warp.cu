#define WARP 32

extern "C" __global__
void slicnost_s2(const float* __restrict__ korpus,
                 const float* __restrict__ upit,
                 float* __restrict__ izlaz,
                 int N, int d)
{
    int red = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    int lane = threadIdx.x % WARP;
    if (red >= N) return;

    const float* r = korpus + (size_t)red * d;
    float s = 0.0f;
    for (int j = lane; j < d; j += WARP) {
        s += r[j] * upit[j];
    }

    for (int pomeranje = WARP / 2; pomeranje > 0; pomeranje >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, pomeranje);
    }
    if (lane == 0) izlaz[red] = s;
}
