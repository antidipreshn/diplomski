#define WARP 32

extern "C" __global__
void slicnost_s4(const float4* __restrict__ korpus,
                 const float4* __restrict__ upit,
                 float* __restrict__ izlaz,
                 int N, int d4)
{
    extern __shared__ float4 q4[];

    for (int j = threadIdx.x; j < d4; j += blockDim.x) {
        q4[j] = upit[j];
    }
    __syncthreads();

    int red = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    int lane = threadIdx.x % WARP;
    if (red >= N) return;

    const float4* r = korpus + (size_t)red * d4;
    float s = 0.0f;
    for (int j = lane; j < d4; j += WARP) {
        float4 a = r[j];
        float4 b = q4[j];
        s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    for (int pomeranje = WARP / 2; pomeranje > 0; pomeranje >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, pomeranje);
    }
    if (lane == 0) izlaz[red] = s;
}
