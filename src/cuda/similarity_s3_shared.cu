#define WARP 32

extern "C" __global__
void slicnost_s3(const float* __restrict__ korpus,
                 const float* __restrict__ upit,
                 float* __restrict__ izlaz,
                 int N, int d)
{
    extern __shared__ float q[];

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        q[j] = upit[j];
    }
    __syncthreads();

    int red = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    int lane = threadIdx.x % WARP;
    if (red >= N) return;

    const float* r = korpus + (size_t)red * d;
    float s = 0.0f;
    for (int j = lane; j < d; j += WARP) {
        s += r[j] * q[j];
    }

    for (int pomak = WARP / 2; pomak > 0; pomak >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, pomak);
    }
    if (lane == 0) izlaz[red] = s;
}
