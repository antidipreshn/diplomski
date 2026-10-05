extern "C" __global__
void slicnost_s1(const float* __restrict__ korpus,
                 const float* __restrict__ upit,
                 float* __restrict__ izlaz,
                 int N, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* red = korpus + (size_t)i * d;
    float s = 0.0f;
    for (int j = 0; j < d; j++) {
        s += red[j] * upit[j];
    }
    izlaz[i] = s;
}
