#include "kljuc.cuh"

extern "C" __global__
void t1_runda(const float* __restrict__ slicnosti, int N,
              unsigned long long* rezultati, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned long long granica = (j == 0) ? MAKS_KLJUC : rezultati[j - 1];
    unsigned long long kljuc = napravi_kljuc(slicnosti[i], (unsigned int)i);
    if (kljuc < granica) {
        atomicMax(&rezultati[j], kljuc);
    }
}
