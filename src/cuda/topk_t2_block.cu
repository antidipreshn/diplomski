#include "kljuc.cuh"

__device__ void lokalni_topk(const float* __restrict__ slicnosti,
                             const unsigned long long* __restrict__ kljucevi,
                             int pocetak, int kraj, int k,
                             unsigned long long* __restrict__ izlaz)
{
    extern __shared__ unsigned long long red[];
    __shared__ unsigned long long granica;

    int t = threadIdx.x;
    if (t == 0) granica = MAKS_KLJUC;
    __syncthreads();

    for (int r = 0; r < k; r++) {
        unsigned long long najbolji = 0;
        for (int i = pocetak + t; i < kraj; i += blockDim.x) {
            unsigned long long kljuc = slicnosti ? napravi_kljuc(slicnosti[i], (unsigned int)i)
                                                 : kljucevi[i];
            if (kljuc < granica && kljuc > najbolji) najbolji = kljuc;
        }

        red[t] = najbolji;
        __syncthreads();
        for (int pola = blockDim.x / 2; pola > 0; pola >>= 1) {
            if (t < pola && red[t + pola] > red[t]) red[t] = red[t + pola];
            __syncthreads();
        }

        if (t == 0) {
            izlaz[r] = red[0];
            granica = red[0];
        }
        __syncthreads();
    }
}

extern "C" __global__
void t2_faza1(const float* __restrict__ slicnosti, int N, int segment, int k,
              unsigned long long* __restrict__ lokalni)
{
    int pocetak = blockIdx.x * segment;
    int kraj = min(pocetak + segment, N);
    lokalni_topk(slicnosti, nullptr, pocetak, kraj, k, lokalni + (size_t)blockIdx.x * k);
}

extern "C" __global__
void t2_faza2(const unsigned long long* __restrict__ kandidati, int n, int k,
              unsigned long long* __restrict__ konacni)
{
    lokalni_topk(nullptr, kandidati, 0, n, k, konacni);
}
