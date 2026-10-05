#pragma once

#include "kljuc.cuh"

#define WARP 32

__device__ __forceinline__ unsigned long long maks_u_warpu(unsigned long long v)
{
    for (int pomak = WARP / 2; pomak > 0; pomak >>= 1) {
        unsigned long long drugi = __shfl_down_sync(0xffffffff, v, pomak);
        if (drugi > v) v = drugi;
    }
    return v;
}

__device__ void lokalni_topk_warp(const float* __restrict__ slicnosti,
                                  const unsigned long long* __restrict__ kljucevi,
                                  int pocetak, int kraj, int k,
                                  unsigned long long* __restrict__ izlaz)
{
    extern __shared__ unsigned long long po_warpu[];
    __shared__ unsigned long long granica;

    int t = threadIdx.x;
    int warpova = blockDim.x / WARP;
    if (t == 0) granica = MAKS_KLJUC;
    __syncthreads();

    for (int r = 0; r < k; r++) {
        unsigned long long najbolji = 0;
        for (int i = pocetak + t; i < kraj; i += blockDim.x) {
            unsigned long long kljuc = slicnosti ? napravi_kljuc(slicnosti[i], (unsigned int)i)
                                                 : kljucevi[i];
            if (kljuc < granica && kljuc > najbolji) najbolji = kljuc;
        }

        najbolji = maks_u_warpu(najbolji);
        if (t % WARP == 0) po_warpu[t / WARP] = najbolji;
        __syncthreads();

        if (t < WARP) {
            unsigned long long v = (t < warpova) ? po_warpu[t] : 0;
            v = maks_u_warpu(v);
            if (t == 0) {
                izlaz[r] = v;
                granica = v;
            }
        }
        __syncthreads();
    }
}
