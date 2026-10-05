#include "redukcija_warp.cuh"

extern "C" __global__
void t3_faza1(const float* __restrict__ slicnosti, int N, int segment, int k,
              unsigned long long* __restrict__ lokalni)
{
    int pocetak = blockIdx.x * segment;
    int kraj = min(pocetak + segment, N);
    lokalni_topk_warp(slicnosti, nullptr, pocetak, kraj, k, lokalni + (size_t)blockIdx.x * k);
}

extern "C" __global__
void t3_faza2(const unsigned long long* __restrict__ kandidati, int n, int k,
              unsigned long long* __restrict__ konacni)
{
    lokalni_topk_warp(nullptr, kandidati, 0, n, k, konacni);
}
