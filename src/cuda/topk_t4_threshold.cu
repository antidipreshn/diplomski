#include "redukcija_warp.cuh"

extern "C" __global__
void t4_max_grupe(const float* __restrict__ slicnosti, int N, int velicina_grupe,
                  int broj_grupa, unsigned long long* __restrict__ maksimumi)
{
    int globalni = blockIdx.x * blockDim.x + threadIdx.x;
    int grupa = globalni / WARP;
    int lane = threadIdx.x % WARP;
    if (grupa >= broj_grupa) return;

    int pocetak = grupa * velicina_grupe;
    int kraj = min(pocetak + velicina_grupe, N);
    unsigned long long najbolji = 0;
    for (int i = pocetak + lane; i < kraj; i += WARP) {
        unsigned long long kljuc = napravi_kljuc(slicnosti[i], (unsigned int)i);
        if (kljuc > najbolji) najbolji = kljuc;
    }

    najbolji = maks_u_warpu(najbolji);
    if (lane == 0) maksimumi[grupa] = najbolji;
}

extern "C" __global__
void t4_prag(const unsigned long long* __restrict__ maksimumi, int broj_grupa, int k,
             unsigned long long* __restrict__ najveci)
{
    lokalni_topk_warp(nullptr, maksimumi, 0, broj_grupa, k, najveci);
}

extern "C" __global__
void t4_filtriraj(const float* __restrict__ slicnosti, int N,
                  const unsigned long long* __restrict__ najveci, int k,
                  unsigned long long* __restrict__ kandidati, int* brojac)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned long long kljuc = napravi_kljuc(slicnosti[i], (unsigned int)i);
    if (kljuc >= najveci[k - 1]) {
        kandidati[atomicAdd(brojac, 1)] = kljuc;
    }
}

extern "C" __global__
void t4_konacni(const unsigned long long* __restrict__ kandidati, const int* __restrict__ brojac,
                int k, unsigned long long* __restrict__ konacni)
{
    lokalni_topk_warp(nullptr, kandidati, 0, *brojac, k, konacni);
}
