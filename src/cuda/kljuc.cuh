#pragma once

#define MAKS_KLJUC 0xFFFFFFFFFFFFFFFFull

__device__ __forceinline__ unsigned long long napravi_kljuc(float v, unsigned int i)
{
    unsigned int b = __float_as_uint(v);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)b << 32) | (unsigned int)(~i);
}

__device__ __forceinline__ float vrednost_kljuca(unsigned long long kljuc)
{
    unsigned int b = (unsigned int)(kljuc >> 32);
    b = (b & 0x80000000u) ? (b & 0x7FFFFFFFu) : ~b;
    return __uint_as_float(b);
}

__device__ __forceinline__ long long indeks_kljuca(unsigned long long kljuc)
{
    return (long long)(~(unsigned int)(kljuc & 0xFFFFFFFFull));
}

extern "C" __global__
void dekodiraj(const unsigned long long* __restrict__ kljucevi,
               long long* __restrict__ indeksi,
               float* __restrict__ vrednosti,
               int k)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= k) return;
    indeksi[r] = indeks_kljuca(kljucevi[r]);
    vrednosti[r] = vrednost_kljuca(kljucevi[r]);
}
