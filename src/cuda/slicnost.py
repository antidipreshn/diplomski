import numpy as np
import cupy as cp

from src.cuda.similarity_s5_cublas import slicnost_s5
from src.cuda.ucitavanje import ucitaj_kernel


def proveri_ulaze(korpus, upit, izlaz):
    for naziv, niz in (("korpus", korpus), ("upit", upit), ("izlaz", izlaz)):
        if niz.dtype != cp.float32:
            raise ValueError(f"{naziv}: tip {niz.dtype}, ocekivan float32")
        if not niz.flags.c_contiguous:
            raise ValueError(f"{naziv}: nije C-contiguous")
    N, d = korpus.shape
    if upit.shape != (d,) or izlaz.shape != (N,):
        raise ValueError(f"oblici: korpus {korpus.shape}, upit {upit.shape}, izlaz {izlaz.shape}")


def slicnost_s1(korpus, upit, izlaz, cfg):
    N, d = korpus.shape
    blok = cfg["blok"]
    kernel = ucitaj_kernel("similarity_s1_naive.cu", "slicnost_s1")
    kernel(((N + blok - 1) // blok,), (blok,),
           (korpus, upit, izlaz, np.int32(N), np.int32(d)))


def slicnost_s2(korpus, upit, izlaz, cfg):
    N, d = korpus.shape
    warpova = cfg["warpova_po_bloku"]
    kernel = ucitaj_kernel("similarity_s2_warp.cu", "slicnost_s2")
    kernel(((N + warpova - 1) // warpova,), (warpova * 32,),
           (korpus, upit, izlaz, np.int32(N), np.int32(d)))


def slicnost_s3(korpus, upit, izlaz, cfg):
    N, d = korpus.shape
    warpova = cfg["warpova_po_bloku"]
    kernel = ucitaj_kernel("similarity_s3_shared.cu", "slicnost_s3")
    kernel(((N + warpova - 1) // warpova,), (warpova * 32,),
           (korpus, upit, izlaz, np.int32(N), np.int32(d)),
           shared_mem=d * 4)


def proveri_float4(korpus, upiti):
    d = korpus.shape[1]
    if d % 4 != 0:
        raise ValueError(f"S4: d = {d} nije deljivo sa 4")
    for naziv, niz in (("korpus", korpus), ("upiti", upiti)):
        if niz.data.ptr % 16 != 0:
            raise ValueError(f"S4: adresa niza {naziv} nije poravnata na 16 B")


def slicnost_s4(korpus, upit, izlaz, cfg):
    N, d = korpus.shape
    warpova = cfg["warpova_po_bloku"]
    kernel = ucitaj_kernel("similarity_s4_vectorized.cu", "slicnost_s4")
    kernel(((N + warpova - 1) // warpova,), (warpova * 32,),
           (korpus, upit, izlaz, np.int32(N), np.int32(d // 4)),
           shared_mem=d * 4)


SLICNOSTI = {
    "gpu_s1": ("s1", slicnost_s1),
    "gpu_s2": ("s2", slicnost_s2),
    "gpu_s3": ("s3", slicnost_s3),
    "gpu_s4": ("s4", slicnost_s4),
    "gpu_s5": ("s5", slicnost_s5),
}

TRAZE_FLOAT4 = {"gpu_s4"}


def bajtova_slicnosti(N, d):
    return (N * d + d + N) * 4


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.common.data import napravi_korpus, napravi_upit

    d = 768
    cfg = {"s1": {"blok": 256}, "s2": {"warpova_po_bloku": 8}, "s3": {"warpova_po_bloku": 8},
           "s4": {"warpova_po_bloku": 8}, "s5": {}}
    for naziv, (kljuc, funkcija) in SLICNOSTI.items():
        for N in (1, 255, 256, 257, 12_345, 100_000):
            korpus = napravi_korpus(N, d, seed=0)
            upit = napravi_upit(d, seed=1)
            korpus_gpu, upit_gpu = cp.asarray(korpus), cp.asarray(upit)
            izlaz = cp.full(N, cp.nan, dtype=cp.float32)
            proveri_ulaze(korpus_gpu, upit_gpu, izlaz)
            if naziv in TRAZE_FLOAT4:
                proveri_float4(korpus_gpu, upit_gpu)

            funkcija(korpus_gpu, upit_gpu, izlaz, cfg[kljuc])
            rezultat = cp.asnumpy(izlaz)
            razlika = float(np.max(np.abs(rezultat.astype(np.float64) - korpus @ upit)))
            status = "OK" if razlika <= 1e-5 else "GRESKA"
            print(f"{naziv} N={N:<7} max razlika {razlika:.3e}  {status}")

    korpus_gpu = cp.asarray(napravi_korpus(100, d, seed=0))
    for opis, korpus_x, upiti_x in (
            ("d = 766 (nije deljivo sa 4)", korpus_gpu[:, :766], cp.zeros((1, 766), cp.float32)),
            ("upit pomeren za 4 B", korpus_gpu, cp.zeros(d + 1, cp.float32)[1:])):
        try:
            proveri_float4(korpus_x, upiti_x)
            print("GRESKA: S4 provera je prosla za:", opis)
        except ValueError as greska:
            print("S4 odbijeno kako treba:", greska)
