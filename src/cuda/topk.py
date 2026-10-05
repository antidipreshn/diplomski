import numpy as np
import cupy as cp

from src.cuda.ucitavanje import ucitaj_kernel

_DEKODIRAJ_BLOK = 128


def _proveri_k(N, k):
    if not 1 <= k <= N:
        raise ValueError(f"k = {k} mora biti izmedju 1 i N = {N}")


def _dekodiraj(ime_fajla, kljucevi, radni, k):
    kernel = ucitaj_kernel(ime_fajla, "dekodiraj")
    kernel(((k + _DEKODIRAJ_BLOK - 1) // _DEKODIRAJ_BLOK,), (_DEKODIRAJ_BLOK,),
           (kljucevi, radni["indeksi"], radni["vrednosti"], np.int32(k)))
    return radni["indeksi"], radni["vrednosti"]


def _izlaz(k):
    return {"indeksi": cp.empty(k, dtype=cp.int64), "vrednosti": cp.empty(k, dtype=cp.float32)}


def pripremi_t1(N, k, cfg):
    _proveri_k(N, k)
    return {"rezultati": cp.empty(k, dtype=cp.uint64), **_izlaz(k)}


def topk_t1(slicnosti, k, cfg, radni):
    N = slicnosti.shape[0]
    blok = cfg["blok"]
    kernel = ucitaj_kernel("topk_t1_atomic.cu", "t1_runda")
    rezultati = radni["rezultati"]
    rezultati.fill(0)
    for j in range(k):
        kernel(((N + blok - 1) // blok,), (blok,), (slicnosti, np.int32(N), rezultati, np.int32(j)))
    return _dekodiraj("topk_t1_atomic.cu", rezultati, radni, k)


def _proveri_blok(blok):
    if blok & (blok - 1) or not 32 <= blok <= 1024:
        raise ValueError(f"blok = {blok} mora biti stepen dvojke izmedju 32 i 1024 (redukcija stablom)")


def pripremi_t2(N, k, cfg):
    _proveri_k(N, k)
    _proveri_blok(cfg["blok"])
    broj_blokova = (N + cfg["segment"] - 1) // cfg["segment"]
    return {"broj_blokova": broj_blokova,
            "lokalni": cp.empty(broj_blokova * k, dtype=cp.uint64),
            "konacni": cp.empty(k, dtype=cp.uint64), **_izlaz(k)}


def topk_t2(slicnosti, k, cfg, radni):
    N = slicnosti.shape[0]
    blok, B = cfg["blok"], radni["broj_blokova"]
    deljena = blok * 8
    faza1 = ucitaj_kernel("topk_t2_block.cu", "t2_faza1")
    faza2 = ucitaj_kernel("topk_t2_block.cu", "t2_faza2")
    faza1((B,), (blok,), (slicnosti, np.int32(N), np.int32(cfg["segment"]), np.int32(k),
                          radni["lokalni"]), shared_mem=deljena)
    faza2((1,), (blok,), (radni["lokalni"], np.int32(B * k), np.int32(k), radni["konacni"]),
          shared_mem=deljena)
    return _dekodiraj("topk_t2_block.cu", radni["konacni"], radni, k)


def pripremi_t3(N, k, cfg):
    radni = pripremi_t2(N, k, cfg)
    if cfg["blok"] // 32 > 32:
        raise ValueError("blok / 32 mora biti najvise 32 (warp 0 spaja vrednosti warp-ova)")
    return radni


def topk_t3(slicnosti, k, cfg, radni):
    N = slicnosti.shape[0]
    blok, B = cfg["blok"], radni["broj_blokova"]
    deljena = (blok // 32) * 8
    faza1 = ucitaj_kernel("topk_t3_warp.cu", "t3_faza1")
    faza2 = ucitaj_kernel("topk_t3_warp.cu", "t3_faza2")
    faza1((B,), (blok,), (slicnosti, np.int32(N), np.int32(cfg["segment"]), np.int32(k),
                          radni["lokalni"]), shared_mem=deljena)
    faza2((1,), (blok,), (radni["lokalni"], np.int32(B * k), np.int32(k), radni["konacni"]),
          shared_mem=deljena)
    return _dekodiraj("topk_t3_warp.cu", radni["konacni"], radni, k)


def pripremi_t4(N, k, cfg):
    _proveri_k(N, k)
    grupa = cfg["velicina_grupe"]
    broj_grupa = (N + grupa - 1) // grupa
    if broj_grupa < k:
        raise ValueError(f"T4: {broj_grupa} grupa za N = {N} je manje od k = {k}; "
                         f"smanjiti velicina_grupe (sada {grupa})")
    return {"broj_grupa": broj_grupa,
            "maksimumi": cp.empty(broj_grupa, dtype=cp.uint64),
            "najveci": cp.empty(k, dtype=cp.uint64),
            "kandidati": cp.empty(N, dtype=cp.uint64),
            "brojac": cp.zeros(1, dtype=cp.int32),
            "konacni": cp.empty(k, dtype=cp.uint64), **_izlaz(k)}


def topk_t4(slicnosti, k, cfg, radni):
    N = slicnosti.shape[0]
    blok, grupa, G = cfg["blok"], cfg["velicina_grupe"], radni["broj_grupa"]
    deljena = (blok // 32) * 8
    fajl = "topk_t4_threshold.cu"
    radni["brojac"].fill(0)

    ucitaj_kernel(fajl, "t4_max_grupe")(
        ((G * 32 + blok - 1) // blok,), (blok,),
        (slicnosti, np.int32(N), np.int32(grupa), np.int32(G), radni["maksimumi"]))
    ucitaj_kernel(fajl, "t4_prag")(
        (1,), (blok,), (radni["maksimumi"], np.int32(G), np.int32(k), radni["najveci"]),
        shared_mem=deljena)
    ucitaj_kernel(fajl, "t4_filtriraj")(
        ((N + blok - 1) // blok,), (blok,),
        (slicnosti, np.int32(N), radni["najveci"], np.int32(k), radni["kandidati"], radni["brojac"]))
    ucitaj_kernel(fajl, "t4_konacni")(
        (1,), (blok,), (radni["kandidati"], radni["brojac"], np.int32(k), radni["konacni"]),
        shared_mem=deljena)
    return _dekodiraj(fajl, radni["konacni"], radni, k)


TOPK = {
    "gpu_t1": ("t1", pripremi_t1, topk_t1),
    "gpu_t2": ("t2", pripremi_t2, topk_t2),
    "gpu_t3": ("t3", pripremi_t3, topk_t3),
    "gpu_t4": ("t4", pripremi_t4, topk_t4),
}


def proveri_topk(indeksi, vrednosti, ref_indeksi, ref_vrednosti, naziv):
    indeksi, vrednosti = cp.asnumpy(indeksi), cp.asnumpy(vrednosti)
    if not np.array_equal(indeksi, ref_indeksi):
        razliciti = np.nonzero(indeksi != ref_indeksi)[0]
        raise AssertionError(f"{naziv}: indeksi se razlikuju od pozicije {razliciti[0]} "
                             f"({indeksi[razliciti[0]]} umesto {ref_indeksi[razliciti[0]]})")
    if not np.array_equal(vrednosti.view(np.uint32), ref_vrednosti.view(np.uint32)):
        raise AssertionError(f"{naziv}: vrednosti nisu bit-identicne")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.cpu.numpy_ref import top_k_numpy_redosled

    cfg = {"t1": {"blok": 256}, "t2": {"blok": 256, "segment": 4096},
           "t3": {"blok": 256, "segment": 4096}, "t4": {"blok": 256, "velicina_grupe": 32}}
    generator = np.random.default_rng(0)

    def slucajevi():
        for N in (1, 255, 257, 10_000, 100_000):
            x = generator.standard_normal(N, dtype=np.float32)
            yield f"slucajno N={N}", x
        x = generator.standard_normal(10_000, dtype=np.float32)
        yield "(a) jednake vrednosti", np.round(x, 1)
        yield "(a) sve jednake", np.full(1000, 0.5, dtype=np.float32)
        yield "(c) sve negativne", -np.abs(x) - 0.1
        yield "(b) k = N", generator.standard_normal(50, dtype=np.float32)

    for naziv, (kljuc, pripremi, funkcija) in TOPK.items():
        for opis, x in slucajevi():
            N = x.shape[0]
            vrednosti_k = [N] if opis.startswith("(b)") else [k for k in (1, 5, 100) if k <= N]
            if naziv == "gpu_t4":
                vrednosti_k = [k for k in vrednosti_k if N // cfg[kljuc]["velicina_grupe"] >= k]
                if not vrednosti_k:
                    print(f"{naziv} {opis:<22} preskoceno (premalo grupa za k)")
                    continue
            x_gpu = cp.asarray(x)
            for k in vrednosti_k:
                radni = pripremi(N, k, cfg[kljuc])
                indeksi, vrednosti = funkcija(x_gpu, k, cfg[kljuc], radni)
                ref_i, ref_v = top_k_numpy_redosled(x, k)
                proveri_topk(indeksi, vrednosti, ref_i, ref_v, f"{naziv} {opis} k={k}")
            print(f"{naziv} {opis:<22} k={vrednosti_k}  OK")

    x = generator.standard_normal(10_000, dtype=np.float32)
    mali = {"blok": 32, "segment": 64}
    for naziv, pripremi, funkcija in (("gpu_t2", pripremi_t2, topk_t2),
                                      ("gpu_t3", pripremi_t3, topk_t3)):
        for k in (1, 63, 64, 65, 100):
            radni = pripremi(x.size, k, mali)
            proveri_topk(*funkcija(cp.asarray(x), k, mali, radni), *top_k_numpy_redosled(x, k),
                         f"{naziv} segment 64 k={k}")
        print(f"{naziv} segment 64, blok 32, k do 100  OK")
