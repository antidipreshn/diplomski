import cupy as cp

from src.cuda.slicnost import slicnost_s2
from src.cuda.topk import pripremi_t3, topk_t3


def pripremi_pretragu(N, d, k, cfg_t3):
    return {"upit": cp.empty(d, dtype=cp.float32),
            "slicnosti": cp.empty(N, dtype=cp.float32),
            "t3": pripremi_t3(N, k, cfg_t3)}


def pretrazi_gpu(korpus_gpu, upit, k, cfg_s2, cfg_t3, radni):
    radni["upit"].set(upit)
    slicnost_s2(korpus_gpu, radni["upit"], radni["slicnosti"], cfg_s2)
    indeksi, vrednosti = topk_t3(radni["slicnosti"], k, cfg_t3, radni["t3"])
    return cp.asnumpy(indeksi), cp.asnumpy(vrednosti)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.common.data import napravi_korpus, napravi_upite
    from src.common.timing import izmeri_gpu, izmeri_gpu_po_upitima
    from src.common.verify import proveri_egzaktnost
    from src.cpu.numpy_ref import pretrazi_numpy

    N, d, Q = 100_000, 768, 20
    cfg_s2 = {"warpova_po_bloku": 8}
    cfg_t3 = {"blok": 256, "segment": 4096}
    korpus = napravi_korpus(N, d, seed=0)
    upiti = napravi_upite(Q, d, seed=1)
    korpus_gpu = cp.asarray(korpus)

    for k in (1, 10, 100):
        radni = pripremi_pretragu(N, d, k, cfg_t3)
        for q in range(Q):
            indeksi, vrednosti = pretrazi_gpu(korpus_gpu, upiti[q], k, cfg_s2, cfg_t3, radni)
            ref_ind, ref_vr = pretrazi_numpy(korpus, upiti[q], k)
            proveri_egzaktnost(indeksi, vrednosti, ref_ind, ref_vr,
                               naziv=f"pretraga (upit {q}, k={k})",
                               slicnost_indeksa=lambda ind, q=q: korpus[ind] @ upiti[q])
        print(f"k={k:<3} egzaktno na {Q} upita")

    k = 100
    radni = pripremi_pretragu(N, d, k, cfg_t3)
    upiti_gpu = cp.asarray(upiti)
    ind_gpu, vr_gpu = topk_t3(radni["slicnosti"], k, cfg_t3, radni["t3"])
    celina = izmeri_gpu_po_upitima(lambda u: pretrazi_gpu(korpus_gpu, u, k, cfg_s2, cfg_t3, radni),
                                   upiti, warmup=5, ponavljanja=100, zagrevanje_ms=1000)
    delovi = {
        "upit na karticu": izmeri_gpu(lambda: radni["upit"].set(upiti[0]), 5, 100, 1000),
        "S2": izmeri_gpu_po_upitima(lambda u: slicnost_s2(korpus_gpu, u, radni["slicnosti"], cfg_s2),
                                    upiti_gpu, 5, 100, 1000),
        "T3": izmeri_gpu(lambda: topk_t3(radni["slicnosti"], k, cfg_t3, radni["t3"]), 5, 100, 1000),
        "rezultat u RAM": izmeri_gpu(lambda: (cp.asnumpy(ind_gpu), cp.asnumpy(vr_gpu)), 5, 100, 1000),
    }
    print(f"\nN={N}, k={k}: celina {celina.medijana_ms:.4f} ms")
    for naziv, merenje in delovi.items():
        print(f"  {naziv:<16} {merenje.medijana_ms:.4f} ms")
    print(f"  zbir delova       {sum(m.medijana_ms for m in delovi.values()):.4f} ms")
