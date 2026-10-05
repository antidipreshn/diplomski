import time

import faiss


def napravi_flat(korpus):
    pocetak = time.perf_counter()
    indeks = faiss.IndexFlatIP(korpus.shape[1])
    indeks.add(korpus)
    return indeks, time.perf_counter() - pocetak


def pretrazi_flat(indeks, upit, k):
    vrednosti, indeksi = indeks.search(upit.reshape(1, -1), k)
    return indeksi[0], vrednosti[0]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.common.data import napravi_korpus, napravi_upite
    from src.common.timing import izmeri_cpu_po_upitima
    from src.common.verify import proveri_egzaktnost
    from src.cpu.numpy_ref import pretrazi_numpy

    N, d, Q = 100_000, 768, 20
    korpus = napravi_korpus(N, d, seed=0)
    upiti = napravi_upite(Q, d, seed=1)

    faiss.omp_set_num_threads(8)
    indeks, vreme_s = napravi_flat(korpus)
    print(f"IndexFlatIP N={N}: izgradnja {vreme_s * 1e3:.1f} ms")

    for k in (1, 10, 100):
        for q in range(Q):
            indeksi, vrednosti = pretrazi_flat(indeks, upiti[q], k)
            ref_ind, ref_vr = pretrazi_numpy(korpus, upiti[q], k)
            proveri_egzaktnost(indeksi, vrednosti, ref_ind, ref_vr,
                               naziv=f"faiss_flat (upit {q}, k={k})",
                               slicnost_indeksa=lambda ind, q=q: korpus[ind] @ upiti[q])
        print(f"k={k:<3} egzaktno na {Q} upita")

    print("\njedan upit, 1 nit naspram 8 niti (k=10):")
    for niti in (1, 8):
        faiss.omp_set_num_threads(niti)
        print(f"  niti={niti}: {izmeri_cpu_po_upitima(lambda u: pretrazi_flat(indeks, u, 10), upiti, 5, 100)}")
