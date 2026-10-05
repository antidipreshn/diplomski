import json
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

KORAK_DODAVANJA = 100_000


def putanja_indeksa(direktorijum, opis, N, d, M, ef_construction):
    return Path(direktorijum) / f"hnsw_{opis['izvor']}_N{N}_d{d}_M{M}_efc{ef_construction}.faiss"


def _proveri_indeks(indeks, korpus, M, ef_construction, putanja):
    N, d = korpus.shape
    greske = []
    if indeks.ntotal != N or indeks.d != d:
        greske.append(f"oblik {indeks.ntotal} x {indeks.d}, ocekivano {N} x {d}")
    if indeks.metric_type != faiss.METRIC_INNER_PRODUCT:
        greske.append("metrika nije unutrasnji proizvod")
    if indeks.hnsw.nb_neighbors(1) != M:
        greske.append(f"M = {indeks.hnsw.nb_neighbors(1)}, ocekivano {M}")
    if indeks.hnsw.efConstruction != ef_construction:
        greske.append(f"efConstruction = {indeks.hnsw.efConstruction}, ocekivano {ef_construction}")
    if not greske:
        for i in (0, N // 2, N - 1):
            if not np.array_equal(indeks.reconstruct(i), korpus[i]):
                greske.append(f"vektor {i} se razlikuje od korpusa")
                break
    if greske:
        raise ValueError(f"{Path(putanja).name}: {'; '.join(greske)}; obrisati fajl")


def napravi_ili_ucitaj_indeks(korpus, M, ef_construction, putanja):
    putanja = Path(putanja)
    putanja_opisa = putanja.with_suffix(".json")

    if putanja.exists() and putanja_opisa.exists():
        indeks = faiss.read_index(str(putanja))
        _proveri_indeks(indeks, korpus, M, ef_construction, putanja)
        with open(putanja_opisa, "r", encoding="utf-8") as fajl:
            izgradnja = json.load(fajl)
        print(f"  HNSW indeks ucitan sa diska: {putanja.name}")
        return indeks, izgradnja

    N, d = korpus.shape
    indeks = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    indeks.hnsw.efConstruction = ef_construction

    print(f"  gradim HNSW indeks: N={N}, M={M}, efConstruction={ef_construction}, "
          f"niti={faiss.omp_get_max_threads()}")
    pocetak = time.perf_counter()
    for od in range(0, N, KORAK_DODAVANJA):
        indeks.add(korpus[od:od + KORAK_DODAVANJA])
        if N > KORAK_DODAVANJA:
            print(f"    {indeks.ntotal}/{N}  {time.perf_counter() - pocetak:.0f} s")
    vreme_s = time.perf_counter() - pocetak

    izgradnja = {
        "vreme_izgradnje_s": vreme_s,
        "niti": faiss.omp_get_max_threads(),
        "N": N, "d": d, "M": M, "ef_construction": ef_construction,
        "faiss": faiss.__version__,
        "napravljeno": datetime.now().isoformat(timespec="seconds"),
    }
    putanja.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indeks, str(putanja))
    with open(putanja_opisa, "w", encoding="utf-8") as fajl:
        json.dump(izgradnja, fajl, indent=2)
    print(f"  HNSW indeks napravljen za {vreme_s:.1f} s i snimljen: {putanja.name}")
    return indeks, izgradnja


def pretrazi_hnsw(indeks, upit, k):
    vrednosti, indeksi = indeks.search(upit.reshape(1, -1), k)
    return indeksi[0], vrednosti[0]


if __name__ == "__main__":
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.common.data import napravi_korpus, napravi_upite
    from src.common.timing import izmeri_cpu_po_upitima
    from src.common.verify import recall_aproksimativno
    from src.cpu.numpy_ref import pretrazi_numpy

    N, d, Q, k, M, efc = 10_000, 768, 20, 10, 32, 200
    korpus = napravi_korpus(N, d, seed=0)
    upiti = napravi_upite(Q, d, seed=1)
    tacni = [pretrazi_numpy(korpus, upiti[q], k) for q in range(Q)]
    opis = {"izvor": "sinteticki"}

    def srednji_recall(indeks):
        return float(np.mean([
            recall_aproksimativno(pretrazi_hnsw(indeks, upiti[q], k)[0], tacni[q][0], tacni[q][1],
                                  lambda ind, q=q: korpus[ind] @ upiti[q])[0]
            for q in range(Q)]))

    faiss.omp_set_num_threads(8)
    with tempfile.TemporaryDirectory() as privremeni:
        putanja = putanja_indeksa(privremeni, opis, N, d, M, efc)
        prvi, izgradnja = napravi_ili_ucitaj_indeks(korpus, M, efc, putanja)
        drugi, _ = napravi_ili_ucitaj_indeks(korpus, M, efc, putanja)

        print("\nrecall@10 po efSearch (sinteticki podaci su tezi od pravih embedinga):")
        for ef in (16, 32, 64, 128, 256, 512):
            prvi.hnsw.efSearch = drugi.hnsw.efSearch = ef
            print(f"  efSearch={ef:<4} recall={srednji_recall(prvi):.4f}")

        prvi.hnsw.efSearch = drugi.hnsw.efSearch = 64
        isti = all(np.array_equal(pretrazi_hnsw(prvi, upiti[q], k)[0],
                                  pretrazi_hnsw(drugi, upiti[q], k)[0]) for q in range(Q))
        print("ucitan indeks daje iste rezultate:", isti)

        try:
            napravi_ili_ucitaj_indeks(korpus, 16, efc, putanja.with_name(putanja.name))
            print("GRESKA: pogresan M nije primecen")
        except ValueError as greska:
            print("pogresan M odbijen kako treba:", greska)

        print("\njedan upit, 1 nit naspram 8 niti (efSearch=64):")
        for niti in (1, 8):
            faiss.omp_set_num_threads(niti)
            print(f"  niti={niti}: {izmeri_cpu_po_upitima(lambda u: pretrazi_hnsw(prvi, u, k), upiti, 5, 100)}")
        del prvi, drugi
