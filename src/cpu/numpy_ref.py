from datetime import datetime
from pathlib import Path

import numpy as np


def slicnosti_numpy(korpus, upit):
    return korpus @ upit


def top_k_numpy(slicnosti, k):
    N = slicnosti.shape[0]
    if k > N:
        raise ValueError(f"k = {k} je vece od broja vektora N = {N}")

    kandidati = np.argpartition(-slicnosti, k - 1)[:k]
    poredak = np.argsort(-slicnosti[kandidati], kind="stable")
    indeksi = kandidati[poredak]

    return indeksi.astype(np.int64), slicnosti[indeksi].astype(np.float32)

def pretrazi_numpy(korpus, upit, k):
    return top_k_numpy(slicnosti_numpy(korpus, upit), k)


def top_k_numpy_redosled(slicnosti, k):
    N = slicnosti.shape[0]
    if k > N:
        raise ValueError(f"k = {k} je vece od broja vektora N = {N}")

    prag = np.partition(slicnosti, N - k)[N - k]
    kandidati = np.nonzero(slicnosti >= prag)[0]
    poredak = np.lexsort((kandidati, -slicnosti[kandidati]))
    indeksi = kandidati[poredak][:k]

    return indeksi.astype(np.int64), slicnosti[indeksi].astype(np.float32)


def snimi_ground_truth(putanja, indeksi, vrednosti, parametri):
    putanja = Path(putanja)
    putanja.parent.mkdir(parents=True, exist_ok=True)
    np.savez(putanja, indeksi=indeksi, vrednosti=vrednosti, **parametri)
    return putanja


def ucitaj_ground_truth(putanja):
    with np.load(putanja) as podaci:
        indeksi = podaci["indeksi"]
        vrednosti = podaci["vrednosti"]
        parametri = {kljuc: podaci[kljuc].item()
                     for kljuc in podaci.files if kljuc not in ("indeksi", "vrednosti")}
    return indeksi, vrednosti, parametri


def ground_truth(korpus, upiti, k_max, direktorijum, opis):
    N, d = korpus.shape
    Q = upiti.shape[0]
    ocekivano = {"izvor": opis["izvor"], "model": opis["model"],
                 "N": N, "d": d, "Q": Q, "k_max": k_max}
    putanja = Path(direktorijum) / f"gt_{opis['izvor']}_N{N}_Q{Q}_k{k_max}.npz"

    if putanja.exists():
        indeksi, vrednosti, parametri = ucitaj_ground_truth(putanja)
        razlike = {kljuc: (parametri.get(kljuc), vrednost)
                   for kljuc, vrednost in ocekivano.items()
                   if parametri.get(kljuc) != vrednost}
        if razlike:
            raise ValueError(f"{putanja.name}: parametri se ne poklapaju {razlike}; obrisati fajl")
        print(f"  ground truth ucitan iz kesa: {putanja.name}")
        return indeksi, vrednosti

    indeksi = np.empty((Q, k_max), dtype=np.int64)
    vrednosti = np.empty((Q, k_max), dtype=np.float32)
    for q in range(Q):
        indeksi[q], vrednosti[q] = pretrazi_numpy(korpus, upiti[q], k_max)

    snimi_ground_truth(putanja, indeksi, vrednosti,
                       {**ocekivano, "napravljeno": datetime.now().isoformat(timespec="seconds")})
    print(f"  ground truth izracunat i snimljen: {putanja.name}")
    return indeksi, vrednosti


if __name__ == "__main__":
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.common.data import napravi_korpus, napravi_upite

    N, d, Q, k_max = 100_000, 768, 5, 100
    korpus = napravi_korpus(N, d, seed=0)
    upiti = napravi_upite(Q, d, seed=1)
    opis = {"izvor": "sinteticki", "model": "nema"}

    with tempfile.TemporaryDirectory() as privremeni:
        prvi = ground_truth(korpus, upiti, k_max, privremeni, opis)
        drugi = ground_truth(korpus, upiti, k_max, privremeni, opis)

    print("oblik:", prvi[0].shape)
    print("kes daje isti rezultat:",
          np.array_equal(prvi[0], drugi[0]) and np.array_equal(prvi[1], drugi[1]))
    print("prefiks [:10] jednak top-10:",
          np.array_equal(prvi[0][0, :10], pretrazi_numpy(korpus, upiti[0], 10)[0]))
    print("sortirano opadajuce:", bool(np.all(np.diff(prvi[1], axis=1) <= 0)))
