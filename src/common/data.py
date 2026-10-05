import json
from pathlib import Path

import numpy as np

TIP = np.float32
KOREN = Path(__file__).resolve().parents[2]


def normalizuj(matrica):
    matrica = np.asarray(matrica, dtype=TIP)
    if matrica.ndim == 1:
        norma = np.linalg.norm(matrica)
        if norma == 0.0:
            norma = 1.0
        return (matrica / norma).astype(TIP)

    norme = np.linalg.norm(matrica, axis=1, keepdims=True)
    norme[norme == 0.0] = 1.0
    return (matrica / norme).astype(TIP)


def napravi_korpus(N, d, seed=0):
    generator = np.random.default_rng(seed)
    korpus = generator.standard_normal((N, d), dtype=TIP)
    return normalizuj(korpus)


def napravi_upit(d, seed=1):
    generator = np.random.default_rng(seed)
    upit = generator.standard_normal(d, dtype=TIP)
    return normalizuj(upit)


def napravi_upite(Q, d, seed=1):
    generator = np.random.default_rng(seed)
    upiti = generator.standard_normal((Q, d), dtype=TIP)
    return normalizuj(upiti)


def _proveri_matricu(matrica, naziv, d=None):
    if matrica.dtype != TIP:
        raise ValueError(f"{naziv}: tip {matrica.dtype}, ocekivan {TIP.__name__}")
    if matrica.ndim != 2:
        raise ValueError(f"{naziv}: ocekivana matrica, oblik {matrica.shape}")
    if d is not None and matrica.shape[1] != d:
        raise ValueError(f"{naziv}: d = {matrica.shape[1]}, u config-u je {d}")


def ucitaj_korpus(putanja, N, d=None):
    ceo = np.load(putanja, mmap_mode="r")
    _proveri_matricu(ceo, "korpus", d)
    if N > ceo.shape[0]:
        raise ValueError(f"N = {N}, a fajl ima {ceo.shape[0]} redova")
    return np.ascontiguousarray(ceo[:N])


def ucitaj_upite(putanja, Q, d=None):
    svi = np.load(putanja)
    _proveri_matricu(svi, "upiti", d)
    if Q > svi.shape[0]:
        raise ValueError(f"Q = {Q}, a fajl ima {svi.shape[0]} upita")
    return np.ascontiguousarray(svi[:Q])


def ucitaj_podatke(konfiguracija, N):
    podaci = konfiguracija["podaci"]
    putanje = konfiguracija["putanje"]
    d, Q = podaci["d"], podaci["broj_upita"]

    if podaci["izvor"] == "msmarco":
        korpus = ucitaj_korpus(KOREN / putanje["korpus"], N, d)
        upiti = ucitaj_upite(KOREN / putanje["upiti"], Q, d)
        with open(KOREN / putanje["opis"], "r", encoding="utf-8") as fajl:
            opis_fajla = json.load(fajl)
        opis = {"izvor": "msmarco", "model": opis_fajla["model"]}
    elif podaci["izvor"] == "sinteticki":
        korpus = napravi_korpus(N, d, seed=podaci["seed_korpus"])
        upiti = napravi_upite(Q, d, seed=podaci["seed_upit"])
        opis = {"izvor": "sinteticki", "model": "nema"}
    else:
        raise ValueError(f"nepoznat izvor podataka: {podaci['izvor']}")

    return korpus, upiti, opis


def zauzece_u_gb(N, d):
    return N * d * np.dtype(TIP).itemsize / 1e9


if __name__ == "__main__":
    import yaml

    with open(KOREN / "bench" / "config.yaml", "r", encoding="utf-8") as fajl:
        konfiguracija = yaml.safe_load(fajl)

    for izvor in ("sinteticki", "msmarco"):
        konfiguracija["podaci"]["izvor"] = izvor
        for N in konfiguracija["podaci"]["N"]:
            korpus, upiti, opis = ucitaj_podatke(konfiguracija, N)
            norme_k = np.linalg.norm(korpus, axis=1)
            norme_u = np.linalg.norm(upiti, axis=1)
            print(f"{izvor:<10} N={N:<8} korpus {korpus.shape} {korpus.dtype}, "
                  f"upiti {upiti.shape}, "
                  f"norme korpusa [{norme_k.min():.7f}, {norme_k.max():.7f}], "
                  f"norme upita [{norme_u.min():.7f}, {norme_u.max():.7f}], "
                  f"{zauzece_u_gb(*korpus.shape):.2f} GB, model {opis['model']}")
            del korpus
