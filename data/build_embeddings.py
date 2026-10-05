import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-mpnet-base-v2"
PODACI = Path(__file__).resolve().parent
ULAZ = PODACI / "msmarco"
IZLAZ = PODACI / "embeddings"


def ugradi_u_fajl(model, tekstovi, putanja, velicina_dela, velicina_serije):
    d = model.encode(["proba"], convert_to_numpy=True).shape[1]
    N = len(tekstovi)
    matrica = np.lib.format.open_memmap(putanja, mode="w+", dtype=np.float32, shape=(N, d))

    pocetak = time.perf_counter()
    for od in range(0, N, velicina_dela):
        do = min(od + velicina_dela, N)
        vektori = model.encode(
            tekstovi[od:do],
            batch_size=velicina_serije,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrica[od:do] = vektori.astype(np.float32)

        proteklo = time.perf_counter() - pocetak
        print(f"  {do}/{N}  ({do / proteklo:.0f} tekstova/s, {proteklo / 60:.1f} min)")

    matrica.flush()
    del matrica
    return d, time.perf_counter() - pocetak


def proveri(putanja):
    matrica = np.load(putanja, mmap_mode="r")
    uzorak = np.asarray(matrica[: min(len(matrica), 100_000)])
    norme = np.linalg.norm(uzorak, axis=1)
    rezultat = {
        "oblik": list(matrica.shape),
        "tip": str(matrica.dtype),
        "ima_nan": bool(np.isnan(uzorak).any()),
        "norma_min": float(norme.min()),
        "norma_max": float(norme.max()),
    }
    print(f"  {putanja.name}: {rezultat}")
    return rezultat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broj-pasusa", type=int, default=1_000_000)
    parser.add_argument("--broj-upita", type=int, default=1000)
    parser.add_argument("--seed-upita", type=int, default=42)
    parser.add_argument("--velicina-serije", type=int, default=256)
    parser.add_argument("--velicina-dela", type=int, default=50_000)
    argumenti = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ne vidi karticu")

    IZLAZ.mkdir(parents=True, exist_ok=True)
    N = argumenti.broj_pasusa

    print("ucitavanje teksta...")
    pasusi = pd.read_parquet(ULAZ / "pasusi.parquet").iloc[:N]
    upiti = pd.read_parquet(ULAZ / "upiti.parquet")

    generator = np.random.default_rng(argumenti.seed_upita)
    izbor = generator.choice(len(upiti), size=argumenti.broj_upita, replace=False)
    izabrani_upiti = upiti.iloc[np.sort(izbor)].reset_index(drop=True)

    print("ucitavanje modela...")
    model = SentenceTransformer(MODEL, device="cuda")
    print("  max_seq_length:", model.max_seq_length)

    putanja_pasusa = IZLAZ / f"pasusi_N{N}.npy"
    putanja_upita = IZLAZ / f"upiti_Q{argumenti.broj_upita}.npy"

    print(f"pasusi ({N})...")
    d, vreme_pasusa = ugradi_u_fajl(
        model, pasusi["text"].tolist(), putanja_pasusa,
        argumenti.velicina_dela, argumenti.velicina_serije,
    )

    print(f"upiti ({argumenti.broj_upita})...")
    _, vreme_upita = ugradi_u_fajl(
        model, izabrani_upiti["text"].tolist(), putanja_upita,
        argumenti.velicina_dela, argumenti.velicina_serije,
    )

    np.save(IZLAZ / f"pasusi_N{N}_pid.npy", pasusi["pid"].to_numpy())
    izabrani_upiti.to_parquet(IZLAZ / f"upiti_Q{argumenti.broj_upita}.parquet", index=False)

    print("provera...")
    provera_pasusa = proveri(putanja_pasusa)
    provera_upita = proveri(putanja_upita)

    opis = {
        "model": MODEL,
        "d": d,
        "max_seq_length": model.max_seq_length,
        "normalizovano": True,
        "tip": "float32",
        "broj_pasusa": N,
        "izbor_pasusa": "prvih N u redosledu skupa",
        "broj_upita": argumenti.broj_upita,
        "izbor_upita": "nasumicno bez ponavljanja iz svih upita",
        "seed_upita": argumenti.seed_upita,
        "velicina_serije": argumenti.velicina_serije,
        "vreme_pasusa_min": round(vreme_pasusa / 60, 2),
        "vreme_upita_s": round(vreme_upita, 2),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "python": platform.python_version(),
        "napravljeno": datetime.now().isoformat(timespec="seconds"),
        "provera_pasusa": provera_pasusa,
        "provera_upita": provera_upita,
    }
    with open(IZLAZ / f"opis_N{N}_Q{argumenti.broj_upita}.json", "w", encoding="utf-8") as fajl:
        json.dump(opis, fajl, indent=2, ensure_ascii=False)

    print()
    print(f"gotovo: d = {d}, pasusi za {vreme_pasusa / 60:.1f} min")
    print("snimljeno u:", IZLAZ)


if __name__ == "__main__":
    main()
