import platform
from datetime import datetime
from pathlib import Path

import datasets
import pandas as pd
from datasets import load_dataset

SKUP = "sentence-transformers/msmarco-corpus"
BROJ_PASUSA = 1_000_000
IZLAZ = Path(__file__).resolve().parent / "msmarco"


def preuzmi(konfiguracija, broj=None):
    tok = load_dataset(SKUP, konfiguracija, split="train", streaming=True)

    redovi = []
    for i, red in enumerate(tok):
        if broj is not None and i >= broj:
            break
        redovi.append(red)
        if (i + 1) % 100_000 == 0:
            print(f"  {konfiguracija}: {i + 1} redova")

    return pd.DataFrame(redovi)


def main():
    IZLAZ.mkdir(parents=True, exist_ok=True)

    print("preuzimanje pasusa...")
    pasusi = preuzmi("passage", BROJ_PASUSA)
    print("kolone:", list(pasusi.columns))
    pasusi.to_parquet(IZLAZ / "pasusi.parquet", index=False)

    print("preuzimanje upita...")
    upiti = preuzmi("query")
    print("kolone:", list(upiti.columns))
    upiti.to_parquet(IZLAZ / "upiti.parquet", index=False)

    with open(IZLAZ / "izvor.txt", "w", encoding="utf-8") as fajl:
        fajl.write(f"skup: {SKUP}\n")
        fajl.write(f"pasusa: {len(pasusi)} (prvih {BROJ_PASUSA} u redosledu skupa)\n")
        fajl.write(f"upita: {len(upiti)} (svi)\n")
        fajl.write(f"preuzeto: {datetime.now().isoformat(timespec='seconds')}\n")
        fajl.write(f"datasets: {datasets.__version__}\n")
        fajl.write(f"python: {platform.python_version()}\n")

    print()
    print("pasusa:", len(pasusi))
    print("upita:", len(upiti))
    print("primer pasusa:", pasusi.iloc[0].to_dict())
    print("primer upita:", upiti.iloc[0].to_dict())
    print("snimljeno u:", IZLAZ)


if __name__ == "__main__":
    main()
