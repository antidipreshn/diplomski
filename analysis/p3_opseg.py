import argparse
import csv
from pathlib import Path

N, D = 1_000_000, 768
K = "10"

METODE = {
    "cpu_numpy": "cpu", "cpu_faiss_flat": "cpu",
    "gpu_s1": "gpu", "gpu_s2": "gpu", "gpu_s3": "gpu", "gpu_s4": "gpu", "gpu_s5": "gpu",
    "gpu_pretraga": "gpu", "gpu_torch_kompletna": "gpu",
}


def procitaj(putanja):
    with open(putanja, encoding="utf-8") as fajl:
        return list(csv.DictReader(fajl))


def okruzenje(putanja_csv):
    podaci = {}
    with open(Path(putanja_csv).with_suffix(".okruzenje.txt"), encoding="utf-8") as fajl:
        for red in fajl:
            kljuc, _, vrednost = red.partition(": ")
            podaci[kljuc] = vrednost.strip()
    return podaci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opseg", required=True, help="CSV sa redovima propusni_opseg i propusni_opseg_cpu")
    parser.add_argument("--rezultati", nargs="+", required=True, help="CSV-ovi sa merenjima pretrage")
    parser.add_argument("--izlaz", default=str(Path(__file__).parent / "p3_opseg.csv"))
    argumenti = parser.parse_args()

    redovi_opsega = {r["metoda"]: r for r in procitaj(argumenti.opseg)}
    okr = okruzenje(argumenti.opseg)
    granice = {
        "gpu": (float(redovi_opsega["propusni_opseg"]["propusni_opseg_gbps"]),
                float(okr["teorijski_opseg_maks_takt_gbps"])),
        "cpu": (float(redovi_opsega["propusni_opseg_cpu"]["propusni_opseg_gbps"]),
                float(okr["teorijski_opseg_cpu_gbps"].split()[0])),
    }

    nadjeni = {}
    for putanja in argumenti.rezultati:
        for red in procitaj(putanja):
            m = red["metoda"]
            if m in METODE and m not in nadjeni and red["N"] == str(N) and red["k"] in ("", K):
                nadjeni[m] = (red, Path(putanja).name)

    bajtova = N * D * 4
    tabela = []
    for metoda, uredjaj in METODE.items():
        if metoda not in nadjeni:
            print(f"nema reda za {metoda}")
            continue
        red, izvor = nadjeni[metoda]
        medijana = float(red["medijana_ms"])
        gbps = bajtova / (medijana * 1e-3) / 1e9
        izmerena, teorijska = granice[uredjaj]
        tabela.append({
            "metoda": metoda, "uredjaj": uredjaj, "k": red["k"], "medijana_ms": f"{medijana:.4f}",
            "gbps": f"{gbps:.1f}",
            "granica_izmerena_gbps": f"{izmerena:.1f}", "procenat_izmerene": f"{100 * gbps / izmerena:.1f}",
            "granica_teorijska_gbps": f"{teorijska:.1f}", "procenat_teorijske": f"{100 * gbps / teorijska:.1f}",
            "izvor": izvor,
        })

    with open(argumenti.izlaz, "w", newline="", encoding="utf-8") as fajl:
        pisac = csv.DictWriter(fajl, fieldnames=list(tabela[0]))
        pisac.writeheader()
        pisac.writerows(tabela)

    print(f"granice: kartica {granice['gpu'][0]:.1f} izmereno / {granice['gpu'][1]:.1f} teorijski GB/s; "
          f"procesor {granice['cpu'][0]:.1f} / {granice['cpu'][1]:.1f} GB/s")
    print(f"{'metoda':<22}{'ms':>10}{'GB/s':>9}{'% izmer.':>10}{'% teor.':>9}")
    for r in tabela:
        print(f"{r['metoda']:<22}{r['medijana_ms']:>10}{r['gbps']:>9}{r['procenat_izmerene']:>10}{r['procenat_teorijske']:>9}")
    print(f"\nzapisano u {argumenti.izlaz}")


if __name__ == "__main__":
    main()
