import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

KOREN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOREN))

from src.common.data import ucitaj_podatke
from src.common.timing import (gpu_je_dostupan, izmeri_cpu_po_upitima, izmeri_gpu,
                               izmeri_gpu_po_upitima, podaci_o_okruzenju,
                               postavi_niti_procesora, stanje_kartice)
from src.common.verify import proveri_egzaktnost, recall_aproksimativno
from src.cpu.naive import pretrazi_naivno
from src.cpu.numpy_ref import ground_truth, pretrazi_numpy

KOLONE = [
    "metoda", "serija", "uredjaj", "izvor", "N", "d", "k", "ef_search", "broj_upita",
    "medijana_ms", "min_ms", "max_ms", "ponavljanja", "warmup",
    "recall", "zamene", "max_razlika", "kandidati", "propusni_opseg_gbps", "procenat_opsega",
    "udeo_pretrage",
    "stanje_gpu",
    "vreme_pokretanja",
]


def pretrazi_razvoj_cupy(korpus, upit, k):
    import cupy as cp
    slicnosti = korpus @ upit
    indeksi = cp.argsort(-slicnosti)[:k]
    return indeksi, slicnosti[indeksi]


METODE = {
    "cpu_numpy": (pretrazi_numpy, "cpu"),
    "cpu_naivno": (pretrazi_naivno, "cpu"),
    "razvoj_cupy": (pretrazi_razvoj_cupy, "gpu"),
    "cpu_hnsw": (None, "cpu"),
    "cpu_faiss_flat": (None, "cpu"),
    "gpu_torch": (None, "gpu"),
}

SLICNOSTI = {"gpu_s1": "s1", "gpu_s2": "s2", "gpu_s3": "s3", "gpu_s4": "s4", "gpu_s5": "s5"}


TOPK = {"gpu_t1": "t1", "gpu_t2": "t2", "gpu_t3": "t3", "gpu_t4": "t4"}

KOMPLETNE = {"gpu_pretraga": ["s2", "t3"], "gpu_torch_kompletna": []}
TORCH_METODE = {"gpu_torch", "gpu_torch_kompletna"}


def uredjaj_metode(naziv):
    if naziv in SLICNOSTI or naziv in TOPK or naziv in KOMPLETNE:
        return "gpu"
    return METODE[naziv][1]


def sve_metode(konfiguracija):
    prelomna = konfiguracija.get("prelomna_tacka") or {}
    return list(konfiguracija["merenje"]["metode"]) + list(prelomna.get("metode", []))


def ucitaj_konfiguraciju(putanja):
    with open(putanja, "r", encoding="utf-8") as fajl:
        konfiguracija = yaml.safe_load(fajl)

    merenje = konfiguracija["merenje"]
    if merenje["warmup"] < 5 or merenje["ponavljanja"] < 20:
        raise ValueError("invarijante 3 i 4: warmup >= 5, ponavljanja >= 20")
    metode = sve_metode(konfiguracija)
    nepoznate = set(metode) - set(METODE) - set(SLICNOSTI) - set(TOPK) - set(KOMPLETNE)
    if nepoznate:
        raise ValueError(f"nepoznate metode u config.yaml: {sorted(nepoznate)}")
    potrebni_kerneli = {SLICNOSTI.get(m) or TOPK.get(m) for m in metode} - {None}
    if set(metode) & set(TOPK):
        potrebni_kerneli.add("s2")
    for m in set(metode) & set(KOMPLETNE):
        potrebni_kerneli.update(KOMPLETNE[m])
    bez_parametara = sorted(kljuc for kljuc in potrebni_kerneli
                            if kljuc not in konfiguracija.get("kerneli", {}))
    if bez_parametara:
        raise ValueError(f"nema parametara u odeljku kerneli za: {bez_parametara}")
    return konfiguracija


def pokreni_pracenje_takta(konfiguracija, putanja):
    pracenje = konfiguracija["pracenje_takta"]
    if not pracenje["ukljuceno"]:
        return None
    fajl = open(putanja, "w", encoding="utf-8")
    proces = subprocess.Popen([
        "nvidia-smi",
        "--query-gpu=timestamp,pstate,clocks.gr,clocks.mem,temperature.gpu,"
        "power.draw,utilization.gpu,clocks_throttle_reasons.active",
        "--format=csv", "-lms", str(pracenje["interval_ms"]),
    ], stdout=fajl)
    return proces, fajl


class Rezultati:
    def __init__(self, putanja):
        self.fajl = open(putanja, "w", newline="", encoding="utf-8")
        self.pisac = csv.DictWriter(self.fajl, fieldnames=KOLONE)
        self.pisac.writeheader()
        self.broj = 0

    def dodaj(self, merenje=None, **polja):
        red = dict.fromkeys(KOLONE, "")
        if merenje is not None:
            red.update({
                "uredjaj": merenje.uredjaj,
                "medijana_ms": round(merenje.medijana_ms, 6),
                "min_ms": round(merenje.min_ms, 6),
                "max_ms": round(merenje.max_ms, 6),
                "ponavljanja": merenje.ponavljanja,
                "warmup": merenje.warmup,
            })
            if merenje.uredjaj == "gpu":
                red["stanje_gpu"] = stanje_kartice()
        red.update(polja)
        red["vreme_pokretanja"] = datetime.now().isoformat(timespec="seconds")
        self.pisac.writerow(red)
        self.fajl.flush()
        self.broj += 1

    def zatvori(self):
        self.fajl.close()


def izmeri_propusni_opseg(konfiguracija, rezultati, okruzenje):
    from bench.bandwidth import izmeri_opseg_gbps

    opseg = izmeri_opseg_gbps(**konfiguracija["propusni_opseg"])
    rezultati.dodaj(opseg["merenje"], metoda="propusni_opseg",
                    propusni_opseg_gbps=round(opseg["izmereni_gbps"], 2))
    okruzenje["teorijski_opseg_maks_takt_gbps"] = round(opseg["teorijski_maks_gbps"], 1)
    okruzenje["teorijski_opseg_pri_taktu_gbps"] = round(opseg["teorijski_pri_taktu_gbps"], 1)
    okruzenje["takt_memorije_pod_opterecenjem"] = f'{opseg["takt_memorije_mhz"]:.0f} MHz ({opseg["stanje"]})'
    print(f"  propusni opseg: {opseg['izmereni_gbps']:.1f} GB/s  {opseg['merenje']}")
    return opseg["izmereni_gbps"]


def izmeri_propusni_opseg_cpu(konfiguracija, rezultati, okruzenje):
    from bench.bandwidth import izmeri_opseg_cpu_gbps, teorijski_opseg_cpu_gbps

    cpu = konfiguracija["propusni_opseg_cpu"]
    opseg = izmeri_opseg_cpu_gbps(cpu["velicina_mb"], konfiguracija["merenje"]["niti_procesora"],
                                  cpu["warmup"], cpu["ponavljanja"])
    rezultati.dodaj(opseg["merenje"], metoda="propusni_opseg_cpu",
                    propusni_opseg_gbps=round(opseg["izmereni_gbps"], 2))
    teorijski, opis_ram = teorijski_opseg_cpu_gbps()
    okruzenje["teorijski_opseg_cpu_gbps"] = f"{teorijski:.1f} ({opis_ram})"
    print(f"  propusni opseg procesora: {opseg['izmereni_gbps']:.1f} GB/s  {opseg['merenje']}")


def izmeri_prenos_korpusa(korpus, konfiguracija, rezultati, osnova):
    import cupy as cp

    prenos = konfiguracija["prenos_korpusa"]
    merenje = izmeri_gpu(lambda: cp.asarray(korpus), warmup=prenos["warmup"],
                         ponavljanja=prenos["ponavljanja"], zagrevanje_ms=prenos["zagrevanje_ms"])
    cp.get_default_memory_pool().free_all_blocks()
    opseg = merenje.propusni_opseg_gbps(korpus.nbytes)
    rezultati.dodaj(merenje, metoda="prenos_korpusa",
                    propusni_opseg_gbps=round(opseg, 2), **osnova)
    print(f"  prenos korpusa: {opseg:.1f} GB/s  {merenje}")


def verifikuj(funkcija, uredjaj, korpus_x, upiti_x, korpus, upiti, gt_ind, gt_vr, k,
              tolerancija, naziv, broj_upita):
    import numpy as np

    razlike, recall_vrednosti, zamene = [], [], 0
    for q in range(broj_upita):
        indeksi, vrednosti = funkcija(korpus_x, upiti_x[q], k)
        if uredjaj == "gpu":
            import cupy as cp
            indeksi, vrednosti = cp.asnumpy(indeksi), cp.asnumpy(vrednosti)

        razlika, recall, broj_zamena = proveri_egzaktnost(
            indeksi, vrednosti, gt_ind[q, :k], gt_vr[q, :k],
            tolerancija=tolerancija, naziv=f"{naziv} (upit {q}, k={k})",
            slicnost_indeksa=lambda ind, q=q: korpus[ind] @ upiti[q],
        )
        razlike.append(razlika)
        recall_vrednosti.append(recall)
        zamene += broj_zamena

    return max(razlike), float(np.mean(recall_vrednosti)), zamene


def izmeri_hnsw(konfiguracija, korpus, upiti, gt_ind, gt_vr, opis, rezultati, osnova):
    from src.baselines.faiss_hnsw import (napravi_ili_ucitaj_indeks, pretrazi_hnsw,
                                          putanja_indeksa)

    hnsw = konfiguracija["hnsw"]
    merenje_cfg = konfiguracija["merenje"]
    N, d = korpus.shape
    putanja = putanja_indeksa(KOREN / konfiguracija["putanje"]["indeksi"], opis, N, d,
                              hnsw["M"], hnsw["ef_construction"])
    indeks, izgradnja = napravi_ili_ucitaj_indeks(korpus, hnsw["M"], hnsw["ef_construction"], putanja)

    vreme_ms = izgradnja["vreme_izgradnje_s"] * 1e3
    rezultati.dodaj(metoda="hnsw_izgradnja", uredjaj="cpu", medijana_ms=round(vreme_ms, 3),
                    min_ms=round(vreme_ms, 3), max_ms=round(vreme_ms, 3),
                    ponavljanja=1, warmup=0, **osnova)
    print(f"  hnsw_izgradnja {izgradnja['vreme_izgradnje_s']:.1f} s "
          f"(niti={izgradnja['niti']}, {izgradnja['napravljeno']})")

    Q = upiti.shape[0]
    for ef in hnsw["ef_search"]:
        indeks.hnsw.efSearch = ef
        for k in konfiguracija["podaci"]["k"]:
            if ef < k:
                continue
            recall_vrednosti, zamene = [], 0
            for q in range(Q):
                indeksi, _ = pretrazi_hnsw(indeks, upiti[q], k)
                recall, broj_zamena = recall_aproksimativno(
                    indeksi, gt_ind[q, :k], gt_vr[q, :k],
                    lambda ind, q=q: korpus[ind] @ upiti[q], merenje_cfg["tolerancija"])
                recall_vrednosti.append(recall)
                zamene += broj_zamena
            recall = float(np.mean(recall_vrednosti))

            merenje = izmeri_cpu_po_upitima(
                lambda upit: pretrazi_hnsw(indeks, upit, k), upiti,
                warmup=merenje_cfg["warmup"], ponavljanja=merenje_cfg["ponavljanja"])
            rezultati.dodaj(merenje, metoda="cpu_hnsw", k=k, ef_search=ef,
                            recall=round(recall, 6), zamene=zamene, **osnova)
            print(f"  cpu_hnsw ef={ef:<3} k={k:<3} {merenje}  recall={recall:.4f} zamene={zamene}")

    del indeks


def izmeri_faiss_flat(konfiguracija, korpus, upiti, gt_ind, gt_vr, rezultati, osnova):
    from src.baselines.faiss_flat import napravi_flat, pretrazi_flat

    merenje_cfg = konfiguracija["merenje"]
    indeks, vreme_s = napravi_flat(korpus)
    rezultati.dodaj(metoda="faiss_flat_izgradnja", uredjaj="cpu", medijana_ms=round(vreme_s * 1e3, 3),
                    min_ms=round(vreme_s * 1e3, 3), max_ms=round(vreme_s * 1e3, 3),
                    ponavljanja=1, warmup=0, **osnova)
    print(f"  faiss_flat_izgradnja {vreme_s * 1e3:.1f} ms")

    def funkcija(_korpus, upit, k):
        return pretrazi_flat(indeks, upit, k)

    for k in konfiguracija["podaci"]["k"]:
        razlika, recall, zamene = verifikuj(
            funkcija, "cpu", korpus, upiti, korpus, upiti, gt_ind, gt_vr, k,
            merenje_cfg["tolerancija"], "cpu_faiss_flat", upiti.shape[0])
        merenje = izmeri_cpu_po_upitima(
            lambda upit: pretrazi_flat(indeks, upit, k), upiti,
            warmup=merenje_cfg["warmup"], ponavljanja=merenje_cfg["ponavljanja"])
        rezultati.dodaj(merenje, metoda="cpu_faiss_flat", k=k, recall=round(recall, 6),
                        zamene=zamene, max_razlika=f"{razlika:.3e}", **osnova)
        print(f"  {'cpu_faiss_flat':<12} k={k:<3} {merenje}  recall={recall:.4f} zamene={zamene}")

    del indeks


def izmeri_kompletnu(konfiguracija, naziv, korpus, upiti, korpus_gpu, gt_ind, gt_vr,
                     rezultati, osnova):
    merenje_cfg = konfiguracija["merenje"]
    N, d = korpus.shape

    for k in konfiguracija["podaci"]["k"]:
        if naziv == "gpu_pretraga":
            from src.cuda.pretraga import pripremi_pretragu, pretrazi_gpu
            cfg_s2, cfg_t3 = konfiguracija["kerneli"]["s2"], konfiguracija["kerneli"]["t3"]
            radni = pripremi_pretragu(N, d, k, cfg_t3)

            def funkcija(korpus_x, upit, k):
                return pretrazi_gpu(korpus_x, upit, k, cfg_s2, cfg_t3, radni)
        else:
            from src.baselines.torch_ref import pretrazi_torch_kompletna as funkcija

        razlika, recall, zamene = verifikuj(
            funkcija, "cpu", korpus_gpu, upiti, korpus, upiti, gt_ind, gt_vr, k,
            merenje_cfg["tolerancija"], naziv, upiti.shape[0])
        merenje = izmeri_gpu_po_upitima(
            lambda upit: funkcija(korpus_gpu, upit, k), upiti,
            warmup=merenje_cfg["warmup"], ponavljanja=merenje_cfg["ponavljanja"],
            zagrevanje_ms=merenje_cfg["zagrevanje_gpu_ms"])
        rezultati.dodaj(merenje, metoda=naziv, k=k, recall=round(recall, 6),
                        zamene=zamene, max_razlika=f"{razlika:.3e}", **osnova)
        print(f"  {naziv:<12} k={k:<3} {merenje}  recall={recall:.4f} zamene={zamene}")


def izmeri_rag(konfiguracija, rezultati, gpu):
    import json
    import time
    import pandas as pd
    from src.common.timing import Merenje
    from src.rag.retrieval import napravi_skladiste, procitaj_tekstove, ucitaj_model, ugradi

    rag = konfiguracija["rag"]
    merenje_cfg = konfiguracija["merenje"]
    putanje = konfiguracija["putanje"]
    N, k = rag["N"], rag["k"]
    if "gpu_pretraga" in rag["varijante"] and not gpu:
        raise RuntimeError("GPU nije dostupan, a trazena je varijanta gpu_pretraga")

    korpus, upiti, opis = ucitaj_podatke(konfiguracija, N)
    Q, d = upiti.shape[0], korpus.shape[1]
    osnova = {"serija": "rag", "izvor": opis["izvor"], "N": N, "d": d, "k": k, "broj_upita": Q}
    gt_ind, gt_vr = ground_truth(korpus, upiti, max(konfiguracija["podaci"]["k"]),
                                 KOREN / putanje["ground_truth"], opis)

    tekstovi_upita = pd.read_parquet(KOREN / putanje["upiti_tekst"])["text"].tolist()[:Q]
    pid_niz = np.load(KOREN / putanje["pid"])[:N]
    skladiste = napravi_skladiste(KOREN / putanje["pasusi_tekst"], N)
    with open(KOREN / putanje["opis"], encoding="utf-8") as fajl:
        model = ucitaj_model(json.load(fajl)["model"], rag["uredjaj_modela"])

    vektori = np.stack([ugradi(model, t) for t in tekstovi_upita])
    razlika = float(np.abs(vektori - upiti).max())
    if razlika > konfiguracija["merenje"]["tolerancija"]:
        raise AssertionError(f"rag: vektori upita se razlikuju od sacuvanih ({razlika:.2e})")
    print(f"  rag: vektori upita = sacuvani (najveca razlika {razlika:.2e})")

    pretrage = {}
    for varijanta in rag["varijante"]:
        if varijanta == "gpu_pretraga":
            import cupy as cp
            from src.cuda.pretraga import pripremi_pretragu, pretrazi_gpu
            cfg_s2, cfg_t3 = konfiguracija["kerneli"]["s2"], konfiguracija["kerneli"]["t3"]
            korpus_gpu = cp.asarray(korpus)
            radni = pripremi_pretragu(N, d, k, cfg_t3)
            pretrage[varijanta] = lambda v: pretrazi_gpu(korpus_gpu, v, k, cfg_s2, cfg_t3, radni)[0]
        elif varijanta == "cpu_numpy":
            pretrage[varijanta] = lambda v: pretrazi_numpy(korpus, v, k)[0]
        elif varijanta == "cpu_hnsw":
            from src.baselines.faiss_hnsw import napravi_ili_ucitaj_indeks, pretrazi_hnsw, putanja_indeksa
            hnsw = konfiguracija["hnsw"]
            indeks, _ = napravi_ili_ucitaj_indeks(
                korpus, hnsw["M"], hnsw["ef_construction"],
                putanja_indeksa(KOREN / putanje["indeksi"], opis, N, d, hnsw["M"], hnsw["ef_construction"]))
            indeks.hnsw.efSearch = rag["ef_search"]
            pretrage[varijanta] = lambda v: pretrazi_hnsw(indeks, v, k)[0]
        else:
            raise ValueError(f"nepoznata rag varijanta: {varijanta}")

    for varijanta, pretrazi in pretrage.items():
        recall_vrednosti, zamene = [], 0
        for q in range(Q):
            recall, broj_zamena = recall_aproksimativno(
                pretrazi(vektori[q]), gt_ind[q, :k], gt_vr[q, :k],
                lambda ind, q=q: korpus[ind] @ upiti[q], merenje_cfg["tolerancija"])
            recall_vrednosti.append(recall)
            zamene += broj_zamena
        recall = float(np.mean(recall_vrednosti))

        def jedan_upit(i):
            t0 = time.perf_counter()
            vektor = ugradi(model, tekstovi_upita[i % Q])
            t1 = time.perf_counter()
            indeksi = pretrazi(vektor)
            t2 = time.perf_counter()
            procitaj_tekstove(skladiste, pid_niz, indeksi)
            t3 = time.perf_counter()
            return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3

        pocetak, broj = time.perf_counter(), 0
        while broj < merenje_cfg["warmup"] or (time.perf_counter() - pocetak) * 1e3 < merenje_cfg["zagrevanje_gpu_ms"]:
            jedan_upit(broj)
            broj += 1

        vremena = np.array([jedan_upit(i) for i in range(merenje_cfg["ponavljanja"])])
        udeo = float(np.median(vremena[:, 1] / vremena[:, 3]))
        for kolona, faza in enumerate(("ugradnja", "pretraga", "tekst", "ukupno")):
            v = vremena[:, kolona]
            merenje = Merenje(float(np.median(v)), float(v.min()), float(v.max()),
                              merenje_cfg["ponavljanja"], broj, "cpu")
            dodatak = {"recall": round(recall, 6), "zamene": zamene} if faza == "pretraga" else {}
            if faza == "ukupno":
                dodatak["udeo_pretrage"] = round(udeo, 4)
            rezultati.dodaj(merenje, metoda=f"rag_{varijanta}_{faza}",
                            ef_search=rag["ef_search"] if varijanta == "cpu_hnsw" else "",
                            **dodatak, **osnova)
            print(f"  rag {varijanta:<13} {faza:<9} {merenje}")
        print(f"  rag {varijanta:<13} udeo pretrage (medijana po upitu) {100 * udeo:.1f} %, recall {recall:.4f}")

    del korpus, pretrage
    if gpu:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


def izmeri_slicnost(konfiguracija, naziv, korpus, upiti, korpus_gpu, upiti_gpu,
                    rezultati, osnova, izmereni_opseg):
    import cupy as cp
    from src.common.verify import uporedi_vrednosti
    from src.cuda.slicnost import (SLICNOSTI as FUNKCIJE, TRAZE_FLOAT4, bajtova_slicnosti,
                                   proveri_float4, proveri_ulaze)

    merenje_cfg = konfiguracija["merenje"]
    kljuc, funkcija = FUNKCIJE[naziv]
    cfg = konfiguracija["kerneli"][kljuc]
    N, d = korpus.shape
    izlaz = cp.empty(N, dtype=cp.float32)
    proveri_ulaze(korpus_gpu, upiti_gpu[0], izlaz)
    if naziv in TRAZE_FLOAT4:
        proveri_float4(korpus_gpu, upiti_gpu)

    najveca = 0.0
    for q in range(upiti.shape[0]):
        izlaz.fill(cp.nan)
        funkcija(korpus_gpu, upiti_gpu[q], izlaz, cfg)
        poklapa_se, razlika = uporedi_vrednosti(cp.asnumpy(izlaz), korpus @ upiti[q],
                                                merenje_cfg["tolerancija"])
        if not poklapa_se:
            raise AssertionError(f"{naziv} (upit {q}): najveca razlika {razlika:.3e}")
        najveca = max(najveca, razlika)

    merenje = izmeri_gpu_po_upitima(
        lambda upit: funkcija(korpus_gpu, upit, izlaz, cfg), upiti_gpu,
        warmup=merenje_cfg["warmup"], ponavljanja=merenje_cfg["ponavljanja"],
        zagrevanje_ms=merenje_cfg["zagrevanje_gpu_ms"])

    opseg = merenje.propusni_opseg_gbps(bajtova_slicnosti(N, d))
    procenat = "" if izmereni_opseg is None else round(100 * opseg / izmereni_opseg, 1)
    rezultati.dodaj(merenje, metoda=naziv, max_razlika=f"{najveca:.3e}",
                    propusni_opseg_gbps=round(opseg, 2), procenat_opsega=procenat, **osnova)
    print(f"  {naziv:<12} {merenje}  {opseg:.1f} GB/s ({procenat} %)  max razlika {najveca:.2e}")
    del izlaz


def pripremi_ulaz_topk(konfiguracija, korpus_gpu, upiti_gpu):
    import cupy as cp
    from src.cpu.numpy_ref import top_k_numpy_redosled
    from src.cuda.slicnost import slicnost_s2

    N = korpus_gpu.shape[0]
    Q = upiti_gpu.shape[0]
    slicnosti = cp.empty((Q, N), dtype=cp.float32)
    for q in range(Q):
        slicnost_s2(korpus_gpu, upiti_gpu[q], slicnosti[q], konfiguracija["kerneli"]["s2"])

    k_max = min(max(konfiguracija["podaci"]["k"]), N)
    ref_ind = np.empty((Q, k_max), dtype=np.int64)
    ref_vr = np.empty((Q, k_max), dtype=np.float32)
    for q in range(Q):
        ref_ind[q], ref_vr[q] = top_k_numpy_redosled(cp.asnumpy(slicnosti[q]), k_max)
    return slicnosti, ref_ind, ref_vr


def izmeri_topk(konfiguracija, naziv, slicnosti, ref_ind, ref_vr, rezultati, osnova):
    from src.cuda.topk import TOPK as FUNKCIJE, proveri_topk

    merenje_cfg = konfiguracija["merenje"]
    kljuc, pripremi, funkcija = FUNKCIJE[naziv]
    cfg = konfiguracija["kerneli"][kljuc]
    Q, N = slicnosti.shape

    for k in konfiguracija["podaci"]["k"]:
        if k > N:
            continue
        radni = pripremi(N, k, cfg)
        for q in range(Q):
            indeksi, vrednosti = funkcija(slicnosti[q], k, cfg, radni)
            proveri_topk(indeksi, vrednosti, ref_ind[q, :k], ref_vr[q, :k],
                         f"{naziv} (upit {q}, k={k})")

        merenje = izmeri_gpu_po_upitima(
            lambda s: funkcija(s, k, cfg, radni), slicnosti,
            warmup=merenje_cfg["warmup"], ponavljanja=merenje_cfg["ponavljanja"],
            zagrevanje_ms=merenje_cfg["zagrevanje_gpu_ms"])
        kandidati = int(radni["brojac"][0]) if "brojac" in radni else ""
        rezultati.dodaj(merenje, metoda=naziv, k=k, recall=1.0, zamene=0,
                        max_razlika="0", kandidati=kandidati, **osnova)
        dodatak = f"  kandidata={kandidati}" if kandidati != "" else ""
        print(f"  {naziv:<12} k={k:<3} {merenje}  bit-tacno{dodatak}")
        del radni


def izmeri_N(konfiguracija, N, rezultati, gpu, izmereni_opseg=None,
             metode=None, vrednosti_k=None, serija="glavna"):
    k_max = max(konfiguracija["podaci"]["k"])
    if vrednosti_k is not None:
        konfiguracija = {**konfiguracija, "podaci": {**konfiguracija["podaci"], "k": vrednosti_k}}
    podaci = konfiguracija["podaci"]
    merenje_cfg = konfiguracija["merenje"]
    naivna = konfiguracija["naivna"]

    korpus, upiti, opis = ucitaj_podatke(konfiguracija, N)
    d, Q = korpus.shape[1], upiti.shape[0]
    osnova = {"serija": serija, "izvor": opis["izvor"], "N": N, "d": d, "broj_upita": Q}

    gt_ind, gt_vr = ground_truth(korpus, upiti, max(k_max, max(podaci["k"])),
                                 KOREN / konfiguracija["putanje"]["ground_truth"], opis)

    metode = [m for m in (metode or merenje_cfg["metode"])
              if not (m == "cpu_naivno" and N > naivna["max_N"])]
    gpu_metode = [m for m in metode if uredjaj_metode(m) == "gpu"]
    if gpu_metode and not gpu:
        raise RuntimeError(f"GPU nije dostupan, a trazene su metode {gpu_metode}")

    korpus_gpu = upiti_gpu = None
    if gpu:
        import cupy as cp
        izmeri_prenos_korpusa(korpus, konfiguracija, rezultati, osnova)
        if gpu_metode:
            korpus_gpu, upiti_gpu = cp.asarray(korpus), cp.asarray(upiti)

    ulaz_topk = None
    for naziv in metode:
        if naziv in TOPK:
            if ulaz_topk is None:
                ulaz_topk = pripremi_ulaz_topk(konfiguracija, korpus_gpu, upiti_gpu)
            izmeri_topk(konfiguracija, naziv, *ulaz_topk, rezultati, osnova)
            continue
        if naziv == "cpu_hnsw":
            izmeri_hnsw(konfiguracija, korpus, upiti, gt_ind, gt_vr, opis, rezultati, osnova)
            continue
        if naziv == "cpu_faiss_flat":
            izmeri_faiss_flat(konfiguracija, korpus, upiti, gt_ind, gt_vr, rezultati, osnova)
            continue
        if naziv in KOMPLETNE:
            izmeri_kompletnu(konfiguracija, naziv, korpus, upiti, korpus_gpu, gt_ind, gt_vr,
                             rezultati, osnova)
            continue
        if naziv in SLICNOSTI:
            izmeri_slicnost(konfiguracija, naziv, korpus, upiti, korpus_gpu, upiti_gpu,
                            rezultati, osnova, izmereni_opseg)
            continue

        funkcija, uredjaj = METODE[naziv]
        korpus_x, upiti_x = (korpus_gpu, upiti_gpu) if uredjaj == "gpu" else (korpus, upiti)

        if naziv == "cpu_naivno":
            vrednosti_k, ponavljanja, broj_upita_provere = naivna["k"], naivna["ponavljanja"], naivna["broj_upita_provere"]
        else:
            vrednosti_k, ponavljanja, broj_upita_provere = podaci["k"], merenje_cfg["ponavljanja"], Q

        for k in vrednosti_k:
            razlika, recall, zamene = verifikuj(
                funkcija, uredjaj, korpus_x, upiti_x, korpus, upiti, gt_ind, gt_vr, k,
                merenje_cfg["tolerancija"], naziv, broj_upita_provere)

            def pozovi(upit):
                return funkcija(korpus_x, upit, k)

            if uredjaj == "gpu":
                merenje = izmeri_gpu_po_upitima(
                    pozovi, upiti_x, warmup=merenje_cfg["warmup"], ponavljanja=ponavljanja,
                    zagrevanje_ms=merenje_cfg["zagrevanje_gpu_ms"])
            else:
                merenje = izmeri_cpu_po_upitima(
                    pozovi, upiti_x, warmup=merenje_cfg["warmup"], ponavljanja=ponavljanja)

            rezultati.dodaj(merenje, metoda=naziv, k=k, recall=round(recall, 6),
                            zamene=zamene, max_razlika=f"{razlika:.3e}", **osnova)
            print(f"  {naziv:<12} k={k:<3} {merenje}  recall={recall:.4f} zamene={zamene}")

    del korpus, korpus_gpu, upiti_gpu, ulaz_topk
    if gpu:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(KOREN / "bench" / "config.yaml"))
    parser.add_argument("--izlaz", default=None)
    argumenti = parser.parse_args()

    konfiguracija = ucitaj_konfiguraciju(argumenti.config)
    postavi_niti_procesora(konfiguracija["merenje"]["niti_procesora"])

    if argumenti.izlaz:
        izlazni_fajl = Path(argumenti.izlaz)
    else:
        oznaka = datetime.now().strftime("%Y%m%d_%H%M%S")
        izlazni_fajl = KOREN / konfiguracija["putanje"]["rezultati"] / f"rezultati_{oznaka}.csv"
    izlazni_fajl.parent.mkdir(parents=True, exist_ok=True)

    metode = sve_metode(konfiguracija)
    rag_ukljucen = bool((konfiguracija.get("rag") or {}).get("ukljuceno"))
    if rag_ukljucen:
        from src.baselines.torch_ref import proveri_tf32
        proveri_tf32()
    if set(metode) & TORCH_METODE:
        from src.baselines.torch_ref import pretrazi_torch, proveri_stream, proveri_tf32
        proveri_tf32()
        proveri_stream()
        METODE["gpu_torch"] = (pretrazi_torch, "gpu")

    okruzenje = podaci_o_okruzenju()
    okruzenje["config"] = str(Path(argumenti.config).resolve())
    if set(metode) & TORCH_METODE:
        okruzenje["torch_metoda"] = ("torch.mv + torch.topk nad CuPy nizovima bez kopiranja; "
                                     "isti default stream kao CUDA events; TF32 iskljucen")
    if set(metode) & set(KOMPLETNE):
        okruzenje["kompletna_pretraga"] = ("gpu_pretraga = S2 + T3, gpu_torch_kompletna = mv + topk; "
                                           "upit iz RAM-a na karticu i k rezultata nazad u RAM ulaze u vreme")
    if rag_ukljucen:
        okruzenje["rag"] = (f'retrieval bez jezickog modela; N={konfiguracija["rag"]["N"]}, k={konfiguracija["rag"]["k"]}, '
                            f'varijante={konfiguracija["rag"]["varijante"]}, efSearch={konfiguracija["rag"]["ef_search"]}, '
                            f'model na {konfiguracija["rag"]["uredjaj_modela"]}; faze se mere perf_counter-om')
    if "cpu_faiss_flat" in metode:
        okruzenje["faiss_flat_niti"] = ("jedan upit po pozivu; 1-8 niti daju priblizno istu medijanu "
                                        "(samotest faiss_flat.py), 8 niti veci raspon")
    if "cpu_hnsw" in konfiguracija["merenje"]["metode"]:
        hnsw = konfiguracija["hnsw"]
        okruzenje["hnsw"] = (f'IndexHNSWFlat, unutrasnji proizvod, M={hnsw["M"]}, '
                             f'efConstruction={hnsw["ef_construction"]}, efSearch={hnsw["ef_search"]}')
        okruzenje["hnsw_niti"] = ("izgradnja: niti_procesora; pretraga: jedan upit po pozivu, "
                                  "FAISS paralelizuje po upitima pa radi jedna nit")
    print("okruzenje:")
    for kljuc, vrednost in okruzenje.items():
        print(f"  {kljuc}: {vrednost}")
    print()

    gpu = gpu_je_dostupan()
    takt = pokreni_pracenje_takta(konfiguracija, izlazni_fajl.with_suffix(".takt.csv")) if gpu else None
    rezultati = Rezultati(izlazni_fajl)

    try:
        izmereni_opseg = None
        if gpu:
            izmereni_opseg = izmeri_propusni_opseg(konfiguracija, rezultati, okruzenje)
        izmeri_propusni_opseg_cpu(konfiguracija, rezultati, okruzenje)
        for N in konfiguracija["podaci"]["N"]:
            print(f"N={N}")
            izmeri_N(konfiguracija, N, rezultati, gpu, izmereni_opseg)

        prelomna = konfiguracija.get("prelomna_tacka")
        if prelomna and prelomna.get("metode"):
            for N in prelomna["N"]:
                print(f"prelomna tacka, N={N}")
                izmeri_N(konfiguracija, N, rezultati, gpu, izmereni_opseg,
                         metode=prelomna["metode"], vrednosti_k=prelomna["k"],
                         serija="prelomna_tacka")

        if (konfiguracija.get("rag") or {}).get("ukljuceno"):
            print("rag (retrieval deo, P7)")
            izmeri_rag(konfiguracija, rezultati, gpu)
    finally:
        rezultati.zatvori()
        if takt is not None:
            proces, fajl = takt
            proces.terminate()
            proces.wait(timeout=10)
            fajl.close()

        with open(izlazni_fajl.with_suffix(".okruzenje.txt"), "w", encoding="utf-8") as fajl:
            for kljuc, vrednost in okruzenje.items():
                fajl.write(f"{kljuc}: {vrednost}\n")

    print(f"\nzapisano {rezultati.broj} redova u {izlazni_fajl}")


if __name__ == "__main__":
    main()
