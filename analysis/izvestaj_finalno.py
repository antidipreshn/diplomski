import argparse
from pathlib import Path

import pandas as pd

KOREN = Path(__file__).resolve().parents[1]
KLJUC = ["metoda", "serija", "N", "k", "ef_search"]
NAZIVI = {
    "cpu_numpy": "NumPy", "cpu_faiss_flat": "FAISS-Flat", "cpu_hnsw": "HNSW",
    "gpu_torch": "PyTorch (mv + topk)", "gpu_pretraga": "S2 + T3 (kompletna)",
    "gpu_torch_kompletna": "PyTorch (kompletna)",
    **{f"gpu_s{i}": f"S{i}" for i in range(1, 6)}, **{f"gpu_t{i}": f"T{i}" for i in range(1, 5)},
}


def broj(x, dec=None):
    if pd.isna(x):
        return "—"
    if dec is None:
        a = abs(x)
        dec = 1 if a >= 100 else 2 if a >= 10 else 3 if a >= 1 else 4
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


def N_tekst(N):
    N = int(N)
    stepen = {1000: "10³", 10000: "10⁴", 100000: "10⁵", 1000000: "10⁶"}
    if N in stepen:
        return stepen[N]
    for s, t in ((100000, "10⁵"), (10000, "10⁴"), (1000, "10³")):
        if N % s == 0:
            return f"{N // s}·{t}"
    return str(N)


def stanje(s):
    if pd.isna(s) or s == "":
        return ""
    return s if str(s).startswith("P2") else f"**{s}** †"


def tabela(redovi, kolone):
    izlaz = ["| " + " | ".join(kolone) + " |", "|" + "|".join("---" for _ in kolone) + "|"]
    izlaz += ["| " + " | ".join(r) + " |" for r in redovi]
    return "\n".join(izlaz)


def ucitaj(putanja):
    f = pd.read_csv(putanja)
    f[KLJUC] = f[KLJUC].fillna(-1)
    return f


def spoji(a, b):
    s = a.merge(b[KLJUC + ["medijana_ms", "stanje_gpu", "recall"]], on=KLJUC, how="left", suffixes=("", "_2"))
    s["razlika"] = (s.medijana_ms_2 / s.medijana_ms - 1) * 100
    return s


def red_merenja(r, sa_k=True, dodatno=()):
    celije = [NAZIVI.get(r.metoda, r.metoda)]
    if sa_k:
        celije.append(str(int(r.k)) if r.k != -1 else "—")
    celije += [broj(r.medijana_ms), broj(r.min_ms), broj(r.max_ms)]
    celije += [d(r) for d in dodatno]
    celije += [broj(r.medijana_ms_2), broj(r.razlika, 1) + " %", stanje(r.stanje_gpu)]
    return celije


OSNOVNE = ["medijana [ms]", "min [ms]", "max [ms]"]
PONOVLJENO = ["2. pok. [ms]", "razlika", "takt (1. pok.)"]


def napravi(p1, p2, okruzenje_putanja):
    a, b = ucitaj(p1), ucitaj(p2)
    s = spoji(a, b)
    gl = s[s.serija == "glavna"]
    T = []

    T.append("# Finalna merenja (korak 11)\n")
    T.append(f"*Generisano skriptom `analysis/izvestaj_finalno.py` iz `{Path(p1).name}` i `{Path(p2).name}`. "
             "Ne menjati ručno; posle izmene CSV-a pokrenuti skriptu ponovo.*\n")
    T.append(f"- **1. pokretanje (izvor brojeva za rad):** `{Path(p1).name}`, {a.vreme_pokretanja.min()} – {a.vreme_pokretanja.max()}, {len(a)} redova")
    T.append(f"- **2. pokretanje (provera ponovljivosti):** `{Path(p2).name}`, {b.vreme_pokretanja.min()} – {b.vreme_pokretanja.max()}, {len(b)} redova")
    T.append("- Konfiguracija: `bench/config.yaml` (100 upita, 100 ponavljanja, warmup 5 + ≥ 1000 ms za GPU, 8 niti procesora); takt kartice nije zaključan")
    T.append("- Svako vreme: **medijana** po pozivu, uz raspon **min–max**. „Razlika“ = (2. pok. / 1. pok. − 1) · 100 %")
    T.append("- **†** = GPU red koji nije izmeren u radnom stanju P2 (kolona `stanje_gpu`, očitano odmah posle merenja)\n")

    T.append("## 1. Okruženje\n")
    T.append("Iz `" + Path(okruzenje_putanja).name + "`:\n")
    T.append("```")
    T.append(Path(okruzenje_putanja).read_text(encoding="utf-8").strip())
    T.append("```\n")

    T.append("## 2. Provere\n")
    for ime, f in (("1.", a), ("2.", b)):
        egz = f[(f.recall != -1) & f.recall.notna() & ~f.metoda.str.contains("hnsw")]
        g = f[f.uredjaj == "gpu"]
        van = g[~g.stanje_gpu.fillna("").str.startswith("P2")]
        T.append(f"- **{ime} pokretanje:** egzaktnih redova sa proverom {len(egz)}, od toga recall < 1: "
                 f"{int((egz.recall < 1).sum())}, sa zamenama na granici: {int((egz.zamene.fillna(0) > 0).sum())}; "
                 f"GPU redova van P2: {len(van)} od {len(g)}")
    T.append("")
    T.append("**GPU redovi van P2 (1. pokretanje):**\n")
    g = s[(s.uredjaj == "gpu") & ~s.stanje_gpu.fillna("").str.startswith("P2")]
    T.append(tabela([[NAZIVI.get(r.metoda, r.metoda), r.serija, N_tekst(r.N), str(int(r.k)), broj(r.medijana_ms),
                      r.stanje_gpu, str(r.stanje_gpu_2)] for r in g.itertuples()],
                    ["metoda", "serija", "N", "k", "medijana [ms]", "stanje (1.)", "stanje (2.)"]))
    T.append("\nSvi su kratki pozivi: kernel traje kraće od vremena između dva poziva, kartica je pretežno neaktivna i drajver spušta takt (razlog `0x1`, GpuIdle).\n")

    T.append("### Ponovljivost (|razlika| medijana između dva pokretanja)\n")
    s["uredjaj_grupa"] = s.uredjaj.fillna("—")
    p = s[~s.metoda.str.contains("izgradnja") & (s.serija != "rag") & (s.N != -1)].copy()
    p["aps"] = p.razlika.abs()
    redovi = []
    for (ur, N), grupa in p[p.serija == "glavna"].groupby(["uredjaj_grupa", "N"]):
        redovi.append([ur, N_tekst(N), str(len(grupa)), broj(grupa.aps.median(), 2) + " %", broj(grupa.aps.max(), 1) + " %",
                       str(int((grupa.aps <= 1).sum()))])
    T.append(tabela(redovi, ["uređaj", "N", "redova", "medijana |razlike|", "najveća |razlika|", "redova ≤ 1 %"]))
    T.append("")

    T.append("## 3. Propusni opseg i prenos korpusa\n")
    redovi = []
    for r in s[s.metoda.isin(["propusni_opseg", "propusni_opseg_cpu"])].itertuples():
        redovi.append([r.metoda, broj(r.medijana_ms), broj(r.min_ms), broj(r.max_ms), broj(r.propusni_opseg_gbps, 1),
                       broj(r.medijana_ms_2), broj(r.razlika, 1) + " %"])
    T.append(tabela(redovi, ["merenje"] + OSNOVNE + ["GB/s", "2. pok. [ms]", "razlika"]))
    T.append("\n`propusni_opseg` = kopiranje 256 MB na kartici (čitanje + upis); `propusni_opseg_cpu` = 8 niti čitaju 2048 MB.\n")
    redovi = []
    for r in s[s.metoda == "prenos_korpusa"].itertuples():
        redovi.append([r.serija, N_tekst(r.N), broj(r.medijana_ms), broj(r.min_ms), broj(r.max_ms),
                       broj(r.propusni_opseg_gbps, 2), broj(r.medijana_ms_2), broj(r.razlika, 1) + " %"])
    T.append("**Prenos korpusa u memoriju kartice** (jednom, ne ulazi u vreme po upitu):\n")
    T.append(tabela(redovi, ["serija", "N"] + OSNOVNE + ["GB/s", "2. pok. [ms]", "razlika"]))
    T.append("")

    T.append("## 4. Sličnost: S1–S5 (P1)\n")
    T.append("Ceo vektor sličnosti za jedan upit; „GB/s“ = veličina korpusa / medijana; „% opsega“ u odnosu na izmereno kopiranje iz odeljka 3.\n")
    for N in sorted(gl.N.unique()):
        d = gl[(gl.N == N) & gl.metoda.str.match(r"gpu_s\d")]
        if d.empty:
            continue
        T.append(f"**N = {N_tekst(N)}**\n")
        T.append(tabela([red_merenja(r, sa_k=False, dodatno=(lambda r: broj(r.propusni_opseg_gbps, 1),
                                                              lambda r: broj(r.procenat_opsega, 1)))
                         for r in d.itertuples()],
                        ["metoda"] + OSNOVNE + ["GB/s", "% opsega"] + PONOVLJENO))
        T.append("")

    T.append("## 5. Selekcija k najvećih: T1–T4 (P2)\n")
    T.append("Ulaz je gotov vektor sličnosti u memoriji kartice; meri se samo selekcija. „Kandidati“ = koliko elemenata T4 propusti kroz prag.\n")
    for N in sorted(gl.N.unique()):
        d = gl[(gl.N == N) & gl.metoda.str.match(r"gpu_t\d")]
        T.append(f"**N = {N_tekst(N)}**\n")
        T.append(tabela([red_merenja(r, dodatno=(lambda r: broj(r.kandidati, 0) if pd.notna(r.kandidati) else "",))
                         for r in d.itertuples()],
                        ["metoda", "k"] + OSNOVNE + ["kandidati"] + PONOVLJENO))
        T.append("")

    T.append("## 6. Reference: NumPy, PyTorch, FAISS-Flat (P5)\n")
    T.append("NumPy i FAISS-Flat: ceo upit na procesoru (8 niti). PyTorch: `mv` + `topk` nad korpusom koji je već na kartici.\n")
    for N in sorted(gl.N.unique()):
        d = gl[(gl.N == N) & gl.metoda.isin(["cpu_numpy", "gpu_torch", "cpu_faiss_flat"])]
        T.append(f"**N = {N_tekst(N)}**\n")
        T.append(tabela([red_merenja(r) for r in d.itertuples()], ["metoda", "k"] + OSNOVNE + PONOVLJENO))
        T.append("")
    izg = s[s.metoda == "faiss_flat_izgradnja"]
    T.append("**Izgradnja FAISS-Flat indeksa** (kopiranje korpusa u indeks):\n")
    T.append(tabela([[r.serija, N_tekst(r.N), broj(r.medijana_ms), broj(r.medijana_ms_2)] for r in izg.itertuples()],
                    ["serija", "N", "1. pok. [ms]", "2. pok. [ms]"]))
    T.append("")

    T.append("## 7. Kompletna pretraga (P4, P5)\n")
    T.append("Upit iz RAM-a na karticu, sličnost, selekcija i k rezultata nazad u RAM; korpus je već na kartici.\n")
    for N in sorted(gl.N.unique()):
        d = gl[(gl.N == N) & gl.metoda.isin(["gpu_pretraga", "gpu_torch_kompletna"])]
        T.append(f"**N = {N_tekst(N)}**\n")
        T.append(tabela([red_merenja(r) for r in d.itertuples()], ["metoda", "k"] + OSNOVNE + PONOVLJENO))
        T.append("")

    T.append("## 8. HNSW na procesoru (P6)\n")
    T.append("`IndexHNSWFlat`, M = 32, efConstruction = 200; jedan upit po pozivu. Kombinacije sa efSearch < k se ne mere.\n")
    izg = s[s.metoda == "hnsw_izgradnja"]
    T.append("**Izgradnja indeksa** (iz prve izgradnje 22.09.2026, indeksi se učitavaju sa diska; odluka 03.10.):\n")
    T.append(tabela([[N_tekst(r.N), broj(r.medijana_ms / 1000, 1) + " s"] for r in izg.itertuples()], ["N", "vreme izgradnje"]))
    T.append("")
    for N in sorted(gl.N.unique()):
        d = gl[(gl.N == N) & (gl.metoda == "cpu_hnsw")].sort_values(["k", "ef_search"])
        T.append(f"**N = {N_tekst(N)}**\n")
        T.append(tabela([[str(int(r.k)), str(int(r.ef_search)), broj(r.medijana_ms), broj(r.min_ms), broj(r.max_ms),
                          broj(r.recall, 4), broj(r.medijana_ms_2), broj(r.razlika, 1) + " %"] for r in d.itertuples()],
                        ["k", "efSearch"] + OSNOVNE + ["recall@k", "2. pok. [ms]", "razlika"]))
        T.append("")

    T.append("## 9. Prelomna tačka (P4)\n")
    T.append("Kompletna pretraga za gušći niz N. Medijane u ms; † = GPU red van P2.\n")
    pr = s[s.serija == "prelomna_tacka"]
    met = ["cpu_numpy", "cpu_faiss_flat", "gpu_pretraga", "gpu_torch_kompletna"]
    intervali = {}
    for k in sorted(pr[pr.k != -1].k.unique()):
        d = pr[pr.k == k]
        redovi = []
        for N in sorted(d.N.unique()):
            red = [N_tekst(N)]
            for m in met:
                r = d[(d.N == N) & (d.metoda == m)].iloc[0]
                znak = " †" if isinstance(r.stanje_gpu, str) and not r.stanje_gpu.startswith("P2") else ""
                red.append(f"{broj(r.medijana_ms)}{znak} / {broj(r.medijana_ms_2)}")
            redovi.append(red)
        T.append(f"**k = {int(k)}** (1. pok. / 2. pok.)\n")
        T.append(tabela(redovi, ["N"] + [NAZIVI[m] for m in met]))
        T.append("")
        for kolona, ime in (("medijana_ms", "1."), ("medijana_ms_2", "2.")):
            t = d.pivot_table(index="N", columns="metoda", values=kolona)
            brzi = t.gpu_pretraga < t.cpu_numpy
            prvi = brzi[brzi].index.min()
            pre = brzi.index[brzi.index < prvi].max()
            intervali[(int(k), ime)] = f"{N_tekst(pre)} – {N_tekst(prvi)}"
    T.append("**Interval prelomne tačke (S2 + T3 naspram NumPy-ja):** poslednje N gde S2 + T3 nije brži do prvog N gde jeste.\n")
    T.append(tabela([[str(k), intervali[(k, "1.")], intervali[(k, "2.")]] for k in sorted({k for k, _ in intervali})],
                    ["k", "1. pokretanje", "2. pokretanje"]))
    T.append("")
    T.append("Za k = 100 se interval pomera jer je na N = 5·10³ razlika između S2 + T3 i NumPy-ja mala, a NumPy na tom N "
             "varira između pokretanja više nego GPU (vidi tabelu iznad). Prelomna tačka za k = 100 je zato oko 5·10³, "
             "a ne oštra granica; za k = 10 je ista u oba pokretanja.\n")

    T.append("## 10. Retrieval deo RAG-a (P7)\n")
    T.append("N = 10⁶, k = 10, efSearch = 128 za HNSW; tekst upita → vektor (model `all-mpnet-base-v2` na kartici) → pretraga → tekst pasusa. Mereno `perf_counter`-om po fazama.\n")
    rag = s[s.serija == "rag"].copy()
    redovi = []
    for var in ["gpu_pretraga", "cpu_numpy", "cpu_hnsw"]:
        for faza in ["ugradnja", "pretraga", "tekst", "ukupno"]:
            r = rag[rag.metoda == f"rag_{var}_{faza}"].iloc[0]
            udeo = broj(r.udeo_pretrage * 100, 1) + " %" if pd.notna(r.udeo_pretrage) else ""
            udeo2 = b[(b.metoda == r.metoda)].udeo_pretrage.iloc[0]
            udeo2 = broj(udeo2 * 100, 1) + " %" if pd.notna(udeo2) else ""
            rec = broj(r.recall, 3) if pd.notna(r.recall) and r.recall != -1 else ""
            redovi.append([NAZIVI.get(var, var), faza, broj(r.medijana_ms), broj(r.min_ms), broj(r.max_ms),
                           udeo, rec, broj(r.medijana_ms_2), broj(r.razlika, 1) + " %", udeo2])
    T.append(tabela(redovi, ["varijanta", "faza"] + OSNOVNE + ["udeo pretrage", "recall", "2. pok. [ms]", "razlika", "udeo (2.)"]))
    T.append("")
    T.append("**Anomalija:** ugradnju upita u sve tri varijante radi isti model na istoj kartici, ali kod varijante sa NumPy pretragom traje ≈ 31 ms umesto ≈ 8,5–8,9 ms, u **oba** pokretanja. "
             "Zapis takta (`.takt.csv`) u oba pokretanja pokazuje da kartica tokom te varijante prelazi P2 → P3 → P5 → P8 sa razlogom `0x1` (GpuIdle), na temperaturi 53–60 °C: "
             "dok procesor ≈ 85–89 ms radi pretragu, kartica nema posla i drajver spušta takt, pa se sledeća ugradnja izvršava sporije. "
             "Nije termičko ograničenje. Udeo pretrage za NumPy je zato manji nego što bi bio uz ugradnju od ≈ 8,5 ms.\n")

    T.append("## 11. Sažetak za N = 10⁶ (1. pokretanje)\n")
    def m(metoda, k=-1, N=1000000, serija="glavna"):
        return s[(s.metoda == metoda) & (s.N == N) & (s.k == k) & (s.serija == serija)].medijana_ms.iloc[0]
    redovi = []
    for k in (10, 100):
        npy, fl, pt, sk, tk = m("cpu_numpy", k), m("cpu_faiss_flat", k), m("gpu_torch", k), m("gpu_pretraga", k), m("gpu_torch_kompletna", k)
        redovi.append([str(k), broj(sk), broj(tk), broj(pt), broj(npy), broj(fl), broj(npy / sk, 1) + "×", broj(fl / sk, 1) + "×"])
    T.append(tabela(redovi, ["k", "S2 + T3 kompletna", "PyTorch kompletna", "PyTorch (bez prenosa upita)", "NumPy", "FAISS-Flat",
                             "NumPy / S2+T3", "FAISS-Flat / S2+T3"]))
    T.append("")
    s2, opseg = m("gpu_s2"), s[s.metoda == "propusni_opseg"].propusni_opseg_gbps.iloc[0]
    T.append(f"- S2: {broj(s2)} ms, {broj(gl[(gl.metoda == 'gpu_s2') & (gl.N == 1000000)].propusni_opseg_gbps.iloc[0], 1)} GB/s "
             f"(izmereno kopiranje {broj(opseg, 1)} GB/s)")
    T.append(f"- S1 / S2 = {broj(m('gpu_s1') / s2, 1)}×; T1 / T3 za k = 100: {broj(m('gpu_t1', 100) / m('gpu_t3', 100), 1)}×")
    T.append("")
    return "\n".join(T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prvo", default=str(KOREN / "bench/results/finalno_1.csv"))
    parser.add_argument("--drugo", default=str(KOREN / "bench/results/finalno_2.csv"))
    parser.add_argument("--izlaz", default=str(KOREN / "docs/finalna_merenja.md"))
    argumenti = parser.parse_args()
    okruzenje = Path(argumenti.prvo).with_suffix(".okruzenje.txt")
    tekst = napravi(argumenti.prvo, argumenti.drugo, okruzenje)
    Path(argumenti.izlaz).write_text(tekst, encoding="utf-8")
    print("zapisano:", argumenti.izlaz)


if __name__ == "__main__":
    main()
