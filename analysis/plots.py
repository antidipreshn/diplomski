import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, NullFormatter
import pandas as pd

D = 768
N_VELIKO = 1_000_000
KLJUC = ["metoda", "serija", "N", "k", "ef_search"]

BOJE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERI = ["o", "s", "^", "D", "v", "P"]
LINIJE = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
TEKST, TEKST2, MREZA, OSA = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

IZGLED_PRETRAGE = {
    "gpu_pretraga": ("S2 + T3 (GPU)", 0),
    "cpu_numpy": ("NumPy (CPU)", 1),
    "gpu_torch_kompletna": ("PyTorch (GPU)", 2),
    "cpu_faiss_flat": ("FAISS-Flat (CPU)", 3),
    "cpu_hnsw": ("HNSW (CPU)", 7),
}
SLICNOST = {"gpu_s1": "S1", "gpu_s2": "S2", "gpu_s3": "S3", "gpu_s4": "S4", "gpu_s5": "S5 (cuBLAS)"}
SELEKCIJA = {"gpu_t1": "T1", "gpu_t2": "T2", "gpu_t3": "T3", "gpu_t4": "T4"}


def zarez(x, decimala=None):
    tekst = f"{x:g}" if decimala is None else f"{x:.{decimala}f}"
    return tekst.replace(".", ",")


def stil(i, boja=None):
    return dict(color=boja or BOJE[i % len(BOJE)], marker=MARKERI[i % len(MARKERI)],
                linestyle=LINIJE[i % len(LINIJE)], linewidth=1.5, markersize=5)


def podesi_izgled():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
        "axes.edgecolor": OSA, "axes.labelcolor": TEKST, "xtick.color": TEKST2, "ytick.color": TEKST2,
        "text.color": TEKST, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": MREZA, "grid.linewidth": 0.5,
        "legend.frameon": False, "savefig.dpi": 300, "svg.fonttype": "none",
    })


def osa_vremena(ax, osa="y"):
    cilj = ax.yaxis if osa == "y" else ax.xaxis
    (ax.set_yscale if osa == "y" else ax.set_xscale)("log")
    cilj.set_major_formatter(FuncFormatter(lambda v, _: zarez(v)))
    cilj.set_minor_formatter(NullFormatter())


def osa_N(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlabel("veličina kolekcije N")


def greska(redovi):
    return [redovi["medijana_ms"] - redovi["min_ms"], redovi["max_ms"] - redovi["medijana_ms"]]


def sacuvaj(fig, folder, ime):
    for nastavak in ("png", "svg"):
        fig.savefig(folder / f"{ime}.{nastavak}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {ime}.png / .svg")


class Podaci:
    def __init__(self, putanje):
        delovi = []
        for redosled, putanja in enumerate(putanje):
            tabela = pd.read_csv(putanja)
            if "serija" not in tabela:
                tabela["serija"] = ""
            tabela["serija"] = tabela["serija"].fillna("").replace("", "glavna")
            tabela["izvor"] = Path(putanja).name
            tabela["redosled"] = redosled
            delovi.append(tabela)
        sve = pd.concat(delovi, ignore_index=True)
        kljuc = sve[KLJUC].astype(str)
        sve = sve.loc[~kljuc.duplicated(keep="last")]
        self.sve = sve
        self.korisceni = []
        self.opseg = self._teorijski_opseg(putanje)

    @staticmethod
    def _teorijski_opseg(putanje):
        for putanja in reversed(putanje):
            okruzenje = Path(putanja).with_suffix(".okruzenje.txt")
            if okruzenje.exists():
                for red in okruzenje.read_text(encoding="utf-8").splitlines():
                    kljuc, _, vrednost = red.partition(": ")
                    if kljuc == "teorijski_opseg_pri_taktu_gbps":
                        return float(vrednost)
        raise SystemExit("nema teorijski_opseg_pri_taktu_gbps ni u jednom .okruzenje.txt")

    def izaberi(self, metoda, serija="glavna", **uslovi):
        t = self.sve[(self.sve["metoda"] == metoda) & (self.sve["serija"] == serija)]
        for kolona, vrednost in uslovi.items():
            t = t[t[kolona] == vrednost]
        t = t.sort_values([c for c in ("N", "k", "ef_search") if t[c].notna().any()] or ["metoda"])
        if not t.empty:
            self.korisceni.append(t)
        return t

    def izvestaj(self):
        t = pd.concat(self.korisceni).drop_duplicates(subset=KLJUC)
        print("\nizvori:")
        for (izvor, metoda), grupa in t.groupby(["izvor", "metoda"]):
            print(f"  {izvor:<38}{metoda:<28}{len(grupa):>4} redova")
        egzaktne = t[~t["metoda"].str.contains("hnsw") & t["recall"].notna()]
        losi_recall = egzaktne[egzaktne["recall"] < 1.0]
        gpu = t[(t["uredjaj"] == "gpu") & t["stanje_gpu"].notna()]
        van_p2 = gpu[~gpu["stanje_gpu"].astype(str).str.startswith("P2")]
        print(f"\nupozorenja: egzaktne sa recall < 1: {len(losi_recall)}, GPU redovi van P2: {len(van_p2)}")
        for _, r in pd.concat([losi_recall, van_p2]).iterrows():
            print(f"  {r['metoda']} N={int(r['N'])} k={r['k']}: recall={r['recall']}, "
                  f"stanje={r['stanje_gpu']} ({r['izvor']})")


def p1_slicnost(p, folder):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for i, (metoda, ime) in enumerate(SLICNOST.items()):
        t = p.izaberi(metoda)
        if t.empty:
            continue
        granica = t["N"] * D * 4 / (p.opseg * 1e9) * 1e3
        a.errorbar(t["N"], t["medijana_ms"], yerr=greska(t), label=ime, capsize=2, **stil(i))
        b.plot(t["N"], t["medijana_ms"] / granica, label=ime, **stil(i))
    N = p.sve.loc[p.sve["metoda"].isin(SLICNOST), "N"].dropna().unique()
    N.sort()
    a.plot(N, N * D * 4 / (p.opseg * 1e9) * 1e3, color=TEKST2, linestyle=":", linewidth=1,
           label=f"donja granica ({zarez(p.opseg)} GB/s)")
    osa_N(a)
    osa_vremena(a)
    a.set_ylabel("vreme po upitu [ms]")
    a.set_title("(a) vreme", loc="left")
    a.legend()
    osa_N(b)
    osa_vremena(b)
    b.axhline(1, color=TEKST2, linestyle=":", linewidth=1)
    b.set_ylabel("vreme / donja granica")
    b.set_title("(b) koliko puta sporije od granice", loc="left")
    fig.tight_layout()
    sacuvaj(fig, folder, "p1_slicnost")


def p2_selekcija(p, folder):
    fig, ose = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
    for ax, N, oznaka in zip(ose, (100_000, N_VELIKO), ("(a) N = $10^5$", "(b) N = $10^6$")):
        for i, (metoda, ime) in enumerate(SELEKCIJA.items()):
            t = p.izaberi(metoda, N=N)
            if not t.empty:
                ax.errorbar(t["k"], t["medijana_ms"], yerr=greska(t), label=ime, capsize=2, **stil(i))
        ax.set_xscale("log")
        k = sorted(p.sve.loc[p.sve["metoda"].isin(SELEKCIJA), "k"].dropna().unique())
        ax.set_xticks(k)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("k")
        osa_vremena(ax)
        ax.set_title(oznaka, loc="left")
    ose[0].set_ylabel("vreme po upitu [ms]")
    ose[0].legend()
    fig.tight_layout()
    sacuvaj(fig, folder, "p2_selekcija")


def p4_prelomna(p, folder):
    fig, ose = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    k_vrednosti = sorted(p.sve.loc[p.sve["serija"] == "prelomna_tacka", "k"].dropna().unique())
    print("  prelomna tačka (S2 + T3 naspram NumPy-ja):")
    for ax, k, slovo in zip(ose, k_vrednosti, "ab"):
        for metoda, (ime, i) in IZGLED_PRETRAGE.items():
            t = p.izaberi(metoda, serija="prelomna_tacka", k=k)
            if not t.empty:
                ax.errorbar(t["N"], t["medijana_ms"], yerr=greska(t), label=ime, capsize=2, **stil(i))
        gpu = p.izaberi("gpu_pretraga", serija="prelomna_tacka", k=k).set_index("N")["medijana_ms"]
        cpu = p.izaberi("cpu_numpy", serija="prelomna_tacka", k=k).set_index("N")["medijana_ms"]
        zajedno = gpu.index.intersection(cpu.index)
        brzi = (gpu[zajedno] < cpu[zajedno]).sort_index()
        sporiji = brzi[~brzi]
        if not sporiji.empty and sporiji.index.max() < brzi.index.max():
            levo = sporiji.index.max()
            desno = brzi.index[brzi.index > levo].min()
            ax.axvspan(levo, desno, color=OSA, alpha=0.4, linewidth=0, label="prelomna tačka (interval)")
            print(f"    k = {int(k)}: između N = {int(levo)} i N = {int(desno)}")
        osa_N(ax)
        osa_vremena(ax)
        ax.set_title(f"({slovo}) k = {int(k)}", loc="left")
    ose[0].set_ylabel("vreme po upitu [ms]")
    ose[0].legend()
    fig.tight_layout()
    sacuvaj(fig, folder, "p4_prelomna_tacka")


def p5_poredjenje(p, folder):
    redovi = []
    for k in (10, 100):
        for metoda, (ime, i) in IZGLED_PRETRAGE.items():
            if metoda == "cpu_hnsw":
                continue
            t = p.izaberi(metoda, N=N_VELIKO, k=k)
            if not t.empty:
                r = t.iloc[0]
                redovi.append({"metoda": metoda, "naziv": ime, "k": k, "medijana_ms": r["medijana_ms"],
                               "min_ms": r["min_ms"], "max_ms": r["max_ms"], "recall": r["recall"],
                               "izvor": r["izvor"], "boja": i})
    tabela = pd.DataFrame(redovi)
    tabela.drop(columns="boja").to_csv(folder.parent / "p5_tabela.csv", index=False)
    print(f"  {folder.parent.name}/p5_tabela.csv")

    t = tabela[tabela["k"] == 10].sort_values("medijana_ms", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    ax.barh(t.index, t["medijana_ms"], color=[BOJE[i] for i in t["boja"]], height=0.6,
            xerr=greska(t), capsize=2, error_kw={"ecolor": TEKST2, "elinewidth": 0.8})
    for y, r in t.iterrows():
        ax.text(r["max_ms"] * 1.15, y, f"{zarez(r['medijana_ms'], 2)} ms", va="center", fontsize=7)
    ax.set_yticks(t.index, t["naziv"])
    osa_vremena(ax, "x")
    ax.set_xlim(right=t["max_ms"].max() * 4)
    ax.set_xlabel("vreme po upitu [ms], N = $10^6$, k = 10")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    sacuvaj(fig, folder, "p5_poredjenje")


def p6_egzaktno_hnsw(p, folder):
    fig, ose = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
    for ax, k, slovo in zip(ose, (10, 100), "ab"):
        h = p.izaberi("cpu_hnsw", N=N_VELIKO, k=k)
        if not h.empty:
            ime, i = IZGLED_PRETRAGE["cpu_hnsw"]
            ax.errorbar(h["medijana_ms"], h["recall"], xerr=greska(h), label=ime, capsize=2, **stil(i))
            for j, (_, r) in enumerate(h.iterrows()):
                pomak, poravnanje = ((5, -9), "left") if j % 2 == 0 else ((-5, 4), "right")
                ax.annotate(f"ef={int(r['ef_search'])}", (r["medijana_ms"], r["recall"]), ha=poravnanje,
                            textcoords="offset points", xytext=pomak, fontsize=6, color=TEKST2)
        for metoda in ("gpu_pretraga", "cpu_numpy", "cpu_faiss_flat"):
            t = p.izaberi(metoda, N=N_VELIKO, k=k)
            if not t.empty:
                ime, i = IZGLED_PRETRAGE[metoda]
                s = stil(i)
                s["linestyle"] = "none"
                ax.errorbar(t["medijana_ms"], t["recall"], xerr=greska(t), label=ime, capsize=2, **s)
        osa_vremena(ax, "x")
        ax.set_xlabel("vreme po upitu [ms]")
        ax.set_title(f"({slovo}) k = {k}", loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: zarez(round(v, 3))))
    dno = p.sve.loc[(p.sve["metoda"] == "cpu_hnsw") & (p.sve["N"] == N_VELIKO), "recall"].min()
    ose[0].set_ylim(bottom=dno - 0.015)
    ose[0].set_ylabel("recall@k")
    ose[0].legend(loc="lower right")
    fig.tight_layout()
    sacuvaj(fig, folder, "p6_egzaktno_hnsw")


def p7_rag(p, folder):
    varijante = [("cpu_numpy", "NumPy (CPU)"), ("gpu_pretraga", "S2 + T3 (GPU)"), ("cpu_hnsw", "HNSW (CPU)")]
    faze = [("ugradnja", "ugradnja upita (model)", BOJE[2]), ("pretraga", "pretraga", BOJE[0]),
            ("tekst", "čitanje teksta", BOJE[3])]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    nacrtane = []
    najvise = 0.0
    for y, (varijanta, ime) in enumerate(varijante):
        ukupno = p.izaberi(f"rag_{varijanta}_ukupno", serija="rag")
        if ukupno.empty:
            continue
        levo = 0.0
        for faza, oznaka, boja in faze:
            t = p.izaberi(f"rag_{varijanta}_{faza}", serija="rag")
            sirina = t["medijana_ms"].iloc[0]
            ax.barh(y, sirina, left=levo, color=boja, height=0.45, edgecolor="white", linewidth=1,
                    label=oznaka if y == 0 else None)
            levo += sirina
        u = ukupno.iloc[0]
        ax.errorbar(u["medijana_ms"], y, xerr=[[u["medijana_ms"] - u["min_ms"]], [u["max_ms"] - u["medijana_ms"]]],
                    color=TEKST2, capsize=2, elinewidth=0.8, linestyle="none")
        ax.text(0, y - 0.3, f"ukupno {zarez(u['medijana_ms'], 1)} ms, "
                f"pretraga {zarez(100 * u['udeo_pretrage'], 1)} %", va="bottom", fontsize=7)
        nacrtane.append((y, ime))
        najvise = max(najvise, u["max_ms"])
    ax.set_yticks([y for y, _ in nacrtane], [ime for _, ime in nacrtane])
    ax.set_ylim(len(varijante) - 0.5, -0.9)
    ax.set_xlim(0, najvise * 1.03)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: zarez(v)))
    ax.set_xlabel("vreme po upitu [ms]; trake: medijane faza, traka greške: raspon ukupnog vremena")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    sacuvaj(fig, folder, "p7_rag")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rezultati", nargs="+", required=True,
                        help="CSV-ovi iz run_all.py; kasniji fajl zamenjuje isti red iz ranijeg")
    parser.add_argument("--izlaz", default=str(Path(__file__).parent / "figures"))
    argumenti = parser.parse_args()

    folder = Path(argumenti.izlaz)
    folder.mkdir(parents=True, exist_ok=True)
    podesi_izgled()
    p = Podaci(argumenti.rezultati)
    print(f"teorijski opseg kartice (stvarni takt): {p.opseg} GB/s\ngrafikoni u {folder}:")
    for crtaj in (p1_slicnost, p2_selekcija, p4_prelomna, p5_poredjenje, p6_egzaktno_hnsw, p7_rag):
        crtaj(p, folder)
    p.izvestaj()


if __name__ == "__main__":
    main()
