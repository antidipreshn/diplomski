import argparse
import sys
from pathlib import Path

import yaml

KOREN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOREN))

import cupy as cp
import torch

from src.baselines.torch_ref import proveri_tf32
from src.common.data import ucitaj_podatke
from src.cuda.slicnost import slicnost_s2
from src.cuda.topk import pripremi_t3, topk_t3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slucaj", choices=["s2", "t3", "torch_topk"], required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--config", default=str(KOREN / "bench" / "config.yaml"))
    argumenti = parser.parse_args()

    with open(argumenti.config, encoding="utf-8") as fajl:
        konfiguracija = yaml.safe_load(fajl)
    proveri_tf32()
    N = max(konfiguracija["podaci"]["N"])
    k = argumenti.k
    kerneli = konfiguracija["kerneli"]

    korpus, upiti, _ = ucitaj_podatke(konfiguracija, N)
    korpus_gpu, upit_gpu = cp.asarray(korpus), cp.asarray(upiti[0])
    del korpus
    slicnosti = cp.empty(N, dtype=cp.float32)
    slicnost_s2(korpus_gpu, upit_gpu, slicnosti, kerneli["s2"])

    if argumenti.slucaj == "s2":
        def pozovi():
            slicnost_s2(korpus_gpu, upit_gpu, slicnosti, kerneli["s2"])
    elif argumenti.slucaj == "t3":
        radni = pripremi_t3(N, k, kerneli["t3"])

        def pozovi():
            topk_t3(slicnosti, k, kerneli["t3"], radni)
    else:
        slicnosti_t = torch.as_tensor(slicnosti, device="cuda")

        def pozovi():
            torch.topk(slicnosti_t, k)

    pozovi()
    cp.cuda.Device().synchronize()
    torch.cuda.nvtx.range_push("profil")
    pozovi()
    cp.cuda.Device().synchronize()
    torch.cuda.nvtx.range_pop()
    print(f"{argumenti.slucaj}: N={N}, k={k}, gotovo")


if __name__ == "__main__":
    main()
