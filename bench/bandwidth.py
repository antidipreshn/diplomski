import subprocess
import sys
from pathlib import Path

import yaml

KOREN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOREN))

import cupy as cp

from src.common.timing import izmeri_gpu

KOPIRAJ = cp.RawKernel(r'''
extern "C" __global__
void kopiraj(const float4* ulaz, float4* izlaz, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        izlaz[i] = ulaz[i];
    }
}
''', "kopiraj")


def teorijski_opseg_gbps(takt_memorije_mhz=None, uredjaj=0):
    osobine = cp.cuda.Device(uredjaj).attributes
    if takt_memorije_mhz is None:
        takt_hz = osobine["MemoryClockRate"] * 1e3
    else:
        takt_hz = takt_memorije_mhz * 1e6
    sirina_bita = osobine["GlobalMemoryBusWidth"]
    return takt_hz * 2 * (sirina_bita / 8) / 1e9


def trenutni_takt_memorije_mhz():
    izlaz = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.mem,pstate", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    takt, stanje = [deo.strip() for deo in izlaz.split(",")]
    return float(takt), stanje


def izmeri_opseg_gbps(velicina_mb, warmup, ponavljanja, zagrevanje_ms):
    broj_float4 = int(velicina_mb * 1024 * 1024 / 16)

    ulaz = cp.random.rand(broj_float4 * 4, dtype=cp.float32)
    izlaz = cp.empty_like(ulaz)

    velicina_bloka = 256
    broj_blokova = (broj_float4 + velicina_bloka - 1) // velicina_bloka

    def pokreni():
        KOPIRAJ((broj_blokova,), (velicina_bloka,),
                (ulaz, izlaz, cp.int32(broj_float4)))

    merenje = izmeri_gpu(pokreni, warmup=warmup, ponavljanja=ponavljanja,
                         zagrevanje_ms=zagrevanje_ms)
    takt_mhz, stanje = trenutni_takt_memorije_mhz()
    bajtova = ulaz.nbytes * 2

    del ulaz, izlaz
    cp.get_default_memory_pool().free_all_blocks()

    return {
        "merenje": merenje,
        "izmereni_gbps": merenje.propusni_opseg_gbps(bajtova),
        "teorijski_maks_gbps": teorijski_opseg_gbps(),
        "teorijski_pri_taktu_gbps": teorijski_opseg_gbps(takt_mhz),
        "takt_memorije_mhz": takt_mhz,
        "stanje": stanje,
    }


def izmeri_opseg_cpu_gbps(velicina_mb, niti, warmup, ponavljanja):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits

    from src.common.timing import izmeri_cpu

    niz = np.ones(int(velicina_mb * 1024 * 1024 / 4), dtype=np.float32)
    delovi = np.array_split(niz, niti)
    with threadpool_limits(1), ThreadPoolExecutor(max_workers=niti) as izvrsilac:
        merenje = izmeri_cpu(lambda: list(izvrsilac.map(lambda d: np.dot(d, d), delovi)),
                             warmup=warmup, ponavljanja=ponavljanja)
    bajtova = niz.nbytes
    del niz, delovi
    return {"merenje": merenje, "izmereni_gbps": merenje.propusni_opseg_gbps(bajtova)}


def teorijski_opseg_cpu_gbps():
    izlaz = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_PhysicalMemory | ForEach-Object { \"$($_.ConfiguredClockSpeed);$($_.DeviceLocator)\" }"],
        capture_output=True, text=True, timeout=30,
    ).stdout.split()
    brzine = {int(red.split(";")[0]) for red in izlaz}
    kanali = {red.split(";")[1].split("-")[0] for red in izlaz}
    brzina = min(brzine)
    return brzina * 8 * len(kanali) / 1e3, f"{len(izlaz)} modula, {brzina} MT/s, {len(kanali)} kanala"


if __name__ == "__main__":
    with open(KOREN / "bench" / "config.yaml", "r", encoding="utf-8") as fajl:
        konfiguracija = yaml.safe_load(fajl)
    parametri = konfiguracija["propusni_opseg"]

    cpu = konfiguracija["propusni_opseg_cpu"]
    for niti in sorted({1, konfiguracija["merenje"]["niti_procesora"]}):
        rezultat_cpu = izmeri_opseg_cpu_gbps(cpu["velicina_mb"], niti, cpu["warmup"], cpu["ponavljanja"])
        print(f"procesor, niti={niti}: {rezultat_cpu['izmereni_gbps']:.1f} GB/s  {rezultat_cpu['merenje']}")
    teorijski_cpu, opis_ram = teorijski_opseg_cpu_gbps()
    print(f"procesor, teorijski: {teorijski_cpu:.1f} GB/s ({opis_ram})")

    rezultat = izmeri_opseg_gbps(**parametri)
    izmereni = rezultat["izmereni_gbps"]

    print("kartica:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print("merenje:", rezultat["merenje"])
    print("izmereni opseg:", round(izmereni, 1), "GB/s")
    print("teorijski (maks. takt memorije):",
          round(rezultat["teorijski_maks_gbps"], 1), "GB/s,",
          "udeo", round(100 * izmereni / rezultat["teorijski_maks_gbps"], 1), "%")
    print(f"teorijski (stvarni takt {rezultat['takt_memorije_mhz']:.0f} MHz, "
          f"{rezultat['stanje']}):", round(rezultat["teorijski_pri_taktu_gbps"], 1), "GB/s,",
          "udeo", round(100 * izmereni / rezultat["teorijski_pri_taktu_gbps"], 1), "%")
