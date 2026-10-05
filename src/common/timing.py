import statistics
import time
from dataclasses import dataclass, asdict


@dataclass
class Merenje:
    medijana_ms: float
    min_ms: float
    max_ms: float
    ponavljanja: int
    warmup: int
    uredjaj: str

    @property
    def raspon_ms(self):
        return self.max_ms - self.min_ms

    def propusni_opseg_gbps(self, bajtova):
        return bajtova / (self.medijana_ms * 1e-3) / 1e9

    def kao_recnik(self):
        return asdict(self)

    def __str__(self):
        return (f"{self.medijana_ms:.4f} ms "
                f"(min {self.min_ms:.4f}, max {self.max_ms:.4f}, "
                f"n={self.ponavljanja}, {self.uredjaj})")


def _rezime(vremena_ms, ponavljanja, warmup, uredjaj):
    return Merenje(
        medijana_ms=float(statistics.median(vremena_ms)),
        min_ms=float(min(vremena_ms)),
        max_ms=float(max(vremena_ms)),
        ponavljanja=ponavljanja,
        warmup=warmup,
        uredjaj=uredjaj,
    )


def izmeri_cpu(funkcija, warmup=5, ponavljanja=20):
    for _ in range(warmup):
        funkcija()

    vremena_ms = []
    for _ in range(ponavljanja):
        pocetak = time.perf_counter()
        funkcija()
        kraj = time.perf_counter()
        vremena_ms.append((kraj - pocetak) * 1e3)

    return _rezime(vremena_ms, ponavljanja, warmup, "cpu")


def _zagrej_gpu(pozovi, warmup, zagrevanje_ms):
    import cupy as cp

    pocetak = time.perf_counter()
    broj = 0
    while broj < warmup or (time.perf_counter() - pocetak) * 1e3 < zagrevanje_ms:
        pozovi(broj)
        cp.cuda.Stream.null.synchronize()
        broj += 1
    return broj


def izmeri_gpu(funkcija, warmup=5, ponavljanja=20, zagrevanje_ms=0):
    import cupy as cp

    warmup = _zagrej_gpu(lambda i: funkcija(), warmup, zagrevanje_ms)

    pocetak = cp.cuda.Event()
    kraj = cp.cuda.Event()

    vremena_ms = []
    for _ in range(ponavljanja):
        pocetak.record()
        funkcija()
        kraj.record()
        kraj.synchronize()
        vremena_ms.append(float(cp.cuda.get_elapsed_time(pocetak, kraj)))

    return _rezime(vremena_ms, ponavljanja, warmup, "gpu")


def izmeri_cpu_po_upitima(funkcija, upiti, warmup=5, ponavljanja=20):
    Q = len(upiti)
    for i in range(warmup):
        funkcija(upiti[i % Q])

    vremena_ms = []
    for i in range(ponavljanja):
        upit = upiti[i % Q]
        pocetak = time.perf_counter()
        funkcija(upit)
        kraj = time.perf_counter()
        vremena_ms.append((kraj - pocetak) * 1e3)

    return _rezime(vremena_ms, ponavljanja, warmup, "cpu")


def izmeri_gpu_po_upitima(funkcija, upiti, warmup=5, ponavljanja=20, zagrevanje_ms=0):
    import cupy as cp

    Q = len(upiti)
    warmup = _zagrej_gpu(lambda i: funkcija(upiti[i % Q]), warmup, zagrevanje_ms)

    pocetak = cp.cuda.Event()
    kraj = cp.cuda.Event()

    vremena_ms = []
    for i in range(ponavljanja):
        upit = upiti[i % Q]
        pocetak.record()
        funkcija(upit)
        kraj.record()
        kraj.synchronize()
        vremena_ms.append(float(cp.cuda.get_elapsed_time(pocetak, kraj)))

    return _rezime(vremena_ms, ponavljanja, warmup, "gpu")


_ogranicenje_niti = None


def postavi_niti_procesora(broj_niti):
    global _ogranicenje_niti
    from threadpoolctl import threadpool_limits

    _ogranicenje_niti = threadpool_limits(limits=broj_niti)

    try:
        import faiss
        faiss.omp_set_num_threads(broj_niti)
    except ImportError:
        pass


def _ime_procesora():
    try:
        import winreg
        kljuc = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(kljuc, "ProcessorNameString")[0].strip()
    except Exception:
        import platform
        return platform.processor()


def stanje_kartice():
    izlaz = _nvidia_smi("pstate,clocks.gr,clocks.mem")
    try:
        stanje, graficki, memorija = [deo.strip().replace(" MHz", "") for deo in izlaz.split(",")]
        return f"{stanje} {graficki}/{memorija} MHz"
    except ValueError:
        return izlaz


def _nvidia_smi(polja):
    import subprocess
    try:
        izlaz = subprocess.run(
            ["nvidia-smi", f"--query-gpu={polja}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return izlaz.stdout.strip()
    except Exception:
        return "nedostupno"


def gpu_je_dostupan():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def podaci_o_okruzenju():
    import os
    import platform
    import numpy as np

    podaci = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "procesor": _ime_procesora(),
        "logicka_jezgra": os.cpu_count(),
        "numpy": np.__version__,
    }

    try:
        from threadpoolctl import threadpool_info
        podaci["blas"] = "; ".join(
            f'{biblioteka["internal_api"]} {biblioteka.get("version")} '
            f'niti={biblioteka["num_threads"]}'
            for biblioteka in threadpool_info()
        )
    except Exception:
        podaci["blas"] = "nepoznato"

    try:
        import cupy as cp
        osobine = cp.cuda.runtime.getDeviceProperties(0)
        podaci["cupy"] = cp.__version__
        podaci["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        podaci["gpu"] = osobine["name"].decode()
        podaci["compute_capability"] = f'{osobine["major"]}.{osobine["minor"]}'
        podaci["cupy_tf32"] = os.environ.get("CUPY_TF32", "nije postavljeno")
        podaci["drajver_i_maks_takt"] = _nvidia_smi(
            "driver_version,clocks.max.graphics,clocks.max.memory")
    except Exception:
        podaci["cupy"] = "nedostupno"

    try:
        import torch
        podaci["torch"] = torch.__version__
        podaci["tf32"] = torch.backends.cuda.matmul.allow_tf32
    except Exception:
        podaci["torch"] = "nedostupno"

    try:
        import faiss
        podaci["faiss"] = faiss.__version__
        podaci["faiss_gpu"] = faiss.get_num_gpus()
        podaci["faiss_niti"] = faiss.omp_get_max_threads()
    except Exception:
        podaci["faiss"] = "nedostupno"

    return podaci


if __name__ == "__main__":
    import numpy as np

    a = np.random.rand(2000, 768).astype(np.float32)
    b = np.random.rand(768).astype(np.float32)

    merenje = izmeri_cpu(lambda: a @ b, warmup=5, ponavljanja=20)
    print("CPU:", merenje)
    print("procitano bajtova:", a.nbytes)
    print("efektivni opseg:", round(merenje.propusni_opseg_gbps(a.nbytes), 2), "GB/s")

    print("GPU dostupan:", gpu_je_dostupan())

    if gpu_je_dostupan():
        import cupy as cp
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        print("GPU:", izmeri_gpu(lambda: a_gpu @ b_gpu))

    print()
    for kljuc, vrednost in podaci_o_okruzenju().items():
        print(f"{kljuc}: {vrednost}")
