from pathlib import Path

import cupy as cp

DIREKTORIJUM = Path(__file__).resolve().parent
_kes = {}


def ucitaj_modul(ime_fajla, makroi=None):
    makroi = dict(makroi or {})
    kljuc = (ime_fajla, tuple(sorted(makroi.items())))
    if kljuc not in _kes:
        kod = (DIREKTORIJUM / ime_fajla).read_text(encoding="utf-8")
        opcije = ["-std=c++17", f"-I{DIREKTORIJUM}"]
        opcije += [f"-D{ime}={vrednost}" for ime, vrednost in sorted(makroi.items())]
        _kes[kljuc] = cp.RawModule(code=kod, options=tuple(opcije))
    return _kes[kljuc]


def ucitaj_kernel(ime_fajla, ime_kernela, makroi=None):
    return ucitaj_modul(ime_fajla, makroi).get_function(ime_kernela)
