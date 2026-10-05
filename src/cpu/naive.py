import numpy as np
import numpy.f2py.f2py2e


def slicnosti_naivno(korpus, upit):
    N, d = korpus.shape
    rezultat = [0.0] * N

    for i in range(N):
        red = korpus[i]
        zbir = 0.0
        for j in range(d):
            zbir += float(red[j]) * float(upit[j])
        rezultat[i] = zbir

    return np.array(rezultat, dtype=np.float32)


def top_k_naivno(slicnosti, k):
    N = len(slicnosti)
    if k > N:
        raise ValueError(f"k = {k} je vece od broja vektora N = {N}")

    parovi = [(float(slicnosti[i]), i) for i in range(N)]
    parovi.sort(key=lambda par: par[0], reverse=True)
    najboljih_k = parovi[:k]

    indeksi = np.array([i for _, i in najboljih_k], dtype=np.int64)
    vrednosti = np.array([s for s, _ in najboljih_k], dtype=np.float32)
    return indeksi, vrednosti


def pretrazi_naivno(korpus, upit, k):
    return top_k_naivno(slicnosti_naivno(korpus, upit), k)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.common.data import napravi_korpus, napravi_upit
    from src.common.verify import proveri_egzaktnost
    from src.cpu.numpy_ref import pretrazi_numpy

    N, d, k = 2000, 768, 10
    korpus = napravi_korpus(N, d)
    upit = napravi_upit(d)

    ind_naivno, vrd_naivno = pretrazi_naivno(korpus, upit, k)
    ind_numpy, vrd_numpy = pretrazi_numpy(korpus, upit, k)

    razlika, _, _ = proveri_egzaktnost(
        ind_naivno, vrd_naivno, ind_numpy, vrd_numpy, naziv="naivna petlja"
    )

    print(f"N = {N}, d = {d}, k = {k}")
    print("naivna i NumPy implementacija daju isti rezultat")
    print(f"najveca razlika u vrednostima: {razlika:.3e}")
    print("prvih 5 indeksa:", ind_naivno[:5])
    print("prvih 5 vrednosti:", vrd_naivno[:5])

    print("jaaaaaaaa: ", (np.show_config(mode="dicts")))
