import numpy as np

TOLERANCIJA = 1e-5


def uporedi_vrednosti(vrednosti, referentne, tolerancija=TOLERANCIJA):
    vrednosti = np.asarray(vrednosti, dtype=np.float64)
    referentne = np.asarray(referentne, dtype=np.float64)

    if vrednosti.shape != referentne.shape:
        raise ValueError(
            f"razliciti oblici: {vrednosti.shape} i {referentne.shape}"
        )

    najveca_razlika = float(np.max(np.abs(vrednosti - referentne)))
    return najveca_razlika <= tolerancija, najveca_razlika


def recall_at_k(indeksi, referentni_indeksi):
    nadjeni = set(np.asarray(indeksi).ravel().tolist())
    tacni = set(np.asarray(referentni_indeksi).ravel().tolist())

    if not tacni:
        raise ValueError("referentni skup je prazan")

    return len(nadjeni & tacni) / len(tacni)


def proveri_egzaktnost(indeksi, vrednosti, ref_indeksi, ref_vrednosti,
                       tolerancija=TOLERANCIJA, naziv="metoda",
                       slicnost_indeksa=None):
    indeksi = np.asarray(indeksi).ravel()
    ref_indeksi = np.asarray(ref_indeksi).ravel()

    if indeksi.shape != ref_indeksi.shape:
        raise AssertionError(
            f"{naziv}: vraceno {indeksi.size} indeksa, ocekivano {ref_indeksi.size}"
        )
    if np.unique(indeksi).size != indeksi.size:
        raise AssertionError(f"{naziv}: isti indeks vracen vise puta")

    poklapaju_se, razlika = uporedi_vrednosti(vrednosti, ref_vrednosti, tolerancija)
    if not poklapaju_se:
        raise AssertionError(
            f"{naziv}: vrednosti se ne poklapaju, najveca razlika {razlika:.3e}"
        )

    recall = recall_at_k(indeksi, ref_indeksi)
    zamene = np.setdiff1d(indeksi, ref_indeksi)

    if zamene.size > 0:
        if slicnost_indeksa is None:
            raise AssertionError(
                f"{naziv}: recall@k = {recall:.3f}, a mora biti 1.0"
            )
        granica = float(np.asarray(ref_vrednosti).ravel()[-1])
        prave = np.asarray(slicnost_indeksa(zamene), dtype=np.float64)
        odstupanje = float(np.max(np.abs(prave - granica)))
        if odstupanje > tolerancija:
            raise AssertionError(
                f"{naziv}: recall@k = {recall:.3f}; zamenjeni indeksi {zamene.tolist()} "
                f"nisu na granici (odstupanje od k-te vrednosti {odstupanje:.3e})"
            )

    return razlika, recall, int(zamene.size)


def recall_aproksimativno(indeksi, ref_indeksi, ref_vrednosti, slicnost_indeksa,
                          tolerancija=TOLERANCIJA):
    indeksi = np.asarray(indeksi).ravel()
    ref_indeksi = np.asarray(ref_indeksi).ravel()
    indeksi = np.unique(indeksi[indeksi >= 0])

    pogoci = np.intersect1d(indeksi, ref_indeksi).size
    ostali = np.setdiff1d(indeksi, ref_indeksi)
    zamene = 0
    if ostali.size > 0:
        granica = float(np.asarray(ref_vrednosti).ravel()[-1])
        prave = np.asarray(slicnost_indeksa(ostali), dtype=np.float64)
        zamene = int(np.sum(prave >= granica - tolerancija))

    return min(1.0, (pogoci + zamene) / ref_indeksi.size), zamene


if __name__ == "__main__":
    ref_ind = np.array([7, 3, 1])
    ref_vrd = np.array([0.91, 0.88, 0.85], dtype=np.float32)
    prave_slicnosti = {7: 0.91, 3: 0.88, 1: 0.85, 9: 0.85 + 1e-7, 4: 0.60}

    def slicnost(indeksi):
        return [prave_slicnosti[int(i)] for i in indeksi]

    print("ista vrednost:", uporedi_vrednosti(ref_vrd + 1e-7, ref_vrd))
    print("tacan rezultat:", proveri_egzaktnost(
        ref_ind, ref_vrd + 1e-7, ref_ind, ref_vrd, naziv="test"))

    print("zamena na granici (9 umesto 1):", proveri_egzaktnost(
        np.array([7, 3, 9]), ref_vrd, ref_ind, ref_vrd,
        naziv="granica", slicnost_indeksa=slicnost))

    for opis, ind in [("pogresan indeks, dobra vrednost (4 umesto 1)", [7, 3, 4]),
                      ("zamena bez funkcije slicnosti", [7, 3, 9]),
                      ("dupli indeks", [7, 7, 1])]:
        try:
            proveri_egzaktnost(np.array(ind), ref_vrd, ref_ind, ref_vrd,
                               naziv=opis,
                               slicnost_indeksa=None if "bez" in opis else slicnost)
            print("GRESKA: provera je prosla za:", opis)
        except AssertionError as greska:
            print("odbijeno kako treba:", greska)

    print("aproksimativno, 1 promasaj (ocekivano 0.667, 0):",
          recall_aproksimativno([7, 3, 4], ref_ind, ref_vrd, slicnost))
    print("aproksimativno, zamena na granici (ocekivano 1.0, 1):",
          recall_aproksimativno([7, 3, 9], ref_ind, ref_vrd, slicnost))
