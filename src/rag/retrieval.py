import numpy as np
import pandas as pd


def ucitaj_model(ime, uredjaj):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ime, device=uredjaj)


def ugradi(model, tekst):
    return model.encode(tekst, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32, copy=False)


def napravi_skladiste(putanja_pasusa, N):
    pasusi = pd.read_parquet(putanja_pasusa).iloc[:N]
    return dict(zip(pasusi["pid"].to_numpy().tolist(), pasusi["text"].tolist()))


def procitaj_tekstove(skladiste, pid_niz, indeksi):
    return [skladiste[int(pid)] for pid in pid_niz[np.asarray(indeksi)]]
