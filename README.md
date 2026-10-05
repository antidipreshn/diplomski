# Ubrzanje pretrage sličnosti u RAG sistemima korišćenjem CUDA tehnologije

Kod za diplomski rad. Egzaktna top-k pretraga embedding vektora (kosinusna sličnost) na grafičkoj kartici, sa sopstvenim CUDA kernelima. Rezultati se porede sa NumPy-jem, PyTorch-om, FAISS-Flat i FAISS-HNSW.

## Sadržaj

- `src/cuda/` – CUDA kerneli i kod koji ih pokreće
  - sličnost: `similarity_s1_naive.cu`, `similarity_s2_warp.cu`, `similarity_s3_shared.cu`, `similarity_s4_vectorized.cu`, `similarity_s5_cublas.py` (cuBLAS)
  - selekcija k najvećih: `topk_t1_atomic.cu`, `topk_t2_block.cu`, `topk_t3_warp.cu`, `topk_t4_threshold.cu`
  - `pretraga.py` – kompletna pretraga na kartici (S2 + T3)
- `src/cpu/` – NumPy referenca (ground truth) i naivna implementacija
- `src/baselines/` – PyTorch, FAISS-Flat, FAISS-HNSW
- `src/rag/` – retrieval deo RAG-a (tekst upita → vektor → pretraga → tekst pasusa)
- `src/common/` – podaci, provera tačnosti, merenje vremena
- `bench/` – merenja (`run_all.py`), propusni opseg, profilisanje; parametri u `config.yaml`
- `bench/results/` – rezultati merenja (CSV i podaci o okruženju)
- `analysis/` – grafici i tabele iz rezultata
- `data/` – skripte za preuzimanje MS MARCO pasusa i pravljenje embedinga

## Okruženje

- Windows 10, NVIDIA RTX 3080 (10 GB, compute capability 8.6), CUDA 12.6
- Python 3.12, NumPy, CuPy, PyTorch, faiss-cpu, sentence-transformers, datasets, PyYAML, threadpoolctl
- Kerneli se prevode u toku izvršavanja preko CuPy `RawModule` (NVRTC), bez `nvcc`.

## Podaci

Podaci nisu u repozitorijumu. Prave se skriptama:

```
python data/download_msmarco.py
python data/build_embeddings.py
```

Korpus je 10^6 MS MARCO pasusa i 1000 upita, model `all-mpnet-base-v2` (d = 768), vektori su normalizovani i u fp32.

## Pokretanje

```
python bench/run_all.py --config bench/config.yaml
python analysis/plots.py --rezultati bench/results/<fajl>.csv
```

Pre merenja se svaka egzaktna metoda proverava u odnosu na NumPy referencu. Na kartici se vreme meri CUDA events, na procesoru `perf_counter`. Prijavljuje se medijana i raspon.
