import cupy as cp
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def proveri_tf32():
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 je ukljucen u PyTorch-u (invarijanta 1)")


def proveri_stream():
    cupy_stream = cp.cuda.get_current_stream().ptr
    torch_stream = torch.cuda.current_stream().cuda_stream
    if cupy_stream != 0 or torch_stream != 0:
        raise RuntimeError(f"CuPy i torch nisu na default stream-u ({cupy_stream}, {torch_stream})")


def pretrazi_torch(korpus, upit, k):
    korpus_t = torch.as_tensor(korpus, device="cuda")
    upit_t = torch.as_tensor(upit, device="cuda")
    vrednosti, indeksi = torch.topk(torch.mv(korpus_t, upit_t), k)
    return cp.from_dlpack(indeksi), cp.from_dlpack(vrednosti)


def pretrazi_torch_kompletna(korpus, upit, k):
    korpus_t = torch.as_tensor(korpus, device="cuda")
    upit_t = torch.from_numpy(upit).to("cuda")
    vrednosti, indeksi = torch.topk(torch.mv(korpus_t, upit_t), k)
    return indeksi.cpu().numpy(), vrednosti.cpu().numpy()


proveri_tf32()


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.common.data import napravi_korpus, napravi_upite
    from src.common.timing import izmeri_gpu_po_upitima
    from src.common.verify import proveri_egzaktnost
    from src.cpu.numpy_ref import pretrazi_numpy

    N, d, Q = 100_000, 768, 20
    korpus = napravi_korpus(N, d, seed=0)
    upiti = napravi_upite(Q, d, seed=1)
    korpus_gpu, upiti_gpu = cp.asarray(korpus), cp.asarray(upiti)

    proveri_tf32()
    proveri_stream()
    print("TF32 iskljucen, CuPy i torch na default stream-u")

    isti_blok = torch.as_tensor(korpus_gpu, device="cuda").data_ptr() == korpus_gpu.data.ptr
    print("CuPy -> torch bez kopiranja:", isti_blok)

    for k in (1, 10, 100):
        for q in range(Q):
            indeksi, vrednosti = pretrazi_torch(korpus_gpu, upiti_gpu[q], k)
            ref_ind, ref_vr = pretrazi_numpy(korpus, upiti[q], k)
            proveri_egzaktnost(cp.asnumpy(indeksi), cp.asnumpy(vrednosti), ref_ind, ref_vr,
                               naziv=f"torch (upit {q}, k={k})",
                               slicnost_indeksa=lambda ind, q=q: korpus[ind] @ upiti[q])
        print(f"k={k:<3} egzaktno na {Q} upita")

    for k in (1, 10, 100):
        for q in range(Q):
            indeksi, vrednosti = pretrazi_torch_kompletna(korpus_gpu, upiti[q], k)
            ref_ind, ref_vr = pretrazi_numpy(korpus, upiti[q], k)
            proveri_egzaktnost(indeksi, vrednosti, ref_ind, ref_vr,
                               naziv=f"torch kompletna (upit {q}, k={k})",
                               slicnost_indeksa=lambda ind, q=q: korpus[ind] @ upiti[q])
        print(f"k={k:<3} kompletna: egzaktno na {Q} upita")

    k = 100
    preko_events = izmeri_gpu_po_upitima(lambda u: pretrazi_torch(korpus_gpu, u, k), upiti_gpu,
                                         warmup=5, ponavljanja=100, zagrevanje_ms=1000)
    vremena = []
    for i in range(100):
        torch.cuda.synchronize()
        pocetak = time.perf_counter()
        pretrazi_torch(korpus_gpu, upiti_gpu[i % Q], k)
        torch.cuda.synchronize()
        vremena.append((time.perf_counter() - pocetak) * 1e3)
    vremena.sort()
    preko_sata = (vremena[49] + vremena[50]) / 2
    print(f"\nN={N}, k={k}: events {preko_events.medijana_ms:.4f} ms, "
          f"synchronize + perf_counter {preko_sata:.4f} ms")
    print("events obuhvataju torch kernele:", preko_events.medijana_ms > 0.8 * preko_sata)

    kompletna = izmeri_gpu_po_upitima(lambda u: pretrazi_torch_kompletna(korpus_gpu, u, k), upiti,
                                      warmup=5, ponavljanja=100, zagrevanje_ms=1000)
    print(f"kompletna (upit iz RAM-a, rezultat u RAM): {kompletna.medijana_ms:.4f} ms")
