import platform
import numpy as np
import cupy as cp
import torch
import faiss

print("OS:", platform.platform())
print("Python:", platform.python_version())
print("NumPy:", np.__version__)

print("CuPy:", cp.__version__)
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
props = cp.cuda.runtime.getDeviceProperties(0)
print("GPU:", props["name"].decode())
print("Compute capability:", f'{props["major"]}.{props["minor"]}')
free, total = cp.cuda.runtime.memGetInfo()
print(f"GPU memorija: {total / 1e9:.1f} GB ukupno, {free / 1e9:.1f} GB slobodno")

kod = r'''
extern "C" __global__
void saberi(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
'''
kernel = cp.RawKernel(kod, "saberi")

n = 1_000_000
a = cp.random.rand(n, dtype=cp.float32)
b = cp.random.rand(n, dtype=cp.float32)
c = cp.empty_like(a)

velicina_bloka = 256
broj_blokova = (n + velicina_bloka - 1) // velicina_bloka
kernel((broj_blokova,), (velicina_bloka,), (a, b, c, np.int32(n)))
cp.cuda.Stream.null.synchronize()

print("Kernel se kompajlira i racuna tacno:", bool(cp.allclose(c, a + b)))

print("PyTorch:", torch.__version__)
print("PyTorch vidi GPU:", torch.cuda.is_available())
print("TF32 matmul ukljucen:", torch.backends.cuda.matmul.allow_tf32)

print("FAISS:", faiss.__version__)
print("FAISS broj GPU-a:", faiss.get_num_gpus())

print("cuda runtime:", cp.cuda.runtime.runtimeGetVersion())
