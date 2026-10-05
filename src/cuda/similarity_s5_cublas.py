import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas

_ALFA = np.ones(1, dtype=np.float32)
_BETA = np.zeros(1, dtype=np.float32)


def slicnost_s5(korpus, upit, izlaz, cfg):
    N, d = korpus.shape
    rucka = cp.cuda.device.get_cublas_handle()
    cublas.sgemv(rucka, cublas.CUBLAS_OP_T, d, N,
                 _ALFA.ctypes.data, korpus.data.ptr, d,
                 upit.data.ptr, 1,
                 _BETA.ctypes.data, izlaz.data.ptr, 1)
