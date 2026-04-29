from typing import Callable, Optional

import math

from . import operators
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import tensor_from_numpy

import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch

# Load the shared library
lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
lib_softmax = ctypes.CDLL("minitorch/cuda_kernels/softmax_kernel.so")
lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
lib_fused = ctypes.CDLL("minitorch/cuda_kernels/fused_decode_attn.so")
try:
    lib_flash = ctypes.CDLL("minitorch/cuda_kernels/flash_decode_attn.so")
except OSError:
    lib_flash = None
try:
    lib_paged = ctypes.CDLL("minitorch/cuda_kernels/paged_decode_attn.so")
except OSError:
    lib_paged = None
datatype = np.float32

lib_fused.fused_decode_attn_host_fp32.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib_fused.fused_decode_attn_host_fp32.restype = None

lib_fused.fused_decode_attn_host_int8.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
]
lib_fused.fused_decode_attn_host_int8.restype = None

try:
    _lib_fused_int4 = lib_fused.fused_decode_attn_host_int4
except AttributeError:
    _lib_fused_int4 = None
else:
    _lib_fused_int4.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    _lib_fused_int4.restype = None

if lib_paged is not None:
    lib_paged.paged_decode_attn_host_fp32.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib_paged.paged_decode_attn_host_fp32.restype = None

    lib_paged.paged_decode_attn_host_int8.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib_paged.paged_decode_attn_host_int8.restype = None

if lib_flash is not None:
    lib_flash.flash_decode_attn_host_fp32.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib_flash.flash_decode_attn_host_fp32.restype = None
    try:
        _lib_flash_full = lib_flash.flash_attn_host_fp32
    except AttributeError:
        _lib_flash_full = None
    else:
        _lib_flash_full.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        _lib_flash_full.restype = None
else:
    _lib_flash_full = None

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17, 
  operators.tanh: 18
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    supports_fused_decode_attn = True
    supports_flash_decode_attn = lib_flash is not None
    supports_flash_attn = _lib_flash_full is not None
    supports_paged_decode_attn = lib_paged is not None

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorZip.restype = None

            # assert out.size == a.size, f"zip {out.size}, {a.size}"
            # assert out.size == b.size, f"zip {out.size}, {b.size}"

            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_double,                                                         # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorReduce.restype = None

            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                start,
                len(a.shape),
                fn_id
            )

            return out

        return ret

    @staticmethod
    def matrix_multiply_cublas(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]

        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2],
                                    a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2],
                                    b.shape[-1])
        assert a.shape[0] == b.shape[0]

        bs, m, n, k = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        A, B = a.to_numpy(), b.to_numpy()

        # Convert A and B to column-major order
        A_fortran = np.transpose(A, (0, 2, 1))
        B_fortran = np.transpose(B, (0, 2, 1))

        # Flatten A and B for sending to GPU
        A_flat = A_fortran.reshape(bs, -1)
        B_flat = B_fortran.reshape(bs, -1)

        # Allocate memory on GPU
        A_gpu = cuda.mem_alloc(A_flat.nbytes)
        B_gpu = cuda.mem_alloc(B_flat.nbytes)
        C_gpu = cuda.mem_alloc(bs * m * n * A.itemsize)

        # Copy data to GPU
        cuda.memcpy_htod(A_gpu, A_flat)
        cuda.memcpy_htod(B_gpu, B_flat)

        # Prepare arrays of pointers
        A_gpu_ptrs = np.array(
            [int(A_gpu) + i * m * k * A.itemsize for i in range(bs)],
            dtype=np.uint64)
        B_gpu_ptrs = np.array(
            [int(B_gpu) + i * k * n * B.itemsize for i in range(bs)],
            dtype=np.uint64)
        C_gpu_ptrs = np.array(
            [int(C_gpu) + i * m * n * A.itemsize for i in range(bs)],
            dtype=np.uint64)

        # Allocate device memory for arrays of pointers
        A_array_gpu = cuda.mem_alloc(A_gpu_ptrs.nbytes)
        B_array_gpu = cuda.mem_alloc(B_gpu_ptrs.nbytes)
        C_array_gpu = cuda.mem_alloc(C_gpu_ptrs.nbytes)

        # Copy arrays of pointers to device memory
        cuda.memcpy_htod(A_array_gpu, A_gpu_ptrs)
        cuda.memcpy_htod(B_array_gpu, B_gpu_ptrs)
        cuda.memcpy_htod(C_array_gpu, C_gpu_ptrs)

        # Set argument types for the kernel function
        lib_mm.batchedMatMulKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int]

        # Launch kernel
        lib_mm.batchedMatMulKernel(
            int(A_array_gpu), int(B_array_gpu), int(C_array_gpu), m, k, n, bs)

        # Synchronize device to ensure computation is complete
        cuda.Context.synchronize()

        # Copy back the result
        C = np.empty((bs, n, m), dtype=A.dtype)
        cuda.memcpy_dtoh(C, C_gpu)
        C = np.transpose(C, (0, 2, 1))

        c = tensor_from_numpy(
            np.ascontiguousarray(C),
            backend=a.backend, requires_grad=a.requires_grad()).contiguous()

        # Undo 3d if we added it.
        if both_2d:
            c = c.view(c.shape[1], c.shape[2])
        if len(ls) > 3:
            c = c.view(*ls)
        return c

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            # print(f"Debug in matmul: output shape {ls}")
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
            # print(f"Debug in matmul: batched dim [:-2] and get the strides {nshape, nstrides}")
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            a.shape[1],
            b.shape[2]
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
            # print(f"Debug in matmul: output shape {out.shape}")
        return out

    @staticmethod
    def fused_decode_attn_fw(
        q: Tensor,
        mask: Tensor,
        kv_mode: str,
        k_np: np.ndarray,
        v_np: np.ndarray,
        k_scale: float,
        v_scale: float,
        q_dim: int,
        seq_len_override: Optional[int] = None,
    ) -> Tensor:
        """Single-query attention with fused CUDA kernel; ``k_np``/``v_np`` are numpy buffers.

        For ``kv_mode='int4'``, K/V are packed uint8 buffers (1D) and ``seq_len_override`` is required.
        """
        batch_size, num_head, queries_len, d_head = q.shape
        assert queries_len == 1
        if kv_mode == "int4":
            if seq_len_override is None:
                raise ValueError("seq_len_override is required for kv_mode='int4'")
            seq_len = int(seq_len_override)
        else:
            assert k_np.shape == v_np.shape
            seq_len = int(k_np.shape[2])
            assert k_np.shape == (batch_size, num_head, seq_len, d_head)
        inv_sqrt_d = 1.0 / math.sqrt(float(q_dim))
        q_flat = np.ascontiguousarray(q.detach().to_numpy(), dtype=np.float32).reshape(-1)
        mask_flat = np.ascontiguousarray(mask.detach().to_numpy(), dtype=np.float32).reshape(-1)
        out = q.zeros((batch_size, num_head, 1, d_head))
        out_flat = out._tensor._storage
        if kv_mode == "fp32":
            k_flat = np.ascontiguousarray(k_np, dtype=np.float32).reshape(-1)
            v_flat = np.ascontiguousarray(v_np, dtype=np.float32).reshape(-1)
            lib_fused.fused_decode_attn_host_fp32(
                q_flat,
                k_flat,
                v_flat,
                mask_flat,
                out_flat,
                ctypes.c_int(batch_size),
                ctypes.c_int(num_head),
                ctypes.c_int(seq_len),
                ctypes.c_int(d_head),
                ctypes.c_float(inv_sqrt_d),
            )
        elif kv_mode == "int8":
            k_flat = np.ascontiguousarray(k_np, dtype=np.int8).reshape(-1)
            v_flat = np.ascontiguousarray(v_np, dtype=np.int8).reshape(-1)
            lib_fused.fused_decode_attn_host_int8(
                q_flat,
                k_flat,
                v_flat,
                mask_flat,
                out_flat,
                ctypes.c_int(batch_size),
                ctypes.c_int(num_head),
                ctypes.c_int(seq_len),
                ctypes.c_int(d_head),
                ctypes.c_float(inv_sqrt_d),
                ctypes.c_float(float(k_scale)),
                ctypes.c_float(float(v_scale)),
            )
        elif kv_mode == "int4":
            if _lib_fused_int4 is None:
                raise RuntimeError(
                    "fused_decode_attn.so does not export INT4 support. Rebuild with bash compile_cuda.sh"
                )
            k_flat = np.ascontiguousarray(k_np, dtype=np.uint8).reshape(-1)
            v_flat = np.ascontiguousarray(v_np, dtype=np.uint8).reshape(-1)
            _lib_fused_int4(
                q_flat,
                k_flat,
                v_flat,
                mask_flat,
                out_flat,
                ctypes.c_int(batch_size),
                ctypes.c_int(num_head),
                ctypes.c_int(seq_len),
                ctypes.c_int(d_head),
                ctypes.c_float(inv_sqrt_d),
                ctypes.c_float(float(k_scale)),
                ctypes.c_float(float(v_scale)),
            )
        else:
            raise ValueError(f"Unsupported fused KV mode '{kv_mode}'")
        return out

    @staticmethod
    def paged_decode_attn_fw(
        q: Tensor,
        mask: Tensor,
        kv_mode: str,
        k_pages_np: np.ndarray,
        v_pages_np: np.ndarray,
        page_offsets_np: np.ndarray,
        k_scales_np: np.ndarray,
        v_scales_np: np.ndarray,
        q_dim: int,
        page_size: int,
    ) -> Tensor:
        if lib_paged is None:
            raise RuntimeError("paged_decode_attn.so is not available; run bash compile_cuda.sh")

        batch_size, num_head, queries_len, d_head = q.shape
        assert queries_len == 1
        total_seq = int(page_offsets_np[-1]) if page_offsets_np.size > 0 else 0
        num_pages = max(int(page_offsets_np.shape[0]) - 1, 0)
        inv_sqrt_d = 1.0 / math.sqrt(float(q_dim))
        q_flat = np.ascontiguousarray(q.detach().to_numpy(), dtype=np.float32).reshape(-1)
        mask_flat = np.ascontiguousarray(mask.detach().to_numpy(), dtype=np.float32).reshape(-1)
        page_offsets = np.ascontiguousarray(page_offsets_np, dtype=np.int32).reshape(-1)
        out = q.zeros((batch_size, num_head, 1, d_head))
        out_flat = out._tensor._storage
        if kv_mode == "fp32":
            k_flat = np.ascontiguousarray(k_pages_np, dtype=np.float32).reshape(-1)
            v_flat = np.ascontiguousarray(v_pages_np, dtype=np.float32).reshape(-1)
            lib_paged.paged_decode_attn_host_fp32(
                q_flat,
                k_flat,
                v_flat,
                page_offsets,
                mask_flat,
                out_flat,
                ctypes.c_int(batch_size),
                ctypes.c_int(num_head),
                ctypes.c_int(num_pages),
                ctypes.c_int(page_size),
                ctypes.c_int(total_seq),
                ctypes.c_int(d_head),
                ctypes.c_float(inv_sqrt_d),
            )
        elif kv_mode == "int8":
            k_flat = np.ascontiguousarray(k_pages_np, dtype=np.int8).reshape(-1)
            v_flat = np.ascontiguousarray(v_pages_np, dtype=np.int8).reshape(-1)
            k_scales = np.ascontiguousarray(k_scales_np, dtype=np.float32).reshape(-1)
            v_scales = np.ascontiguousarray(v_scales_np, dtype=np.float32).reshape(-1)
            lib_paged.paged_decode_attn_host_int8(
                q_flat,
                k_flat,
                v_flat,
                page_offsets,
                k_scales,
                v_scales,
                mask_flat,
                out_flat,
                ctypes.c_int(batch_size),
                ctypes.c_int(num_head),
                ctypes.c_int(num_pages),
                ctypes.c_int(page_size),
                ctypes.c_int(total_seq),
                ctypes.c_int(d_head),
                ctypes.c_float(inv_sqrt_d),
            )
        else:
            raise ValueError(f"Unsupported paged KV mode '{kv_mode}'")
        return out

    @staticmethod
    def flash_decode_attn_fw(
        q: Tensor,
        mask: Tensor,
        k_np: np.ndarray,
        v_np: np.ndarray,
        q_dim: int,
    ) -> Tensor:
        if lib_flash is None:
            raise RuntimeError("flash_decode_attn.so is not available; run bash compile_cuda.sh")

        batch_size, num_head, queries_len, d_head = q.shape
        assert queries_len == 1
        assert k_np.shape == v_np.shape
        seq_len = int(k_np.shape[2])
        inv_sqrt_d = 1.0 / math.sqrt(float(q_dim))
        q_flat = np.ascontiguousarray(q.detach().to_numpy(), dtype=np.float32).reshape(-1)
        mask_flat = np.ascontiguousarray(mask.detach().to_numpy(), dtype=np.float32).reshape(-1)
        k_flat = np.ascontiguousarray(k_np, dtype=np.float32).reshape(-1)
        v_flat = np.ascontiguousarray(v_np, dtype=np.float32).reshape(-1)
        out = q.zeros((batch_size, num_head, 1, d_head))
        out_flat = out._tensor._storage
        lib_flash.flash_decode_attn_host_fp32(
            q_flat,
            k_flat,
            v_flat,
            mask_flat,
            out_flat,
            ctypes.c_int(batch_size),
            ctypes.c_int(num_head),
            ctypes.c_int(seq_len),
            ctypes.c_int(d_head),
            ctypes.c_float(inv_sqrt_d),
        )
        return out

    @staticmethod
    def flash_attn_fw(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,
        q_dim: int,
    ) -> Tensor:
        if _lib_flash_full is None:
            raise RuntimeError("flash_decode_attn.so does not export full flash attention; run bash compile_cuda.sh")

        batch_size, num_head, queries_len, d_head = q.shape
        assert k.shape == v.shape
        key_len = int(k.shape[2])
        inv_sqrt_d = 1.0 / math.sqrt(float(q_dim))
        q_flat = np.ascontiguousarray(q.detach().to_numpy(), dtype=np.float32).reshape(-1)
        k_flat = np.ascontiguousarray(k.detach().to_numpy(), dtype=np.float32).reshape(-1)
        v_flat = np.ascontiguousarray(v.detach().to_numpy(), dtype=np.float32).reshape(-1)
        mask_flat = np.ascontiguousarray(mask.detach().to_numpy(), dtype=np.float32).reshape(-1)
        out = q.zeros((batch_size, num_head, queries_len, d_head))
        out_flat = out._tensor._storage
        _lib_flash_full(
            q_flat,
            k_flat,
            v_flat,
            mask_flat,
            out_flat,
            ctypes.c_int(batch_size),
            ctypes.c_int(num_head),
            ctypes.c_int(queries_len),
            ctypes.c_int(key_len),
            ctypes.c_int(d_head),
            ctypes.c_float(inv_sqrt_d),
        )
        return out

    @staticmethod
    def attn_softmax_fw(inp: Tensor, mask: Tensor):
      batch_size, nhead, from_len, to_len = inp.shape
      is_dec_self_attn = False
      stream = torch.cuda.current_stream().cuda_stream

      lib_softmax.launch_attn_softmax.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_void_p
      ]
      lib_softmax.launch_attn_softmax.restype = None

      lib_softmax.launch_attn_softmax(
        inp._tensor._storage,
        mask._tensor._storage,
        batch_size,
        nhead,
        from_len,
        to_len,
        is_dec_self_attn,
        stream
      ) 

      return inp

    @staticmethod
    def attn_softmax_bw(out_grad: Tensor, soft_inp: Tensor):
      #   BEGIN ASSIGN4_1_2
      rows = np.prod(out_grad.shape[:-1])
      softmax_len = out_grad.shape[-1]
      stream = torch.cuda.current_stream().cuda_stream
      lib_softmax.launch_attn_softmax_bw.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
      ]
      lib_softmax.launch_attn_softmax_bw.restype = None

      lib_softmax.launch_attn_softmax_bw(
        out_grad._tensor._storage,
        soft_inp._tensor._storage,
        rows,
        softmax_len,
        stream
      ) 

      return out_grad
      #   END ASSIGN4_1_2

    @staticmethod
    def layernorm_fw(inp: Tensor, gamma: Tensor, beta: Tensor):
      #   BEGIN ASSIGN4_2_1
      rows, hidden_dim = inp.shape
      stream = torch.cuda.current_stream().cuda_stream
      
      ln_res = inp.zeros(inp.shape)
      var = inp.zeros((rows,))
      mean = inp.zeros((rows,))

      lib_layernorm.launch_layernorm.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
      ]
      lib_layernorm.launch_layernorm.restype = None

      lib_layernorm.launch_layernorm(
        ln_res._tensor._storage,
        var._tensor._storage,
        mean._tensor._storage,
        inp._tensor._storage,
        gamma._tensor._storage,
        beta._tensor._storage,
        rows,
        hidden_dim,
        stream
      ) 

      return ln_res, var, mean
    #   raise("Not implemented")
      #   END ASSIGN4_2_1
      
    @staticmethod
    def layernorm_bw(out_grad: Tensor, inp: Tensor, gamma: Tensor, beta: Tensor, var: Tensor, mean: Tensor):
      #   BEGIN ASSIGN4_2_2
      rows, hidden_dim = inp.shape
      stream_1 = torch.cuda.current_stream().cuda_stream
      stream_2 = torch.cuda.current_stream().cuda_stream
      gamma_grad = gamma.zeros(gamma.shape)
      beta_grad = beta.zeros(beta.shape)
      inp_grad = inp.zeros(inp.shape)
      
      lib_layernorm.launch_layernorm_bw.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p
      ]
      lib_layernorm.launch_layernorm_bw.restype = None

      lib_layernorm.launch_layernorm_bw(
        gamma_grad._tensor._storage,
        beta_grad._tensor._storage,
        inp_grad._tensor._storage,
        out_grad._tensor._storage,
        inp._tensor._storage,
        gamma._tensor._storage,
        beta._tensor._storage,
        var._tensor._storage,
        mean._tensor._storage,
        rows,
        hidden_dim,
        stream_1,
        stream_2
      )

      return (inp_grad, gamma_grad, beta_grad)
    #   raise("Not implemented")
      #   END ASSIGN4_2_2
      
