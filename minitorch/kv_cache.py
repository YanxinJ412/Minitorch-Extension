from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .tensor import Tensor
from .tensor_functions import tensor_from_numpy
from .tensor_ops import TensorBackend


_SUPPORTED_QUANTIZATION = {"none", "int8", "int4"}
_DEFAULT_CACHE_BUDGET_BYTES = 2 * 1024 * 1024


@dataclass
class QuantizedTensorStorage:
    data: np.ndarray
    scale: float
    shape: Tuple[int, ...]
    bits: int

    @property
    def nbytes(self) -> int:
        return self.data.nbytes + np.dtype(np.float32).itemsize


def _validate_quantization(quantization: Optional[str]) -> str:
    mode = "none" if quantization is None else str(quantization).lower()
    if mode not in _SUPPORTED_QUANTIZATION:
        raise ValueError(
            f"Unsupported KV-cache quantization '{quantization}'. "
            f"Expected one of {sorted(_SUPPORTED_QUANTIZATION)}."
        )
    return mode


def _pack_int4(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1).astype(np.int8)
    unsigned = (flat + 8).astype(np.uint8)
    if unsigned.size % 2 == 1:
        unsigned = np.concatenate((unsigned, np.zeros(1, dtype=np.uint8)))
    packed = unsigned[0::2] | (unsigned[1::2] << 4)
    return packed


def _unpack_int4(packed: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    unpacked = np.empty(packed.size * 2, dtype=np.uint8)
    unpacked[0::2] = packed & 0x0F
    unpacked[1::2] = packed >> 4
    total = int(np.prod(shape))
    signed = unpacked[:total].astype(np.int8) - 8
    return signed.reshape(shape)


def _quantize_array(array: np.ndarray, quantization: str) -> QuantizedTensorStorage:
    if quantization == "int8":
        qmax = 127.0
        bits = 8
    elif quantization == "int4":
        qmax = 7.0
        bits = 4
    else:
        raise ValueError(f"Quantization mode '{quantization}' is not quantized.")

    array = array.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    scale = max(max_abs / qmax, 1e-8)
    quantized = np.clip(np.round(array / scale), -qmax, qmax).astype(np.int8)

    if quantization == "int8":
        payload = quantized
    else:
        payload = _pack_int4(quantized)

    return QuantizedTensorStorage(
        data=payload,
        scale=np.float32(scale).item(),
        shape=tuple(array.shape),
        bits=bits,
    )


def _dequantize_array(storage: QuantizedTensorStorage) -> np.ndarray:
    if storage.bits == 8:
        quantized = storage.data.astype(np.int8, copy=False)
    elif storage.bits == 4:
        quantized = _unpack_int4(storage.data, storage.shape)
    else:
        raise ValueError(f"Unsupported quantized storage bit-width: {storage.bits}")
    return quantized.astype(np.float32) * storage.scale


@dataclass
class LayerKVCache:
    backend: TensorBackend
    quantization: str = "none"
    _key_tensor: Optional[Tensor] = None
    _value_tensor: Optional[Tensor] = None
    _key_quantized: Optional[QuantizedTensorStorage] = None
    _value_quantized: Optional[QuantizedTensorStorage] = None

    def __post_init__(self) -> None:
        self.quantization = _validate_quantization(self.quantization)

    def _get_shape(self) -> Optional[Tuple[int, ...]]:
        if self._key_tensor is not None:
            return self._key_tensor.shape
        if self._key_quantized is not None:
            return self._key_quantized.shape
        return None

    def _tensor_to_numpy(self, tensor: Optional[Tensor], quantized: Optional[QuantizedTensorStorage]) -> Optional[np.ndarray]:
        if tensor is not None:
            return tensor.detach().to_numpy()
        if quantized is not None:
            return _dequantize_array(quantized)
        return None

    def _store_tensor(self, key_np: np.ndarray, value_np: np.ndarray) -> None:
        key_np = key_np.astype(np.float32, copy=False)
        value_np = value_np.astype(np.float32, copy=False)
        if self.quantization == "none":
            self._key_tensor = tensor_from_numpy(key_np, backend=self.backend)
            self._value_tensor = tensor_from_numpy(value_np, backend=self.backend)
            self._key_quantized = None
            self._value_quantized = None
        else:
            self._key_quantized = _quantize_array(key_np, self.quantization)
            self._value_quantized = _quantize_array(value_np, self.quantization)
            self._key_tensor = None
            self._value_tensor = None

    @property
    def key(self) -> Optional[Tensor]:
        key_np = self._tensor_to_numpy(self._key_tensor, self._key_quantized)
        if key_np is None:
            return None
        return tensor_from_numpy(key_np, backend=self.backend)

    @property
    def value(self) -> Optional[Tensor]:
        value_np = self._tensor_to_numpy(self._value_tensor, self._value_quantized)
        if value_np is None:
            return None
        return tensor_from_numpy(value_np, backend=self.backend)

    @property
    def seq_len(self) -> int:
        shape = self._get_shape()
        if shape is None:
            return 0
        return shape[2]

    def append(self, key: Tensor, value: Tensor) -> None:
        key_np = key.detach().to_numpy()
        value_np = value.detach().to_numpy()
        cached_key = self._tensor_to_numpy(self._key_tensor, self._key_quantized)
        cached_value = self._tensor_to_numpy(self._value_tensor, self._value_quantized)
        if cached_key is None:
            merged_key = key_np
            merged_value = value_np
        else:
            merged_key = np.concatenate((cached_key, key_np), axis=2)
            merged_value = np.concatenate((cached_value, value_np), axis=2)
        self._store_tensor(merged_key, merged_value)

    def trim_left(self, n_tokens: int) -> None:
        if n_tokens <= 0 or self.seq_len == 0:
            return

        if n_tokens >= self.seq_len:
            self.clear()
            return

        cached_key = self._tensor_to_numpy(self._key_tensor, self._key_quantized)
        cached_value = self._tensor_to_numpy(self._value_tensor, self._value_quantized)
        assert cached_key is not None
        assert cached_value is not None
        self._store_tensor(cached_key[:, :, n_tokens:, :], cached_value[:, :, n_tokens:, :])

    def clear(self) -> None:
        self._key_tensor = None
        self._value_tensor = None
        self._key_quantized = None
        self._value_quantized = None

    @property
    def storage_nbytes(self) -> int:
        total = 0
        if self._key_tensor is not None:
            total += self._key_tensor._tensor._storage.nbytes
        elif self._key_quantized is not None:
            total += self._key_quantized.nbytes

        if self._value_tensor is not None:
            total += self._value_tensor._tensor._storage.nbytes
        elif self._value_quantized is not None:
            total += self._value_quantized.nbytes
        return total


class KVCache:
    def __init__(self, n_layers: int, backend: TensorBackend, quantization: Optional[str]=None, max_cache_bytes: Optional[int]=None):
        self.quantization = _validate_quantization(quantization)
        self.max_cache_bytes = _DEFAULT_CACHE_BUDGET_BYTES if max_cache_bytes is None else max_cache_bytes
        self.layers: List[LayerKVCache] = [
            LayerKVCache(backend=backend, quantization=self.quantization) for _ in range(n_layers)
        ]

    def __getitem__(self, idx: int) -> LayerKVCache:
        return self.layers[idx]

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()

    @property
    def storage_nbytes(self) -> int:
        return sum(layer.storage_nbytes for layer in self.layers)

    @property
    def seq_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].seq_len

    def _bytes_per_token(self) -> int:
        seq_len = self.seq_len
        if seq_len <= 0:
            return 0
        return max(1, int(np.ceil(self.storage_nbytes / seq_len)))

    def enforce_budget(self) -> None:
        if self.max_cache_bytes is None or self.max_cache_bytes <= 0:
            return

        while self.seq_len > 0 and self.storage_nbytes > self.max_cache_bytes:
            bytes_per_token = self._bytes_per_token()
            if bytes_per_token <= 0:
                break
            overflow = self.storage_nbytes - self.max_cache_bytes
            tokens_to_trim = max(1, int(np.ceil(overflow / bytes_per_token)))
            for layer in self.layers:
                layer.trim_left(tokens_to_trim)
