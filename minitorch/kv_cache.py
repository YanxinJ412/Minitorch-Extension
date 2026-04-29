from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .tensor import Tensor
from .tensor_functions import tensor_from_numpy
from .tensor_ops import TensorBackend


_SUPPORTED_QUANTIZATION = {"none", "int8", "int4"}
_DEFAULT_CACHE_BUDGET_BYTES = -1
FUSED_DECODE_MAX_SEQ = 1024
PAGED_DECODE_MAX_SEQ = 4096


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
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
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
    _key_array: Optional[np.ndarray] = None
    _value_array: Optional[np.ndarray] = None
    _key_quantized: Optional[QuantizedTensorStorage] = None
    _value_quantized: Optional[QuantizedTensorStorage] = None
    _positions: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.quantization = _validate_quantization(self.quantization)

    def _get_shape(self) -> Optional[Tuple[int, ...]]:
        if self._key_array is not None:
            return self._key_array.shape
        if self._key_quantized is not None:
            return self._key_quantized.shape
        return None

    def _cached_to_numpy(
        self,
        array: Optional[np.ndarray],
        quantized: Optional[QuantizedTensorStorage],
    ) -> Optional[np.ndarray]:
        if array is not None:
            return array
        if quantized is not None:
            return _dequantize_array(quantized)
        return None

    def _store_array(self, key_np: np.ndarray, value_np: np.ndarray) -> None:
        key_np = key_np.astype(np.float32, copy=False)
        value_np = value_np.astype(np.float32, copy=False)
        if self.quantization == "none":
            self._key_array = key_np
            self._value_array = value_np
            self._key_quantized = None
            self._value_quantized = None
        else:
            self._key_quantized = _quantize_array(key_np, self.quantization)
            self._value_quantized = _quantize_array(value_np, self.quantization)
            self._key_array = None
            self._value_array = None

    @property
    def positions(self) -> Optional[np.ndarray]:
        return self._positions

    @property
    def key(self) -> Optional[Tensor]:
        key_np = self._cached_to_numpy(self._key_array, self._key_quantized)
        if key_np is None:
            return None
        return tensor_from_numpy(key_np, backend=self.backend)

    @property
    def value(self) -> Optional[Tensor]:
        value_np = self._cached_to_numpy(self._value_array, self._value_quantized)
        if value_np is None:
            return None
        return tensor_from_numpy(value_np, backend=self.backend)

    @property
    def seq_len(self) -> int:
        shape = self._get_shape()
        if shape is None:
            return 0
        return shape[2]

    def append(self, key: Tensor, value: Tensor, positions: np.ndarray) -> None:
        key_np = key.detach().to_numpy()
        value_np = value.detach().to_numpy()
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1:
            raise ValueError("positions must be a 1D array")
        if positions.shape[0] != key_np.shape[2]:
            raise ValueError("positions length must match appended sequence length")
        cached_key = self._cached_to_numpy(self._key_array, self._key_quantized)
        cached_value = self._cached_to_numpy(self._value_array, self._value_quantized)
        if cached_key is None:
            merged_key = key_np
            merged_value = value_np
            merged_positions = positions
        else:
            merged_key = np.concatenate((cached_key, key_np), axis=2)
            merged_value = np.concatenate((cached_value, value_np), axis=2)
            assert self._positions is not None
            merged_positions = np.concatenate((self._positions, positions), axis=0)
        self._store_array(merged_key, merged_value)
        self._positions = merged_positions

    def fused_decode_buffers(
        self, max_seq: int = FUSED_DECODE_MAX_SEQ
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray, float, float, int]]:
        shape = self._get_shape()
        if shape is None:
            return None
        _b, _h, seq_len, _d = shape
        if seq_len < 1 or seq_len > max_seq:
            return None
        if self.quantization == "none":
            if self._key_array is None or self._value_array is None:
                return None
            k = np.ascontiguousarray(self._key_array, dtype=np.float32)
            v = np.ascontiguousarray(self._value_array, dtype=np.float32)
            return ("fp32", k, v, 1.0, 1.0, int(seq_len))
        if self.quantization == "int8":
            if self._key_quantized is None or self._value_quantized is None:
                return None
            k = np.ascontiguousarray(self._key_quantized.data)
            v = np.ascontiguousarray(self._value_quantized.data)
            if k.dtype != np.int8 or v.dtype != np.int8:
                return None
            return (
                "int8",
                k,
                v,
                float(self._key_quantized.scale),
                float(self._value_quantized.scale),
                int(seq_len),
            )
        if self.quantization == "int4":
            if self._key_quantized is None or self._value_quantized is None:
                return None
            k = np.ascontiguousarray(self._key_quantized.data)
            v = np.ascontiguousarray(self._value_quantized.data)
            if k.dtype != np.uint8 or v.dtype != np.uint8:
                return None
            return (
                "int4",
                k,
                v,
                float(self._key_quantized.scale),
                float(self._value_quantized.scale),
                int(seq_len),
            )
        return None

    def trim_left(self, n_tokens: int) -> None:
        if n_tokens <= 0 or self.seq_len == 0:
            return

        if n_tokens >= self.seq_len:
            self.clear()
            return

        cached_key = self._cached_to_numpy(self._key_array, self._key_quantized)
        cached_value = self._cached_to_numpy(self._value_array, self._value_quantized)
        assert cached_key is not None
        assert cached_value is not None
        self._store_array(cached_key[:, :, n_tokens:, :], cached_value[:, :, n_tokens:, :])
        assert self._positions is not None
        self._positions = self._positions[n_tokens:]

    def clear(self) -> None:
        self._key_array = None
        self._value_array = None
        self._key_quantized = None
        self._value_quantized = None
        self._positions = None

    @property
    def storage_nbytes(self) -> int:
        total = 0
        if self._key_array is not None:
            total += self._key_array.nbytes
        elif self._key_quantized is not None:
            total += self._key_quantized.nbytes

        if self._value_array is not None:
            total += self._value_array.nbytes
        elif self._value_quantized is not None:
            total += self._value_quantized.nbytes
        if self._positions is not None:
            total += self._positions.nbytes
        return total


@dataclass
class PagedLayerKVCache:
    backend: TensorBackend
    quantization: str = "none"
    page_size: int = 64
    _key_pages: List[np.ndarray] = field(default_factory=list)
    _value_pages: List[np.ndarray] = field(default_factory=list)
    _key_quantized_pages: List[QuantizedTensorStorage] = field(default_factory=list)
    _value_quantized_pages: List[QuantizedTensorStorage] = field(default_factory=list)
    _position_pages: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.quantization = _validate_quantization(self.quantization)
        self.page_size = max(int(self.page_size), 1)

    def _page_count(self) -> int:
        return len(self._position_pages)

    def _page_length(self, idx: int) -> int:
        return int(self._position_pages[idx].shape[0])

    def _page_key_numpy(self, idx: int) -> np.ndarray:
        if self.quantization == "none":
            return self._key_pages[idx]
        return _dequantize_array(self._key_quantized_pages[idx])

    def _page_value_numpy(self, idx: int) -> np.ndarray:
        if self.quantization == "none":
            return self._value_pages[idx]
        return _dequantize_array(self._value_quantized_pages[idx])

    def _store_page(self, key_np: np.ndarray, value_np: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[QuantizedTensorStorage], Optional[QuantizedTensorStorage]]:
        key_np = np.ascontiguousarray(key_np, dtype=np.float32)
        value_np = np.ascontiguousarray(value_np, dtype=np.float32)
        if self.quantization == "none":
            return key_np, value_np, None, None
        return None, None, _quantize_array(key_np, self.quantization), _quantize_array(value_np, self.quantization)

    def _append_new_page(self, key_np: np.ndarray, value_np: np.ndarray, positions: np.ndarray) -> None:
        key_arr, value_arr, key_q, value_q = self._store_page(key_np, value_np)
        if key_arr is not None:
            self._key_pages.append(key_arr)
            assert value_arr is not None
            self._value_pages.append(value_arr)
        else:
            assert key_q is not None
            assert value_q is not None
            self._key_quantized_pages.append(key_q)
            self._value_quantized_pages.append(value_q)
        self._position_pages.append(np.ascontiguousarray(positions, dtype=np.int64))

    def _replace_page(self, idx: int, key_np: np.ndarray, value_np: np.ndarray, positions: np.ndarray) -> None:
        key_arr, value_arr, key_q, value_q = self._store_page(key_np, value_np)
        if self.quantization == "none":
            assert key_arr is not None
            assert value_arr is not None
            self._key_pages[idx] = key_arr
            self._value_pages[idx] = value_arr
        else:
            assert key_q is not None
            assert value_q is not None
            self._key_quantized_pages[idx] = key_q
            self._value_quantized_pages[idx] = value_q
        self._position_pages[idx] = np.ascontiguousarray(positions, dtype=np.int64)

    def _drop_page(self, idx: int) -> None:
        if self.quantization == "none":
            del self._key_pages[idx]
            del self._value_pages[idx]
        else:
            del self._key_quantized_pages[idx]
            del self._value_quantized_pages[idx]
        del self._position_pages[idx]

    @property
    def positions(self) -> Optional[np.ndarray]:
        if not self._position_pages:
            return None
        return np.concatenate(self._position_pages, axis=0)

    @property
    def key(self) -> Optional[Tensor]:
        if not self._position_pages:
            return None
        page_arrays = [self._page_key_numpy(i) for i in range(self._page_count())]
        return tensor_from_numpy(np.concatenate(page_arrays, axis=2), backend=self.backend)

    @property
    def value(self) -> Optional[Tensor]:
        if not self._position_pages:
            return None
        page_arrays = [self._page_value_numpy(i) for i in range(self._page_count())]
        return tensor_from_numpy(np.concatenate(page_arrays, axis=2), backend=self.backend)

    @property
    def seq_len(self) -> int:
        return int(sum(page.shape[0] for page in self._position_pages))

    def append(self, key: Tensor, value: Tensor, positions: np.ndarray) -> None:
        key_np = np.ascontiguousarray(key.detach().to_numpy(), dtype=np.float32)
        value_np = np.ascontiguousarray(value.detach().to_numpy(), dtype=np.float32)
        positions = np.ascontiguousarray(np.asarray(positions, dtype=np.int64))
        if positions.ndim != 1:
            raise ValueError("positions must be a 1D array")
        if positions.shape[0] != key_np.shape[2]:
            raise ValueError("positions length must match appended sequence length")

        start = 0
        total = int(positions.shape[0])
        while start < total:
            if self._position_pages and self._page_length(-1) < self.page_size:
                room = self.page_size - self._page_length(-1)
                take = min(room, total - start)
                last_key = self._page_key_numpy(-1)
                last_value = self._page_value_numpy(-1)
                merged_key = np.concatenate((last_key, key_np[:, :, start:start + take, :]), axis=2)
                merged_value = np.concatenate((last_value, value_np[:, :, start:start + take, :]), axis=2)
                merged_positions = np.concatenate((self._position_pages[-1], positions[start:start + take]), axis=0)
                self._replace_page(-1, merged_key, merged_value, merged_positions)
                start += take
                continue

            take = min(self.page_size, total - start)
            self._append_new_page(
                key_np[:, :, start:start + take, :],
                value_np[:, :, start:start + take, :],
                positions[start:start + take],
            )
            start += take

    def paged_decode_buffers(
        self,
        max_seq: int = PAGED_DECODE_MAX_SEQ,
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]:
        if self.seq_len < 1 or self.seq_len > max_seq or not self._position_pages:
            return None

        num_pages = self._page_count()
        first_shape = self._page_key_numpy(0).shape
        batch_size, num_head, _page_tokens, d_head = first_shape
        page_offsets = np.zeros(num_pages + 1, dtype=np.int32)
        for i in range(num_pages):
            page_offsets[i + 1] = page_offsets[i] + self._page_length(i)

        if self.quantization == "none":
            key_pages = np.zeros((num_pages, batch_size, num_head, self.page_size, d_head), dtype=np.float32)
            value_pages = np.zeros_like(key_pages)
            for i in range(num_pages):
                page_len = self._page_length(i)
                key_pages[i, :, :, :page_len, :] = self._key_pages[i]
                value_pages[i, :, :, :page_len, :] = self._value_pages[i]
            ones = np.ones(num_pages, dtype=np.float32)
            return ("fp32", key_pages, value_pages, page_offsets, ones, ones, self.seq_len, self.page_size)

        if self.quantization == "int8":
            key_pages = np.zeros((num_pages, batch_size, num_head, self.page_size, d_head), dtype=np.int8)
            value_pages = np.zeros_like(key_pages)
            key_scales = np.ones(num_pages, dtype=np.float32)
            value_scales = np.ones(num_pages, dtype=np.float32)
            for i in range(num_pages):
                page_len = self._page_length(i)
                page_key = self._key_quantized_pages[i]
                page_value = self._value_quantized_pages[i]
                key_pages[i, :, :, :page_len, :] = page_key.data.reshape(batch_size, num_head, page_len, d_head)
                value_pages[i, :, :, :page_len, :] = page_value.data.reshape(batch_size, num_head, page_len, d_head)
                key_scales[i] = float(page_key.scale)
                value_scales[i] = float(page_value.scale)
            return ("int8", key_pages, value_pages, page_offsets, key_scales, value_scales, self.seq_len, self.page_size)
        return None

    def fused_decode_buffers(
        self, max_seq: int = FUSED_DECODE_MAX_SEQ
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray, float, float, int]]:
        if self.seq_len < 1 or self.seq_len > max_seq:
            return None
        key = self.key
        value = self.value
        if key is None or value is None:
            return None
        k = np.ascontiguousarray(key.detach().to_numpy(), dtype=np.float32)
        v = np.ascontiguousarray(value.detach().to_numpy(), dtype=np.float32)
        return ("fp32", k, v, 1.0, 1.0, self.seq_len)

    def trim_left(self, n_tokens: int) -> None:
        remaining = int(n_tokens)
        while remaining > 0 and self._position_pages:
            first_len = self._page_length(0)
            if remaining >= first_len:
                self._drop_page(0)
                remaining -= first_len
                continue
            keep_key = self._page_key_numpy(0)[:, :, remaining:, :]
            keep_value = self._page_value_numpy(0)[:, :, remaining:, :]
            keep_positions = self._position_pages[0][remaining:]
            self._replace_page(0, keep_key, keep_value, keep_positions)
            remaining = 0

    def clear(self) -> None:
        self._key_pages.clear()
        self._value_pages.clear()
        self._key_quantized_pages.clear()
        self._value_quantized_pages.clear()
        self._position_pages.clear()

    @property
    def storage_nbytes(self) -> int:
        total = sum(page.nbytes for page in self._position_pages)
        if self.quantization == "none":
            total += sum(page.nbytes for page in self._key_pages)
            total += sum(page.nbytes for page in self._value_pages)
        else:
            total += sum(page.nbytes for page in self._key_quantized_pages)
            total += sum(page.nbytes for page in self._value_quantized_pages)
        return total


class KVCache:
    def __init__(
        self,
        n_layers: int,
        backend: TensorBackend,
        quantization: Optional[Union[str, Sequence[str]]] = None,
        max_cache_bytes: Optional[int]=None,
    ):
        if isinstance(quantization, (list, tuple)):
            if len(quantization) != int(n_layers):
                raise ValueError(
                    f"KVCache quantization list must have length n_layers={n_layers}, got {len(quantization)}"
                )
            self.quantization = [_validate_quantization(q) for q in quantization]
        else:
            self.quantization = _validate_quantization(quantization)
        self.max_cache_bytes = _DEFAULT_CACHE_BUDGET_BYTES if max_cache_bytes is None else max_cache_bytes
        self.tokens_seen = 0
        if isinstance(self.quantization, list):
            self.layers = [
                LayerKVCache(backend=backend, quantization=self.quantization[i])
                for i in range(int(n_layers))
            ]
        else:
            self.layers = [
                LayerKVCache(backend=backend, quantization=self.quantization) for _ in range(int(n_layers))
            ]

    def __getitem__(self, idx: int) -> LayerKVCache:
        return self.layers[idx]

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()
        self.tokens_seen = 0

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

    def record_tokens(self, n_tokens: int) -> None:
        self.tokens_seen += int(n_tokens)

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


class PagedKVCache:
    def __init__(
        self,
        n_layers: int,
        backend: TensorBackend,
        quantization: Optional[Union[str, Sequence[str]]] = None,
        max_cache_bytes: Optional[int] = None,
        page_size: int = 64,
    ):
        if isinstance(quantization, (list, tuple)):
            if len(quantization) != int(n_layers):
                raise ValueError(
                    f"KVCache quantization list must have length n_layers={n_layers}, got {len(quantization)}"
                )
            self.quantization = [_validate_quantization(q) for q in quantization]
        else:
            self.quantization = _validate_quantization(quantization)
        self.max_cache_bytes = _DEFAULT_CACHE_BUDGET_BYTES if max_cache_bytes is None else max_cache_bytes
        self.tokens_seen = 0
        self.page_size = max(int(page_size), 1)
        if isinstance(self.quantization, list):
            self.layers = [
                PagedLayerKVCache(backend=backend, quantization=self.quantization[i], page_size=self.page_size)
                for i in range(int(n_layers))
            ]
        else:
            self.layers = [
                PagedLayerKVCache(backend=backend, quantization=self.quantization, page_size=self.page_size)
                for _ in range(int(n_layers))
            ]

    def __getitem__(self, idx: int) -> PagedLayerKVCache:
        return self.layers[idx]

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()
        self.tokens_seen = 0

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

    def record_tokens(self, n_tokens: int) -> None:
        self.tokens_seen += int(n_tokens)

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
