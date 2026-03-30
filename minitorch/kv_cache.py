from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .tensor import Tensor
from .tensor_functions import tensor_from_numpy
from .tensor_ops import TensorBackend


@dataclass
class LayerKVCache:
    backend: TensorBackend
    key: Optional[Tensor] = None
    value: Optional[Tensor] = None

    @property
    def seq_len(self) -> int:
        if self.key is None:
            return 0
        return self.key.shape[2]

    def append(self, key: Tensor, value: Tensor) -> None:
        key_np = key.detach().to_numpy()
        value_np = value.detach().to_numpy()
        if self.key is None:
            merged_key = key_np
            merged_value = value_np
        else:
            merged_key = np.concatenate((self.key.detach().to_numpy(), key_np), axis=2)
            merged_value = np.concatenate((self.value.detach().to_numpy(), value_np), axis=2)
        self.key = tensor_from_numpy(merged_key, backend=self.backend)
        self.value = tensor_from_numpy(merged_value, backend=self.backend)

    def clear(self) -> None:
        self.key = None
        self.value = None


class KVCache:
    def __init__(self, n_layers: int, backend: TensorBackend):
        self.layers: List[LayerKVCache] = [
            LayerKVCache(backend=backend) for _ in range(n_layers)
        ]

    def __getitem__(self, idx: int) -> LayerKVCache:
        return self.layers[idx]

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()

    @property
    def seq_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].seq_len
