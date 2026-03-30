import json
from typing import Dict

import numpy as np

import minitorch


def save_model_weights(model, path: str) -> None:
    state: Dict[str, np.ndarray] = {}
    for name, parameter in model.named_parameters():
        if parameter.value is None:
            continue
        state[name] = parameter.value.detach().to_numpy()
    np.savez(path, **state)


def load_model_weights(model, path: str, backend) -> None:
    with np.load(path) as state:
        loaded = {name: state[name] for name in state.files}

    missing = []
    for name, parameter in model.named_parameters():
        if name not in loaded:
            missing.append(name)
            continue
        parameter.update(
            minitorch.tensor_from_numpy(loaded[name], backend=backend, requires_grad=True)
        )

    if missing:
        raise ValueError(f"Missing parameters in checkpoint: {missing}")


def save_model_config(config: Dict[str, object], path: str) -> None:
    serializable = {
        key: value
        for key, value in config.items()
        if key != "backend"
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=4)


def validate_model_config(config: Dict[str, object], path: str) -> None:
    with open(path) as f:
        saved = json.load(f)

    expected = {
        key: value
        for key, value in config.items()
        if key != "backend"
    }

    mismatches = {}
    for key, expected_value in expected.items():
        saved_value = saved.get(key)
        if saved_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "found": saved_value,
            }

    if mismatches:
        raise ValueError(f"Checkpoint config mismatch: {mismatches}")
