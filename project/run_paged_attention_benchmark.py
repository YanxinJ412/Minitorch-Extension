from functools import partial
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import fire
import numpy as np
import tqdm

import minitorch
from minitorch import DecoderLM
from minitorch.cuda_kernel_ops import CudaKernelOps
from project.checkpointing import (
    load_model_weights,
    save_model_config,
    save_model_weights,
    validate_model_config,
)
from project.run_autoregressive_benchmark import (
    _mean_token_match_vs_reference,
    build_token_blocks,
    collate_token_blocks,
    get_text_dataset,
    get_tokenizer,
    train,
)


def estimate_memory_usage(kv_cache) -> int:
    if kv_cache is None:
        return 0
    if hasattr(kv_cache, "storage_nbytes"):
        return int(kv_cache.storage_nbytes)
    total = 0
    for layer_cache in getattr(kv_cache, "layers", []):
        total += int(getattr(layer_cache, "storage_nbytes", 0))
    return int(total)


def _mean_curve(curves: Sequence[Sequence[float]]) -> List[float]:
    if not curves:
        return []
    arr = np.asarray(curves, dtype=np.float64)
    return arr.mean(axis=0).tolist()


def _append_log(log_path: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def greedy_decode_with_memory(
    *,
    model,
    prompt_ids,
    backend,
    num_new_tokens: int,
    kv_cache_quantization: str,
) -> Dict[str, object]:
    prompt_ids = list(prompt_ids)
    generated_ids: List[int] = []
    kv_cache = None
    context_ids = list(prompt_ids)
    time_per_token_sec: List[float] = []
    dynamic_memory_usage_bytes: List[int] = []

    if prompt_ids:
        logits, kv_cache = model(
            minitorch.tensor([prompt_ids], backend=backend),
            use_cache=True,
            kv_cache_quantization=kv_cache_quantization,
        )
        dynamic_memory_usage_bytes.append(estimate_memory_usage(kv_cache))
    else:
        logits = model(minitorch.tensor([context_ids], backend=backend))
        dynamic_memory_usage_bytes.append(0)

    next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))

    for _ in range(num_new_tokens):
        generated_ids.append(next_token)
        step_start = time.perf_counter()
        logits, kv_cache = model(
            minitorch.tensor([[next_token]], backend=backend),
            kv_cache=kv_cache,
            use_cache=True,
            kv_cache_quantization=kv_cache_quantization,
        )
        time_per_token_sec.append(float(time.perf_counter() - step_start))
        dynamic_memory_usage_bytes.append(estimate_memory_usage(kv_cache))
        next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))

    peak_memory_usage_bytes = max(dynamic_memory_usage_bytes) if dynamic_memory_usage_bytes else 0
    avg_time_per_token_sec = (
        float(np.mean(np.asarray(time_per_token_sec, dtype=np.float64)))
        if time_per_token_sec
        else 0.0
    )
    return {
        "generated_ids": generated_ids,
        "time_per_token_sec": time_per_token_sec,
        "avg_time_per_token_sec": avg_time_per_token_sec,
        "dynamic_memory_usage_bytes": dynamic_memory_usage_bytes,
        "peak_memory_usage_bytes": int(peak_memory_usage_bytes),
    }


def benchmark_paged_attention_generation(
    *,
    load_weights_path: str,
    eval_blocks: Sequence[List[int]],
    backend,
    n_vocab: int,
    n_embd: int,
    n_head: int,
    n_positions: int,
    n_layer: int,
    p_dropout: float,
    ln_eps: float,
    prompt_length: int,
    num_new_tokens: int,
    num_prompts: int,
    kv_cache_page_size: int = 64,
    outdir: str = "figures_paged",
    log_stem: str = "paged_attention_benchmark",
) -> Dict[str, object]:
    base_config = {
        "n_vocab": n_vocab,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_positions": n_positions,
        "n_layer": n_layer,
        "p_dropout": p_dropout,
        "ln_eps": ln_eps,
        "backend": backend,
        "use_fused_kernel": False,
    }
    mode_specs: List[Tuple[int, str, bool, str]] = [
        (1, "paged_attention", True, "none"),
        (2, "paged_attention", True, "int8"),
        (3, "old_attention", False, "none"),
        (4, "old_attention", False, "int8"),
    ]

    def _make_model(use_paged_attention: bool) -> DecoderLM:
        model = DecoderLM(
            **{
                **base_config,
                "use_paged_attention": use_paged_attention,
                "kv_cache_page_size": kv_cache_page_size,
            }
        )
        load_model_weights(model=model, path=load_weights_path, backend=backend)
        return model

    model_standard: Optional[DecoderLM] = None
    model_paged: Optional[DecoderLM] = None
    eval_subset = [block for block in eval_blocks if len(block) > prompt_length][:num_prompts]
    runs_out: List[Dict[str, object]] = []
    run_generations: Dict[int, List[List[int]]] = {}
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"{log_stem}.log")
    json_path = os.path.join(outdir, f"{log_stem}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")
    _append_log(log_path, f"Starting paged attention benchmark with {len(eval_subset)} prompts")

    for run_id, mode_name, use_paged_attention, quantization in tqdm.tqdm(
        mode_specs,
        desc="Paged benchmark runs",
        total=len(mode_specs),
    ):
        if use_paged_attention:
            if model_paged is None:
                model_paged = _make_model(True)
            model = model_paged
        else:
            if model_standard is None:
                model_standard = _make_model(False)
            model = model_standard

        was_training = model.training
        model.eval()
        _append_log(
            log_path,
            f"Run {run_id}: mode={mode_name}, paged={use_paged_attention}, quant={quantization}",
        )

        time_curves: List[List[float]] = []
        memory_curves: List[List[float]] = []
        peak_memory_values: List[int] = []
        generations: List[List[int]] = []

        for block in tqdm.tqdm(
            eval_subset,
            desc=f"Run {run_id}",
            total=len(eval_subset),
            leave=False,
        ):
            result = greedy_decode_with_memory(
                model=model,
                prompt_ids=block[:prompt_length],
                backend=backend,
                num_new_tokens=num_new_tokens,
                kv_cache_quantization=quantization,
            )
            time_curves.append([float(x) for x in result["time_per_token_sec"]])  # type: ignore[index]
            memory_curves.append([float(x) for x in result["dynamic_memory_usage_bytes"]])  # type: ignore[index]
            peak_memory_values.append(int(result["peak_memory_usage_bytes"]))  # type: ignore[index]
            generations.append(list(result["generated_ids"]))  # type: ignore[index]

        if was_training:
            model.train()

        avg_time_curve = _mean_curve(time_curves)
        avg_memory_curve = _mean_curve(memory_curves)
        run_generations[int(run_id)] = generations
        run_payload = {
            "run_id": int(run_id),
            "mode": mode_name,
            "use_paged_attention": bool(use_paged_attention),
            "kv_cache_quantization": quantization,
            "page_size": int(kv_cache_page_size) if use_paged_attention else None,
            "avg_time_per_token_sec": float(np.mean(np.asarray(avg_time_curve, dtype=np.float64))) if avg_time_curve else 0.0,
            "time_per_token_sec": avg_time_curve,
            "dynamic_memory_usage_bytes": avg_memory_curve,
            "peak_memory_usage_bytes": int(max(peak_memory_values) if peak_memory_values else 0),
            "generated_tokens": int(num_new_tokens),
            "n_prompts_averaged": int(len(eval_subset)),
        }
        runs_out.append(run_payload)
        _append_log(
            log_path,
            json.dumps(
                {
                    "run_id": int(run_id),
                    "avg_time_per_token_sec": run_payload["avg_time_per_token_sec"],
                    "peak_memory_usage_bytes": run_payload["peak_memory_usage_bytes"],
                }
            ),
        )

    baseline_run_id = 3
    baseline_generations = run_generations.get(baseline_run_id, [])
    for run in runs_out:
        run["vs_baseline_run3_token_match_rate"] = float(
            _mean_token_match_vs_reference(
                run_generations.get(int(run["run_id"]), []),
                baseline_generations,
            )
        )

    payload = {
        "baseline_run_id": baseline_run_id,
        "kv_cache_page_size": int(kv_cache_page_size),
        "prompt_length": int(prompt_length),
        "generated_tokens": int(num_new_tokens),
        "runs": runs_out,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    _append_log(log_path, f"Wrote JSON results to {json_path}")
    print(json.dumps(payload, indent=4))
    return payload


def main(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    text_key="text",
    model_max_length=256,
    n_epochs=0,
    batch_size=64,
    learning_rate=0.002,
    samples_per_epoch=20000,
    n_vocab=10000,
    n_embd=256,
    seed=11111,
    load_weights_path=None,
    save_weights_path=None,
    generation_examples=20,
    generation_prompt_length=1,
    generation_max_new_tokens=32,
    max_train_texts=0,
    max_validation_texts=0,
    max_test_texts=0,
    max_train_blocks=0,
    max_eval_blocks=512,
    kv_cache_page_size=64,
    outdir="figures_paged",
    log_stem="paged_attention_benchmark",
):
    np.random.seed(seed)
    random.seed(seed)

    workdir = f"./workdir_lm_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}"
    if save_weights_path is not None:
        artifact_dir = os.path.dirname(save_weights_path) or "."
    elif load_weights_path is not None:
        artifact_dir = os.path.dirname(load_weights_path) or "."
    else:
        artifact_dir = workdir
    os.makedirs(artifact_dir, exist_ok=True)

    backend = minitorch.TensorBackend(CudaKernelOps)
    config = {
        "n_vocab": n_vocab,
        "n_embd": n_embd,
        "n_head": 8,
        "n_positions": model_max_length,
        "n_layer": 4,
        "p_dropout": 0.1,
        "ln_eps": 1e-5,
        "backend": backend,
        "use_fused_kernel": False,
        "use_paged_attention": False,
        "kv_cache_page_size": kv_cache_page_size,
    }

    model_config_path = f"{artifact_dir}/model_config.json"
    if load_weights_path is not None and os.path.exists(model_config_path):
        validate_model_config(config=config, path=model_config_path)

    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)

    if load_weights_path is not None:
        load_model_weights(model=model, path=load_weights_path, backend=backend)
        print(f"loaded model weights from {load_weights_path}")

    dataset = get_text_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        text_key=text_key,
        max_train_examples=max_train_texts,
        max_validation_examples=max_validation_texts,
        max_test_examples=max_test_texts,
    )
    tokenizer = get_tokenizer(texts=dataset["train"], vocab_size=config["n_vocab"], workdir=artifact_dir)
    train_blocks = build_token_blocks(
        texts=dataset["train"],
        tokenizer=tokenizer,
        block_size=model_max_length,
        max_blocks=max_train_blocks,
    )
    eval_blocks = build_token_blocks(
        texts=dataset["validation"],
        tokenizer=tokenizer,
        block_size=model_max_length,
        max_blocks=max_eval_blocks,
    )

    collate_fn = partial(collate_token_blocks, backend=backend)
    for epoch_idx in range(int(n_epochs)):
        train(
            model=model,
            optimizer=optimizer,
            blocks=train_blocks,
            n_samples=min(samples_per_epoch, len(train_blocks)) if samples_per_epoch > 0 else len(train_blocks),
            collate_fn=collate_fn,
            batch_size=batch_size,
            desc=f"epoch {epoch_idx} / {n_epochs}",
        )

    if save_weights_path is None:
        save_weights_path = f"{artifact_dir}/model_weights.npz"
    save_dir = os.path.dirname(save_weights_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_model_weights(model=model, path=save_weights_path)
    save_model_config(config=config, path=f"{artifact_dir}/model_config.json")

    suite_weights = load_weights_path if load_weights_path is not None else save_weights_path
    benchmark_paged_attention_generation(
        load_weights_path=suite_weights,
        eval_blocks=eval_blocks,
        backend=backend,
        n_vocab=n_vocab,
        n_embd=n_embd,
        n_head=config["n_head"],
        n_positions=model_max_length,
        n_layer=config["n_layer"],
        p_dropout=config["p_dropout"],
        ln_eps=config["ln_eps"],
        prompt_length=generation_prompt_length,
        num_new_tokens=generation_max_new_tokens,
        num_prompts=generation_examples,
        kv_cache_page_size=kv_cache_page_size,
        outdir=outdir,
        log_stem=log_stem,
    )


if __name__ == "__main__":
    fire.Fire(main)
