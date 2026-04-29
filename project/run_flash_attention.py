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
    _aggregate_generation_speed_curves,
    _build_generation_speed_curve,
    _mean_token_match_vs_reference,
    build_token_blocks,
    collate_token_blocks,
    get_text_dataset,
    get_tokenizer,
    train,
)


def _append_log(log_path: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _mean_curve(curves: Sequence[Sequence[float]]) -> List[float]:
    if not curves:
        return []
    arr = np.asarray(curves, dtype=np.float64)
    return arr.mean(axis=0).tolist()


def _speed_curve_to_sec_per_token(curve: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if curve is None:
        return None
    chunk_size = int(curve["chunk_size_tokens"])
    avg_elapsed = [float(x) for x in curve.get("avg_elapsed_sec_per_chunk", [])]  # type: ignore[arg-type]
    std_elapsed = [float(x) for x in curve.get("std_elapsed_sec_per_chunk", [])]  # type: ignore[arg-type]
    return {
        "chunk_size_tokens": chunk_size,
        "chunk_end_token_index": list(curve["chunk_end_token_index"]),  # type: ignore[arg-type]
        "n_prompts_averaged": int(curve["n_prompts_averaged"]),
        "avg_sec_per_token_per_chunk": [
            0.0 if dt <= 0.0 else dt / float(chunk_size)
            for dt in avg_elapsed
        ],
        "std_elapsed_sec_per_chunk": std_elapsed,
    }


def greedy_decode_flash_benchmark(
    *,
    model,
    prompt_ids,
    backend,
    num_new_tokens: int,
) -> Dict[str, object]:
    prompt_ids = list(prompt_ids)
    generated_ids: List[int] = []
    context_ids = list(prompt_ids)
    sec_per_token: List[float] = []
    chunk_elapsed: List[float] = []
    chunk_n_tokens: List[int] = []
    chunk_size = 10
    chunk_start: Optional[float] = None
    n_in_chunk = 0

    logits = model(minitorch.tensor([context_ids], backend=backend))

    next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))

    for _ in range(num_new_tokens):
        generated_ids.append(next_token)
        if n_in_chunk == 0:
            chunk_start = time.perf_counter()
        step_start = time.perf_counter()
        context_ids.append(next_token)
        logits = model(minitorch.tensor([context_ids], backend=backend))
        dt = float(time.perf_counter() - step_start)
        sec_per_token.append(dt)
        next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))

        assert chunk_start is not None
        n_in_chunk += 1
        if n_in_chunk == chunk_size:
            chunk_elapsed.append(float(time.perf_counter() - chunk_start))
            chunk_n_tokens.append(n_in_chunk)
            n_in_chunk = 0
            chunk_start = None

    generation_speed_curve = _build_generation_speed_curve(chunk_elapsed, chunk_n_tokens, chunk_size)
    total_elapsed = float(np.sum(np.asarray(sec_per_token, dtype=np.float64))) if sec_per_token else 0.0
    avg_tokens_per_sec = float(len(sec_per_token) / max(total_elapsed, 1e-8)) if sec_per_token else 0.0
    return {
        "generated_ids": generated_ids,
        "sec_per_token": sec_per_token,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "generation_speed_curve": generation_speed_curve,
    }


def benchmark_flash_attention_generation(
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
    outdir: str = "figures_flash",
    log_stem: str = "flash_attention_benchmark",
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
        "use_paged_attention": False,
        "kv_cache_page_size": 64,
    }
    run_specs: List[Tuple[int, str, bool]] = [
        (1, "full_compute_no_cache", False),
        (2, "flash_attention_no_cache", True),
    ]

    def _make_model(use_flash_attention: bool) -> DecoderLM:
        if use_flash_attention and not getattr(backend, "supports_flash_attn", False):
            raise RuntimeError(
                "flash_decode_attn.so does not export full flash attention support; run bash compile_cuda.sh"
            )
        model = DecoderLM(
            **{
                **base_config,
                "use_flash_attention": use_flash_attention,
            }
        )
        load_model_weights(model=model, path=load_weights_path, backend=backend)
        return model

    model_standard: Optional[DecoderLM] = None
    model_flash: Optional[DecoderLM] = None
    eval_subset = [block for block in eval_blocks if len(block) > prompt_length][:num_prompts]
    runs_out: List[Dict[str, object]] = []
    run_generations: Dict[int, List[List[int]]] = {}
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"{log_stem}.log")
    json_path = os.path.join(outdir, f"{log_stem}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")
    _append_log(log_path, f"Starting flash attention benchmark with {len(eval_subset)} prompts")

    for run_id, mode_name, use_flash_attention in tqdm.tqdm(
        run_specs,
        desc="Flash benchmark runs",
        total=len(run_specs),
    ):
        if use_flash_attention:
            if model_flash is None:
                model_flash = _make_model(True)
            model = model_flash
        else:
            if model_standard is None:
                model_standard = _make_model(False)
            model = model_standard

        was_training = model.training
        model.eval()
        _append_log(
            log_path,
            f"Run {run_id}: mode={mode_name}, flash={use_flash_attention}",
        )

        tps_values: List[float] = []
        sec_curves: List[List[float]] = []
        speed_curves: List[Dict[str, object]] = []
        generations: List[List[int]] = []

        for block in tqdm.tqdm(
            eval_subset,
            desc=f"Run {run_id}",
            total=len(eval_subset),
            leave=False,
        ):
            result = greedy_decode_flash_benchmark(
                model=model,
                prompt_ids=block[:prompt_length],
                backend=backend,
                num_new_tokens=num_new_tokens,
            )
            tps_values.append(float(result["avg_tokens_per_sec"]))  # type: ignore[arg-type]
            sec_curves.append([float(x) for x in result["sec_per_token"]])  # type: ignore[index]
            speed_curves.append(result["generation_speed_curve"])  # type: ignore[arg-type]
            generations.append(list(result["generated_ids"]))  # type: ignore[index]

        if was_training:
            model.train()

        avg_sec_curve = _mean_curve(sec_curves)
        avg_sec_per_10_tokens = _speed_curve_to_sec_per_token(
            _aggregate_generation_speed_curves(speed_curves)
        )
        run_generations[int(run_id)] = generations
        run_payload = {
            "run_id": int(run_id),
            "mode": mode_name,
            "use_flash_attention": bool(use_flash_attention),
            "avg_tokens_per_sec": float(np.mean(np.asarray(tps_values, dtype=np.float64))) if tps_values else 0.0,
            "sec_per_token": avg_sec_curve,
            "avg_sec_per_token_every_10_tokens": avg_sec_per_10_tokens,
            "generated_tokens": int(num_new_tokens),
            "n_prompts_averaged": int(len(eval_subset)),
        }
        runs_out.append(run_payload)
        _append_log(
            log_path,
            json.dumps(
                {
                    "run_id": int(run_id),
                    "avg_tokens_per_sec": run_payload["avg_tokens_per_sec"],
                }
            ),
        )

    baseline_run_id = 1
    baseline_generations = run_generations.get(baseline_run_id, [])
    for run in runs_out:
        run["vs_baseline_run1_token_match_rate"] = float(
            _mean_token_match_vs_reference(
                run_generations.get(int(run["run_id"]), []),
                baseline_generations,
            )
        )

    payload = {
        "baseline_run_id": baseline_run_id,
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
    outdir="figures_flash",
    log_stem="flash_attention_benchmark",
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
        "use_flash_attention": False,
        "kv_cache_page_size": 64,
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
    benchmark_flash_attention_generation(
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
        outdir=outdir,
        log_stem=log_stem,
    )


if __name__ == "__main__":
    fire.Fire(main)
