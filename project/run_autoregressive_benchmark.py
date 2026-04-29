from functools import partial
import inspect
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import datasets
import fire
import numpy as np
import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer

import minitorch
from minitorch import DecoderLM
from minitorch.cuda_kernel_ops import CudaKernelOps
from project.checkpointing import (
    load_model_weights,
    save_model_weights,
    save_model_config,
    validate_model_config,
)


def get_text_dataset(
    dataset_name: str,
    dataset_config: str,
    text_key: str,
    max_train_examples: int = 0,
    max_validation_examples: int = 0,
    max_test_examples: int = 0,
):
    dataset = {
        split: datasets.load_dataset(dataset_name, dataset_config, split=split)
        for split in ("train", "validation", "test")
    }

    def clean_split(split_examples):
        texts = []
        for example in split_examples:
            text = example[text_key].strip()
            if text:
                texts.append(text)
        return texts

    dataset = {split: clean_split(values) for split, values in dataset.items()}

    limits = {
        "train": max_train_examples,
        "validation": max_validation_examples,
        "test": max_test_examples,
    }
    for split, limit in limits.items():
        if limit > 0:
            dataset[split] = dataset[split][:limit]

    print(json.dumps({"data_size": {split: len(values) for split, values in dataset.items()}}, indent=4))
    return dataset


def get_tokenizer(texts, vocab_size, workdir, reuse_if_available=True):
    tokenizer_path = f"{workdir}/tokenizer.json"
    config_path = f"{workdir}/config.json"

    if reuse_if_available and os.path.exists(tokenizer_path) and os.path.exists(config_path):
        return AutoTokenizer.from_pretrained(
            workdir,
            eos_token=None,
            bos_token=None,
            pad_token=None,
            unk_token=None,
        )

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(texts, vocab_size=vocab_size, special_tokens=["<eos>", "<pad>"])
    tokenizer.save(tokenizer_path)
    json.dump({"model_type": "gpt2"}, open(config_path, "w"))

    return AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None,
    )


def build_token_blocks(
    texts: Sequence[str],
    tokenizer,
    block_size: int,
    max_blocks: int = 0,
) -> List[List[int]]:
    eos_token_id = tokenizer.vocab["<eos>"]
    all_ids: List[int] = []
    for text in texts:
        all_ids.extend(tokenizer(text)["input_ids"])
        all_ids.append(eos_token_id)

    chunk_len = block_size + 1
    blocks = []
    for start in range(0, len(all_ids) - chunk_len + 1, chunk_len):
        block = all_ids[start:start + chunk_len]
        if len(block) == chunk_len:
            blocks.append(block)
        if max_blocks > 0 and len(blocks) >= max_blocks:
            break
    return blocks


def collate_token_blocks(blocks, backend):
    tokens = np.array(blocks, dtype=np.int64)
    input_ids = minitorch.tensor_from_numpy(tokens[:, :-1], backend=backend)
    labels = minitorch.tensor_from_numpy(tokens[:, 1:], backend=backend)
    return {"input_ids": input_ids, "labels": labels}


def loss_fn(batch, model):
    idx = batch["input_ids"]
    idx.requires_grad_(True)
    logits = model(idx=idx)
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(batch_size * seq_len, vocab_size)
    targets = batch["labels"].view(batch_size * seq_len)
    return minitorch.nn.softmax_loss(logits=logits, target=targets).mean()


def train(model, optimizer, blocks, n_samples, collate_fn, batch_size, desc):
    model.train()
    random.shuffle(blocks)
    blocks = blocks[:n_samples]

    for i in (prog_bar := tqdm.trange(0, len(blocks), batch_size, desc=f"Training ({desc})")):
        batch = collate_fn(blocks=blocks[i:i + batch_size])
        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        loss.backward()
        optimizer.step()
        batch_time = time.time() - t0
        prog_bar.set_postfix(
            tokens_per_sec=np.prod(batch["input_ids"].shape) / max(batch_time, 1e-8),
            loss=loss.item(),
            lr=optimizer.lr,
        )


def estimate_kv_cache_bytes(kv_cache) -> int:
    if hasattr(kv_cache, "storage_nbytes"):
        return kv_cache.storage_nbytes
    total = 0
    for layer_cache in kv_cache.layers:
        if hasattr(layer_cache, "storage_nbytes"):
            total += layer_cache.storage_nbytes
    return total

def model_forward_with_optional_cache_args(model, idx, use_cache=False, kv_cache=None, kv_cache_quantization="none"):
    kwargs = {}
    if use_cache:
        kwargs["use_cache"] = True
    if kv_cache is not None:
        kwargs["kv_cache"] = kv_cache

    signature = inspect.signature(model.forward)
    if "kv_cache_quantization" in signature.parameters:
        kwargs["kv_cache_quantization"] = kv_cache_quantization
    if "kv_cache_max_bytes" in signature.parameters:
        kwargs["kv_cache_max_bytes"] = model_forward_with_optional_cache_args.kv_cache_max_bytes
    return model(idx, **kwargs)


model_forward_with_optional_cache_args.kv_cache_max_bytes = None


def _build_generation_speed_curve(
    chunk_elapsed: List[float],
    chunk_n_tokens: List[int],
    chunk_size: int,
) -> Dict[str, object]:
    cum = 0
    ends: List[int] = []
    for dt, nt in zip(chunk_elapsed, chunk_n_tokens):
        cum += int(nt)
        ends.append(cum)
    return {
        "chunk_size_tokens": int(chunk_size),
        "chunk_end_token_index": ends,
        "elapsed_sec_per_chunk": [float(x) for x in chunk_elapsed],
    }


def _aggregate_generation_speed_curves(
    curves: Sequence[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if not curves:
        return None
    first = curves[0]
    n = len(first["elapsed_sec_per_chunk"])  # type: ignore[arg-type]
    if n == 0:
        return None
    for c in curves:
        if len(c["elapsed_sec_per_chunk"]) != n:  # type: ignore[arg-type]
            return None
    out: Dict[str, object] = {
        "chunk_size_tokens": first["chunk_size_tokens"],
        "chunk_end_token_index": list(first["chunk_end_token_index"]),
        "n_prompts_averaged": len(curves),
    }
    stacks = np.array(
        [[float(x) for x in c["elapsed_sec_per_chunk"]] for c in curves],
        dtype=np.float64,
    )
    out["avg_elapsed_sec_per_chunk"] = stacks.mean(axis=0).tolist()
    out["std_elapsed_sec_per_chunk"] = stacks.std(axis=0).tolist()
    return out


def greedy_decode_fixed_tokens(
    model,
    prompt_ids,
    backend,
    num_new_tokens,
    use_cache,
    kv_cache_quantization,
    kv_cache_max_bytes,
):
    prompt_ids = list(prompt_ids)
    generated_ids: List[int] = []
    kv_cache = None

    start_time = time.time()
    model_forward_with_optional_cache_args.kv_cache_max_bytes = kv_cache_max_bytes
    if use_cache:
        logits, kv_cache = model_forward_with_optional_cache_args(
            model,
            minitorch.tensor([prompt_ids], backend=backend),
            use_cache=True,
            kv_cache_quantization=kv_cache_quantization,
        )
    else:
        context_ids = list(prompt_ids)
        logits = model(minitorch.tensor([context_ids], backend=backend))

    next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))

    chunk_elapsed: List[float] = []
    chunk_n_tokens: List[int] = []
    cs = 10
    chunk_start: Optional[float] = None
    n_in_chunk = 0

    for step_i in range(num_new_tokens):
        if n_in_chunk == 0:
            chunk_start = time.perf_counter()
        generated_ids.append(next_token)
        if use_cache:
            logits, kv_cache = model_forward_with_optional_cache_args(
                model,
                minitorch.tensor([[next_token]], backend=backend),
                use_cache=True,
                kv_cache=kv_cache,
                kv_cache_quantization=kv_cache_quantization,
            )
        else:
            context_ids.append(next_token)
            logits = model(minitorch.tensor([context_ids], backend=backend))
        next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))

        assert chunk_start is not None
        n_in_chunk += 1
        # Only record FULL chunks so each entry is "time per cs tokens".
        if n_in_chunk == 10:
            dt = time.perf_counter() - chunk_start
            chunk_elapsed.append(dt)
            chunk_n_tokens.append(n_in_chunk)
            n_in_chunk = 0
            chunk_start = None

    elapsed = time.time() - start_time
    speed_curve = _build_generation_speed_curve(chunk_elapsed, chunk_n_tokens, 10)
    return {
        "generated_ids": generated_ids,
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(generated_ids) / max(elapsed, 1e-8),
        "kv_cache_bytes": 0 if kv_cache is None else estimate_kv_cache_bytes(kv_cache),
        "generation_speed_curve": speed_curve,
    }


def _mean_token_match_vs_reference(
    generations: Sequence[Sequence[int]],
    reference: Sequence[Sequence[int]],
) -> float:
    if not generations or not reference or len(generations) != len(reference):
        return 0.0
    rates: List[float] = []
    for gen, ref in zip(generations, reference):
        g = np.asarray(gen, dtype=np.int64)
        r = np.asarray(ref, dtype=np.int64)
        n = int(min(g.size, r.size))
        if n <= 0:
            continue
        rates.append(float(np.mean(g[:n] == r[:n])))
    return float(np.mean(rates)) if rates else 0.0


def _first_divergence_index(a: Sequence[int], b: Sequence[int]) -> int:
    """First index i where a[i] != b[i]. If identical up to min length, returns min length."""
    n = int(min(len(a), len(b)))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    return n


def benchmark_autoregressive_generation(
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
    suite_kv_budget_bytes: int = 128 * 1024,
    debug_compare_run_id: int = 0,
    debug_print_first_n_tokens: int = 80,
    suite_run_ids: str = "",
) -> Dict[str, object]:
    """14 runs: 1–7 non-fused, 8–14 fused (cache / quant / KV budget). Baseline is run 1 (full recompute)."""
    base_config = {
        "n_vocab": n_vocab,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_positions": n_positions,
        "n_layer": n_layer,
        "p_dropout": p_dropout,
        "ln_eps": ln_eps,
        "backend": backend,
    }
    suite_specs: List[Tuple] = [
        (1, False, False, "none", False),
        (2, False, True, "none", False),
        (3, False, True, "int8", False),
        (4, False, True, "int4", False),
        (5, False, True, "none", True),
        (6, False, True, "int8", True),
        (7, False, True, "int4", True),
        (8, True, False, "none", False),
        (9, True, True, "none", False),
        (10, True, True, "int8", False),
        (11, True, True, "int4", False),
        (12, True, True, "none", True),
        (13, True, True, "int8", True),
        (14, True, True, "int4", True),
    ]

    run_id_filter: Optional[set[int]] = None
    suite_run_ids = str(suite_run_ids or "").strip()
    if suite_run_ids:
        # Fire may pass tuples like "(1,3)"; accept any non-digit separators.
        run_id_filter = {int(x) for x in re.findall(r"-?\d+", suite_run_ids)}
        suite_specs = [s for s in suite_specs if int(s[0]) in run_id_filter]

    def _make_model(use_fused: bool) -> DecoderLM:
        m = DecoderLM(**{**base_config, "use_fused_kernel": use_fused})
        load_model_weights(model=m, path=load_weights_path, backend=backend)
        return m

    runs_out: List[Dict[str, object]] = []
    baseline_generations: Optional[List[List[int]]] = None
    baseline_tps = 0.0
    model_nf: Optional[DecoderLM] = None
    model_f: Optional[DecoderLM] = None

    eval_subset = [b for b in eval_blocks if len(b) > prompt_length][:num_prompts]

    for run_id, use_fused, use_cache, quant, budget_limited in suite_specs:
        if use_fused:
            if model_f is None:
                model_f = _make_model(True)
            model = model_f
        else:
            if model_nf is None:
                model_nf = _make_model(False)
            model = model_nf

        if not use_cache:
            q = "none"
            max_bytes: Optional[int] = None
        else:
            q = quant
            max_bytes = suite_kv_budget_bytes if budget_limited else 0

        was_training = model.training
        model.eval()
        desc = f"Run {run_id} (fused={use_fused}, cache={use_cache}, q={q}, budget={budget_limited})"
        print(f"Starting evaluation for {desc} on {len(eval_subset)} prompts")
        all_generations: List[float] = []
        cache_sizes: List[float] = []
        mode_generations: List[List[int]] = []
        speed_curves: List[Dict[str, object]] = []

        for block in tqdm.tqdm(eval_subset, desc=desc):
            prompt_ids = block[:prompt_length]
            model_forward_with_optional_cache_args.kv_cache_max_bytes = max_bytes
            generation = greedy_decode_fixed_tokens(
                model=model,
                prompt_ids=prompt_ids,
                backend=backend,
                num_new_tokens=num_new_tokens,
                use_cache=use_cache,
                kv_cache_quantization=q,
                kv_cache_max_bytes=max_bytes,
            )
            all_generations.append(float(generation["tokens_per_sec"]))
            cache_sizes.append(float(generation["kv_cache_bytes"]))
            mode_generations.append(list(generation["generated_ids"]))
            speed_curves.append(generation["generation_speed_curve"])  # type: ignore[arg-type]

        if was_training:
            model.train()

        tps = float(np.mean(all_generations)) if all_generations else 0.0
        kv_b = float(np.mean(cache_sizes)) if use_cache else 0.0
        gens = mode_generations

        if run_id == 1:
            baseline_generations = gens
            baseline_tps = tps

        assert baseline_generations is not None
        match_vs = _mean_token_match_vs_reference(gens, baseline_generations)
        if debug_compare_run_id and run_id == int(debug_compare_run_id) and gens and baseline_generations:
            n_show = max(int(debug_print_first_n_tokens), 1)
            # Summarize divergence across ALL prompts, and show token prefixes for prompt 0.
            per_prompt_div = [
                _first_divergence_index(list(r), list(g))
                for r, g in zip(baseline_generations, gens)
            ]
            per_prompt_match = [
                float(np.mean(np.asarray(g[: min(len(g), len(r))], dtype=np.int64) == np.asarray(r[: min(len(g), len(r))], dtype=np.int64)))
                if min(len(g), len(r)) > 0
                else 0.0
                for r, g in zip(baseline_generations, gens)
            ]

            a = list(baseline_generations[0])
            b = list(gens[0])
            div = int(per_prompt_div[0]) if per_prompt_div else _first_divergence_index(a, b)
            print(
                json.dumps(
                    {
                        "debug_compare": {
                            "run_id": int(run_id),
                            "prompt_index_shown": 0,
                            "first_divergence_token_index_shown": int(div),
                            "first_divergence_token_index_per_prompt": per_prompt_div,
                            "token_match_rate_vs_run1_per_prompt": per_prompt_match,
                            "baseline_run1_generated_ids_prefix": a[:n_show],
                            "this_run_generated_ids_prefix": b[:n_show],
                        }
                    },
                    indent=4,
                )
            )
        speedup_vs = tps / max(baseline_tps, 1e-8)
        agg_curve = _aggregate_generation_speed_curves(speed_curves)

        runs_out.append(
            {
                "run_id": run_id,
                "use_fused_kernel": use_fused,
                "use_cache": use_cache,
                "kv_cache_quantization": q if use_cache else "none",
                "kv_cache_max_bytes": max_bytes if use_cache else None,
                "kv_budget_limited": bool(budget_limited) if use_cache else False,
                "avg_tokens_per_sec": tps,
                "avg_kv_cache_bytes": kv_b,
                "vs_baseline_run1_token_match_rate": match_vs,
                "vs_baseline_run1_speedup": speedup_vs,
                "generation_speed_curve": agg_curve,
            }
        )

    payload = {
        "baseline_run_id": 1,
        "baseline_avg_tokens_per_sec": baseline_tps,
        "suite_kv_budget_bytes": suite_kv_budget_bytes,
        "runs": runs_out,
    }
    print(json.dumps(payload, indent=4))
    return payload


def main(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    text_key="text",
    model_max_length=256,
    n_epochs=1,
    batch_size=64,
    learning_rate=0.002,
    samples_per_epoch=20000,
    n_vocab=10000,
    n_embd=256,
    seed=11111,
    use_fused_kernel=False,
    load_weights_path=None,
    save_weights_path=None,
    run_generation_eval=True,
    generation_examples=20,
    generation_prompt_length=1,
    generation_max_new_tokens=32,
    max_train_texts=0,
    max_validation_texts=0,
    max_test_texts=0,
    max_train_blocks=0,
    max_eval_blocks=512,
    suite_kv_budget_bytes=128 * 1024,
    debug_compare_run_id=0,
    debug_print_first_n_tokens=80,
    suite_run_ids="",
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
    print("use_fused_kernel: ", use_fused_kernel)

    config = {
        "n_vocab": n_vocab,
        "n_embd": n_embd,
        "n_head": 8,
        "n_positions": model_max_length,
        "n_layer": 4,
        "p_dropout": 0.1,
        "ln_eps": 1e-5,
        "backend": backend,
        "use_fused_kernel": use_fused_kernel,
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

    print(json.dumps({
        "block_count": {
            "train": len(train_blocks),
            "validation": len(eval_blocks),
        }
    }, indent=4))

    collate_fn = partial(collate_token_blocks, backend=backend)

    for epoch_idx in range(n_epochs):
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
    print(f"saved model weights to {save_weights_path}")
    save_model_config(config=config, path=f"{artifact_dir}/model_config.json")

    if run_generation_eval:
        suite_weights = load_weights_path if load_weights_path is not None else save_weights_path
        benchmark_autoregressive_generation(
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
            suite_kv_budget_bytes=int(suite_kv_budget_bytes),
            debug_compare_run_id=int(debug_compare_run_id),
            debug_print_first_n_tokens=int(debug_print_first_n_tokens),
            suite_run_ids=str(suite_run_ids),
        )


if __name__ == "__main__":
    fire.Fire(main)