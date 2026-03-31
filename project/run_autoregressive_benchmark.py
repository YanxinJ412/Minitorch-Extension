from functools import partial
import inspect
import json
import os
import random
import time
from typing import Dict, List, Sequence

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
    return model(idx, **kwargs)


def greedy_decode_fixed_tokens(model, prompt_ids, backend, num_new_tokens, use_cache, kv_cache_quantization):
    prompt_ids = list(prompt_ids)
    generated_ids: List[int] = []
    kv_cache = None

    start_time = time.time()
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

    for _ in range(num_new_tokens):
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

    elapsed = time.time() - start_time
    return {
        "generated_ids": generated_ids,
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(generated_ids) / max(elapsed, 1e-8),
        "kv_cache_bytes": 0 if kv_cache is None else estimate_kv_cache_bytes(kv_cache),
    }


def benchmark_autoregressive_generation(
    model,
    blocks: Sequence[List[int]],
    backend,
    prompt_length: int = 1,
    num_new_tokens: int = 64,
    num_prompts: int = 20,
    kv_cache_quantization: str = "none",
) -> Dict[str, object]:
    outputs = {}
    was_training = model.training
    model.eval()

    eval_blocks = [block for block in blocks if len(block) > prompt_length][:num_prompts]

    for use_cache in (False, True):
        mode = "kv_cache" if use_cache else "full_recompute"
        print(f"Starting evaluation for mode={mode} on {len(eval_blocks)} prompts")
        all_generations = []
        cache_sizes = []
        match_rates = []

        for block in tqdm.tqdm(eval_blocks, desc=f"Evaluating ({mode})"):
            prompt_ids = block[:prompt_length]
            generation = greedy_decode_fixed_tokens(
                model=model,
                prompt_ids=prompt_ids,
                backend=backend,
                num_new_tokens=num_new_tokens,
                use_cache=use_cache,
                kv_cache_quantization=kv_cache_quantization,
            )
            all_generations.append(generation["tokens_per_sec"])
            cache_sizes.append(generation["kv_cache_bytes"])

            if use_cache:
                baseline = greedy_decode_fixed_tokens(
                    model=model,
                    prompt_ids=prompt_ids,
                    backend=backend,
                    num_new_tokens=num_new_tokens,
                    use_cache=False,
                    kv_cache_quantization=kv_cache_quantization,
                )
                generated = np.array(generation["generated_ids"])
                baseline_generated = np.array(baseline["generated_ids"])
                if baseline_generated.size:
                    match_rates.append(float(np.mean(generated == baseline_generated)))

        outputs[mode] = {
            "prompt_length": prompt_length,
            "generated_tokens": num_new_tokens,
            "avg_tokens_per_sec": float(np.mean(all_generations)) if all_generations else 0.0,
            "avg_kv_cache_bytes": float(np.mean(cache_sizes)) if cache_sizes else 0.0,
            "kv_cache_quantization": kv_cache_quantization if use_cache else "none",
        }
        if use_cache and match_rates:
            outputs[mode]["avg_token_match_rate_vs_full"] = float(np.mean(match_rates))

    if outputs.get("full_recompute") and outputs.get("kv_cache"):
        base_tps = outputs["full_recompute"]["avg_tokens_per_sec"]
        cache_tps = outputs["kv_cache"]["avg_tokens_per_sec"]
        outputs["speedup"] = cache_tps / max(base_tps, 1e-8)

    if was_training:
        model.train()

    print(json.dumps(outputs, indent=4))
    return outputs


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
    kv_cache_quantization="none",
    max_train_texts=0,
    max_validation_texts=0,
    max_test_texts=0,
    max_train_blocks=0,
    max_eval_blocks=512,
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
    print("kv_cache_quantization: ", kv_cache_quantization)

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
        benchmark_autoregressive_generation(
            model=model,
            blocks=eval_blocks,
            backend=backend,
            prompt_length=generation_prompt_length,
            num_new_tokens=generation_max_new_tokens,
            num_prompts=generation_examples,
            kv_cache_quantization=kv_cache_quantization,
        )


if __name__ == "__main__":
    fire.Fire(main)
