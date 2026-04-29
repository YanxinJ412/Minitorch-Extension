import json
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sacrebleu.metrics import BLEU

import minitorch


def build_translation_prompt(example, src_key, tokenizer) -> List[int]:
    return tokenizer(f"{example[src_key]}<eos_{src_key}>")["input_ids"]


def estimate_kv_cache_bytes(kv_cache) -> int:
    total = 0
    for layer_cache in kv_cache.layers:
        if hasattr(layer_cache, "storage_nbytes"):
            total += layer_cache.storage_nbytes
            continue
        if layer_cache.key is not None:
            total += layer_cache.key._tensor._storage.nbytes
        if layer_cache.value is not None:
            total += layer_cache.value._tensor._storage.nbytes
    return total


def greedy_decode(
    model,
    prompt_ids,
    tokenizer,
    backend,
    max_new_tokens,
    eos_token_id,
    use_cache,
    kv_cache_quantization: str="none",
):
    prompt_ids = list(prompt_ids)
    generated_ids: List[int] = []
    kv_cache = None

    start_time = time.time()
    if use_cache:
        prompt_tensor = minitorch.tensor([prompt_ids], backend=backend)
        logits, kv_cache = model(
            prompt_tensor,
            use_cache=True,
            kv_cache_quantization=kv_cache_quantization,
        )
        next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))
    else:
        context_ids = list(prompt_ids)
        logits = model(minitorch.tensor([context_ids], backend=backend))
        next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))

    for _ in range(max_new_tokens):
        generated_ids.append(next_token)
        if next_token == eos_token_id:
            break

        if use_cache:
            step_tensor = minitorch.tensor([[next_token]], backend=backend)
            logits, kv_cache = model(step_tensor, kv_cache=kv_cache, use_cache=True)
        else:
            context_ids.append(next_token)
            step_tensor = minitorch.tensor([context_ids], backend=backend)
            logits = model(step_tensor)

        next_token = int(np.argmax(logits.to_numpy()[0, -1, :]))

    elapsed = time.time() - start_time
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    return {
        "generated_ids": generated_ids,
        "generated_text": decoded_text,
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(generated_ids) / max(elapsed, 1e-8),
        "kv_cache_bytes": 0 if kv_cache is None else estimate_kv_cache_bytes(kv_cache),
    }


def benchmark_generation(
    model,
    examples: Sequence[Dict[str, str]],
    tokenizer,
    src_key: str,
    tgt_key: str,
    backend,
    max_new_tokens: int = 32,
    num_examples: int = 5,
    kv_cache_quantization: str = "none",
) -> Dict[str, object]:
    bleu = BLEU()
    outputs = {}
    was_training = model.training
    model.eval()

    eval_examples = list(examples[:num_examples])
    eos_token_id = tokenizer.vocab[f"<eos_{tgt_key}>"]

    for use_cache in (False, True):
        predictions = []
        references = []
        tokens_per_sec = []
        cache_sizes = []
        sample_outputs = []

        for example in eval_examples:
            prompt_ids = build_translation_prompt(example, src_key, tokenizer)
            generation = greedy_decode(
                model=model,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                backend=backend,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                use_cache=use_cache,
                kv_cache_quantization=kv_cache_quantization,
            )

            prediction_text = generation["generated_text"].split(f"<eos_{tgt_key}>")[0]
            predictions.append(prediction_text)
            references.append(example[tgt_key])
            tokens_per_sec.append(generation["tokens_per_sec"])
            cache_sizes.append(generation["kv_cache_bytes"])
            sample_outputs.append(
                {
                    "source": example[src_key],
                    "reference": example[tgt_key],
                    "prediction": prediction_text,
                }
            )

        mode = "kv_cache" if use_cache else "full_recompute"
        outputs[mode] = {
            "bleu": bleu.corpus_score(predictions, [references]).score,
            "avg_tokens_per_sec": float(np.mean(tokens_per_sec)) if tokens_per_sec else 0.0,
            "avg_kv_cache_bytes": float(np.mean(cache_sizes)) if cache_sizes else 0.0,
            "kv_cache_quantization": kv_cache_quantization if use_cache else "none",
            "samples": sample_outputs[: min(3, len(sample_outputs))],
        }

    if was_training:
        model.train()

    print(json.dumps(outputs, indent=4, ensure_ascii=False))
    return outputs
