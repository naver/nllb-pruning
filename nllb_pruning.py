#!/usr/bin/env python3
# The "NLLB pruning" Python library, Copyright (C) 2023 Naver Corporation, is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license.

import json
import os
import sys
import shutil
import torch
import tempfile
from typing import Optional, Any
from torch import Tensor


default_expert_index = os.path.join(os.path.dirname(__file__), 'experts.json')
cache = {}


def read_json_file(path: str) -> dict:
    """
    Load and cache a JSON file (the files containing the expert ids and gate statistics can be quite large).
    """
    path = os.path.realpath(path)
    if path not in cache:
        cache[path] = json.load(open(path))
    return cache[path]


def load_tokenizer(cache_dir: Optional[str] = None):
    """
    Load the HuggingFace Tokenizer for NLLB-200.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('facebook/nllb-moe-54b', cache_dir=cache_dir)


def load_and_prune_for_lang_pair(
    source_lang: str,
    target_lang: str,
    expert_index: str = default_expert_index,
    cache_dir: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
):
    """
    Download, load and prune the NLLB-200 MoE model.

    Args:
        - source_lang, target_lang: prune the model for this specific language pair
        - expert_index: path to the JSON file containing the ids of the experts to keep per language
        - cache_dir: HuggingFace cache directory where the checkpoints will be downloaded ("~/.cache/huggingface/hub" 
            by default)
        - low_cpu_mem_usage: use less CPU memory by only loading the experts for this specific language pair (if False,
            the default, all experts are loaded but stored in CPU memory)
    """
    expert_ids = get_expert_ids(source_lang=source_lang, target_lang=target_lang, expert_index=expert_index)

    return load_model(
        expert_ids=expert_ids,
        cache_dir=cache_dir,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


def load_model(
    expert_ids: Optional[list[int]] = None,
    cache_dir: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
):
    """
    Download and load the NLLB-200 MoE model. The default cache directory is "~/.cache/huggingface/hub", but it can be
    modified with the `cache_dir` argument. Note that NLLB-200 takes ~200G of disk.
    """
    from transformers import AutoModelForSeq2SeqLM

    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id='facebook/nllb-moe-54b', cache_dir=cache_dir)

    ckpt_format = os.path.join(local_dir, 'pytorch_model-{part:05}-of-00023.bin')
    if os.path.getsize(ckpt_format.format(part=1)) > 7*2**30:  # >5GiB = float32
        # convert all checkpoints to float16 to save disk space and load time
        for part in range(1, 24):
            ckpt_path = ckpt_format.format(part=part)
            print(f'converting {ckpt_path} to float16', file=sys.stderr)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt = {k: v.half() for k, v in ckpt.items()}
            torch.save(ckpt, ckpt_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        if low_cpu_mem_usage:
            assert expert_ids
            # Prune the model to this specific language pair from the start to save CPU memory. This is done by creating
            # a new "pytorch_model.bin.index.json" which references the names of the parameters to keep. When loading 
            # the model, the parameters that are not in there will be initialized on the "meta" device, and will not 
            # take any memory. We still need to call `prune` to move the non-pruned parameters from "meta" to "cuda".
            prefixes = [get_prefix(id) for id in expert_ids]

            for filename in os.listdir(local_dir):
                if filename.endswith('.json'):
                    shutil.copy(
                        os.path.join(local_dir, filename),
                        os.path.join(tmp_dir, filename),
                    )

            param_index = json.load(open(os.path.join(local_dir, 'pytorch_model.bin.index.json')))
            
            param_index['weight_map'] = {
                param_name: os.path.join('..', ckpt_path)
                for param_name, ckpt_path in param_index['weight_map'].items()
                if any(param_name.startswith(prefix) for prefix in prefixes)
                or '.ffn.experts.expert_' not in param_name  # non-expert params
            }

            json.dump(param_index, open(os.path.join(tmp_dir, 'pytorch_model.bin.index.json'), 'w'))
            local_dir = tmp_dir

        model = AutoModelForSeq2SeqLM.from_pretrained(
            local_dir or 'facebook/nllb-moe-54b',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,  # much faster to load and takes less CPU memory with no downsides
        )

        if expert_ids:
            prune(model, expert_ids)

    return model


def move_to_device(module: torch.nn.Module, device: str) -> None:
    """
    Move given module to given device while making sure that unused experts stay on the CPU (this enables unpruning /
    pruning to another language pair).
    """
    if device is None:
        return
    for name, param in module.named_parameters():
        if 'old_experts' in name.split('.'):
            if param.data.device.type != 'meta':
                param.data = param.data.to('cpu')
    
    for name, param in module.named_parameters():
        if 'old_experts' not in name.split('.'):
            if param.data.device.type != 'meta':
                param.data = param.data.to(device)
            assert device == 'cpu' or param.data.device.type != 'meta', \
                'this model was partially initialized, reload it with the right language pair'
    
    for name, buffer in module.named_buffers():
        if buffer.data.device.type != 'meta':
            buffer.data = buffer.data.to(device)


def translate(
    model,
    tokenizer,
    lines: list[str],
    source_lang: str,
    target_lang: str,
    num_beams: int = 4,
    max_length: int = 100,
    batch_size: int = 10,
) -> list[str]:
    """
    Use the model to translate given inputs in given language pair.
    """
    if not lines:
        return []
    
    tokenizer.src_lang = source_lang
    hypotheses = []

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
        ).to('cuda')
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            num_beams=num_beams,
            max_length=max_length,
        )
        hypotheses += tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return hypotheses


def decode_flores(
    model,
    tokenizer,
    source_lang: str,
    target_lang: str,
    split: str = 'devtest',  # dev or devtest
    cache_dir: Optional[str] = None,
    num_beams: int = 4,
    max_length: int = 100,
    batch_size: int = 10,
) -> tuple[list[str], Any]:
    """
    Evaluate the model for given language pair on FLORES.
    
    Returns: list of translation outputs and spBLEU score
    """
    import sacrebleu
    from datasets import load_dataset
    
    def load_lines(lang):
        return [
            entry['sentence'] for entry in
            load_dataset('facebook/flores', lang, cache_dir=cache_dir)[split]
        ]
    
    sources = load_lines(source_lang)
    references = load_lines(target_lang)
    
    hypotheses = translate(
        model,
        tokenizer,
        sources,
        source_lang=source_lang,
        target_lang=target_lang,
        num_beams=num_beams,
        max_length=max_length,
        batch_size=batch_size,
    )
    score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='flores200')
    return hypotheses, score


def prune_for_lang_pair(
    model,
    source_lang: str,
    target_lang: str,
    expert_index: str = default_expert_index,
) -> None:
    """
    Prune the model for given language pair, with the default pruning strategy or using given JSON expert index.
    """
    expert_ids = get_expert_ids(source_lang=source_lang, target_lang=target_lang, expert_index=expert_index)
    prune(model, expert_ids)


def prune(model, expert_ids: list[int]) -> None:
    """
    Prune a HuggingFace model that is already loaded into CPU memory. This is done by keeping the pruned experts 
    in CPU memory (so that we can later prune the model for another language pair) but moving the relevant experts
    and dense parameters to the GPU.
    
    Args:
        model: instance of NllbMoeForConditionalGeneration, obtained by calling `load_model`
        expert_ids: numerical ids of the experts that should be kept (between 0 and 1535)
    """
    model.half()    # in case the model is in float32
    unprune(model)  # unprune the model and move it back to the CPU

    for layer_id, layer in enumerate(model.model.encoder.layers + model.model.decoder.layers):
        if not layer.is_sparse:
            continue

        # expert ids are numbered from 0 to 1535 (128 experts per layer in 12 layers), get the ids that correspond
        # to the current layer and convert them to values between 0 and 127
        ids = [id % 128 for id in expert_ids if (id // 128 * 4 + 3 == layer_id)]

        keys = [f'expert_{id}' for id in ids]

        layer.ffn.old_experts = layer.ffn.experts  # backup old weights
        layer.ffn.experts = torch.nn.ModuleDict({key: layer.ffn.experts[key] for key in keys})

        router = layer.ffn.router
        dtype = router.classifier.weight.dtype
        embed_dim = router.classifier.in_features

        router.old_classifier = router.classifier  # backup old router
        router.classifier = torch.nn.Linear(
            in_features=embed_dim,
            out_features=len(ids),
            bias=False,
            dtype=dtype,
        )
        router.classifier.weight.data[:] = router.old_classifier.weight.data[ids]
        router.num_experts = len(ids)

    move_to_device(model, 'cuda')


def unprune(model) -> None:
    """
    Unprune a model by moving all its experts back to the CPU: automatically called by `prune` and `prune_for_lang_pair`.
    """
    for layer in model.model.encoder.layers + model.model.decoder.layers:
        if not layer.is_sparse:
            continue
        
        if hasattr(layer.ffn, 'old_experts'):
            layer.ffn.experts = layer.ffn.old_experts
            del layer.ffn.old_experts
        router = layer.ffn.router
        if hasattr(router, 'old_classifier'):
            router.classifier = router.old_classifier
            del router.old_classifier
            router.num_experts = 128

    move_to_device(model, 'cpu')


def get_expert_ids(
    source_lang: str,
    target_lang: str,
    expert_index: str = default_expert_index,
) -> list[int]:
    """
    Given a language pair and the path to a JSON expert index, get the list of expert ids to keep.
    """
    expert_index = read_json_file(expert_index)
    if isinstance(expert_index, list):
        return expert_index  # global pruning (i.e., all lang pairs have the same expert ids)
    pair = f'{source_lang}-{target_lang}'
    assert pair in expert_index or source_lang in expert_index and target_lang in expert_index
    if pair in expert_index:  # per-lang-pair pruning
        return expert_index[pair]
    else:  # per-lang pruning
        source_expert_ids = [id for id in expert_index[source_lang] if id < 768]
        target_expert_ids = [id for id in expert_index[target_lang] if id >= 768]
        return source_expert_ids + target_expert_ids


def get_prefix(expert_id: int) -> str:
    """
    Convert an expert id (integer between 0 and 1535) to a parameter name prefix.
    """
    layer_id = expert_id // 128 * 4 + 3
    expert_id = expert_id % 128
    if layer_id < 24:
        module = 'encoder'
    else:
        module = 'decoder'
        layer_id -= 24

    return f'{module}.layers.{layer_id}.ffn.experts.expert_{expert_id}.'


def get_metrics(
    source_lang: str,
    target_lang: str,
    gate_stats_path: str,
    metric: str = 'importance',
) -> Tensor:
    """
    Read metrics from a JSON file containing gate statistics per language or per language pair.
    These importance metrics can be used to apply variants of our pruning algorithm, thanks to
    `select_experts_globally` or `select_experts_per_layer`.
    """
    # Note: to get the real top 2 activity do top1 + top2
    assert metric in ['top1', 'top2', 'conf1', 'conf2', 'mean', 'rank', 'importance']

    if metric == 'importance':
        top1 = get_metrics(source_lang, target_lang, gate_stats_path, 'top1')
        conf1 = get_metrics(source_lang, target_lang, gate_stats_path, 'conf1')
        return top1 * conf1.exp()

    gate_stats = read_json_file(gate_stats_path)  # {lang_pair: {metric: [values]}}
    
    pair = f'{source_lang}-{target_lang}'
    if pair in gate_stats:
        metrics = gate_stats[pair][metric]
        return torch.tensor(metrics).reshape(12, 128)
    else:
        # If this specific lang pair does not have gate statistics, try aggregating statistics of other lang pairs that
        # involve this source or target language.
        # This also works with 'stats-202.json' which only has per-language aggregated statistics named X-TGT and SRC-X
        pairs = [pair_.split('-') for pair_ in gate_stats]
        source_langs = [src for src, tgt in pairs if tgt == target_lang]  # can be ['X']
        target_langs = [tgt for src, tgt in pairs if src == source_lang]  # can be ['X']
        assert source_langs, f'{source_lang} does not have gate statistics'
        assert target_langs, f'{target_lang} does not have gate statistics'

        src_metrics = [get_metrics(source_lang, tgt, gate_stats_path, metric) for tgt in target_langs]
        tgt_metrics = [get_metrics(src, target_lang, gate_stats_path, metric) for src in source_langs]
        src_metrics = sum(src_metrics) / len(src_metrics)
        tgt_metrics = sum(tgt_metrics) / len(tgt_metrics)
        return torch.cat([src_metrics[:6], tgt_metrics[6:]])


def threshold(metrics: Tensor, count: int, min_per_layer: int = 4) -> list[int]:
    """
    Implementation of the "global threshold" algorithm (called by `select_experts_globally`).
    """
    assert min_per_layer * 12 <= count

    metrics /= metrics.sum(dim=-1, keepdim=True)   # normalize layers to sum to 1
    
    values, indices = metrics.sort(dim=-1, descending=True)

    selected = set()
    for layer_id, indices_ in enumerate(indices):
        for i in range(len(indices_)):
            layer_count = sum((1 for i in selected if i // 128 == layer_id), 0)
            if layer_count >= min_per_layer:
                break
            selected.add(layer_id * 128 + indices_[i].item())
    
    mask = torch.ones_like(values, dtype=torch.bool)
    for expert_id in selected:
        layer_id = expert_id // 128
        i = next(k for k, v in enumerate(indices[layer_id]) if v == expert_id % 128)
        mask[layer_id,i] = 0
    values = torch.cumsum(values * mask, dim=-1)
    values.masked_fill_(~mask, torch.inf)
    for t in torch.arange(0.001, 1.001, 0.001):
        n = (values < t).sum() + len(selected)
        if n >= count:
            break

    for layer_id, (indices_, values_) in enumerate(zip(indices, values)):
        for i, v in zip(indices_, values_):
            if len(selected) >= count:
                break
            if v < t:
                selected.add(layer_id * 128 + i.item())

    assert len(selected) == count
    return sorted(selected)


def select_experts_globally(metrics: Tensor, count: int = 288, enc_count: Optional[int] = None) -> list[int]:
    """
    Apply our "global threshold" algorithm for pruning experts.

    Args:
        - metrics: Tensor of shape 12 x 128 containing the importance of each expert, obtained by calling `get_metrics`
        - count: total number of experts to keep
        - enc_count: number of encoder experts to keep

    Returns: list of expert ids that can be used to prune the model with `prune` or `load_model`
    """
    if enc_count:
        selected = threshold(
            metrics[:768],
            enc_count,
        )
        selected += [
            i + 768
            for i in threshold(
                metrics[768:],
                count - len(selected),
            )
        ]
        return selected
    else:
        return threshold(metrics, count)


def select_experts_per_layer(metrics: Tensor, count: int = 288, enc_count: Optional[int] = 216) -> list[int]:
    """
    Apply our "fixed per-layer" pruning strategy.

    Args:
        - metrics: Tensor of shape 12 x 128 containing the importance of each expert, obtained by calling `get_metrics`
        - count: total number of experts to keep
        - enc_count: number of encoder experts to keep

    Returns: list of expert ids that can be used to prune the model with `prune` or `load_model`
    """
    assert count % 12 == 0
    assert not enc_count or enc_count < count
    selected = set()
    enc_count = (enc_count or count // 2) // 6
    dec_count = (count - enc_count * 6) // 6
    for layer_id, layer_metrics in enumerate(metrics):
        max_count = enc_count if layer_id < 6 else dec_count
        for expert_id in layer_metrics.argsort(descending=True):
            layer_count = sum((1 for i in selected if i // 128 == layer_id), 0)
            if layer_count >= max_count:
                break
            selected.add(expert_id.item() + layer_id * 128)
    return sorted(selected)


def plot_experts_per_layer(expert_ids: list[int], fig_path: Optional[str] = None):
    """
    Create a bar chart showing the number of experts at each layer.
    """
    from matplotlib import pyplot as plt
    counts = {}

    for expert_id in sorted(expert_ids):
        layer_id = expert_id // 128 * 4 + 3
        if layer_id < 24:
            layer_name = f'enc-{layer_id + 1}'
        else:
            layer_name = f'dec-{layer_id - 24 + 1}'
        counts[layer_name] = counts.get(layer_name, 0) + 1

    x = list(counts.keys())
    y = list(counts.values())
    bars = plt.bar(x, y)
    for i in range(6):
        bars[i].set_color('mediumseagreen')
    plt.plot([-1, 12], [32, 32], "k--")
    plt.xticks(rotation=45)
    plt.ylabel('Experts per layer')
    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path)
    else:
        plt.show()
    plt.close()
