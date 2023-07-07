# NLLB Pruning

[`nllb_pruning.py`](nllb_pruning.py) contains a small library to use the pruning method described in our paper
[Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model](https://arxiv.org/abs/2212.09811).

By default, our recommended pruning is used (80% fixed-per-layer and per-language with a 3:1 ratio), thanks to the expert ids stored in [`experts.json`](experts.json).
However, custom pruning can be applied (e.g., different pruning rates, granularities or algorithms described in the paper).

The JSON files `stats-*.json` contain expert statistics gathered by decoding FLORES dev with NLLB in 30, 53 or all 202 languages. The 30 and 53-language versions contain
statistics for each language direction (obtained with beam search decoding). The 202-language version contains aggregated statistics to each language and from each language (obtained with teacher forcing). These statistics can be used to compute expert importance metrics (`get_metrics`) and prune experts with the desired algorithm (`select_experts_globally` or `select_experts_per_layer`).

The TSV files [`scores/*.tsv`](scores/) contain spBLEU and chrF++ scores for each language direction by our main pruning method and by the baseline models (NLLB MoE 54B and NLLB dense 3.3B).

## How to use the Python library

### Install dependencies

```bash
pip install transformers accelerate datasets huggingface_hub sacrebleu --upgrade
```

### Basic usage

The function `load_and_prune_for_lang_pair` downloads the NLLB-200 checkpoints from [HuggingFace](https://huggingface.co/facebook/nllb-moe-54b) (200G in total),
if they are not in the HuggingFace cache already.
Then it loads the model in CPU memory (100G), prunes it for a given language pair and moves it to GPU memory (approx. 28G).
The pruned experts are actually not removed, but stored in CPU memory for later use.

`prune_for_lang_pair` can then be called to prune the model to another language pair, without need to reload it from the disk.

```python
from nllb_pruning import load_and_prune_for_lang_pair, prune_for_lang_pair, load_tokenizer, translate

model = load_and_prune_for_lang_pair(
    source_lang='eng_Latn',
    target_lang='fra_Latn',
)  # ~28G of GPU memory, ~100G of CPU memory

tokenizer = load_tokenizer()

translate(model, tokenizer, ['She sells seashells by the seashore.'], source_lang='eng_Latn', target_lang='fra_Latn')
# ['Elle vend des coquillages au bord de la mer.']

prune_for_lang_pair(model, source_lang='eng_Latn', target_lang='deu_Latn')
translate(model, tokenizer, ['She sells seashells by the seashore.'], source_lang='eng_Latn', target_lang='deu_Latn')
```

CPU memory can be saved by pruning the model right from the start. However, you will have to reload the entire model
if you want to use it on another language pair. A different cache directory can also be specified if `.cache` is not
large enough to store the model checkpoints (200G in total).

```python
model = load_and_prune_for_lang_pair(
    source_lang='eng_Latn',
    target_lang='fra_Latn',
    cache_dir='/some_large_disk/.cache/huggingface/hub',
    low_cpu_mem_usage=True,
)  # ~28G of GPU memory, ~28G of CPU memory

prune_for_lang_pair(model, source_lang='eng_Latn', target_lang='deu_Latn')  # this won't work, the model needs to be reloaded
```

### Evaluation

The function `decode_flores` can be used to translate [FLORES](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) dev or devtest in a given translation direction with a pruned model and compute a spBLEU score:

```python
hyps, score = decode_flores(
    model,
    tokenizer,
    source_lang='eng_Latn',
    target_lang='fra_Latn',
    num_beams=4,
    max_length=100,
    batch_size=10,
)  # BLEU = 57.13, ~9.5 min (on a A100), ~34G of GPU memory
```
GPU memory usage can be reduced by decreasing `batch_size`, `max_length` or `num_beams`.

### Custom pruning

The examples above are using the recommended expert pruning method, whose expert ids are stored in `experts.json`.
However, different pruning can be done either by specifying a path to a different JSON expert index (with the `expert_index`
argument of `load_and_prune_for_lang_pair` or `prune_for_lang_pair`), or by manually defining a list of experts to keep and calling
`prune` or `load_model` with it (`expert_ids` argument). More JSON files corresponding to the pruning strategies presented in the paper are available in [`more_results.tar.gz`](more_results.tar.gz).

Small clarification: an "expert index" is a JSON file containing the list of expert ids to keep for each language direction. "Expert ids" are just a list of experts to keep (identified with an integer between 0 and 1535).

Here is an example of how to apply custom pruning settings:

```python
metrics = get_metrics('stats-202.json', source_lang='eng_Latn', target_Lang='fra_Latn', metric='importance')
expert_ids = select_experts_per_layer(metrics, count=384, enc_count=192)  # 75% pruning at 1:1 ratio
prune(model, expert_ids)
```

## Translation outputs

The translations of FLORES devtest (in all 202*201 directions) with our default pruning approach can be downloaded from [here](https://download.europe.naverlabs.com/nllb-pruning-outputs.tar.gz). These correspond to the results presented in Tables 3, 12 and 13 in the paper.

## Citation

```bibtex
@inproceedings{koishekenov-etal-2023-nllb-pruning,
    title = "Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model",
    author = "Koishekenov, Yeskendir  and
      Berard, Alexandre  and
      Nikoulina, Vassilina",
    booktitle = "Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2212.09811",
}
```

## License

NLLB pruning, Copyright (C) 2023 Naver Corporation, is under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license](LICENSE.txt).

Meta AI's [NLLB models](https://github.com/facebookresearch/fairseq/tree/nllb), which we used in this work, is licensed under [CC-BY-NC 4.0](https://github.com/facebookresearch/fairseq/blob/nllb/LICENSE.model.md).

Meta AI's [FLORES datasets](https://github.com/facebookresearch/flores), which we used for evaluation and for extracting gate statistics, is licensed under [CC-BY-SA 4.0](https://github.com/facebookresearch/flores/blob/main/LICENSE_CC-BY-SA). The file `outputs.tar.gz` contains translations of FLORES devtest by our pruned NLLB. The JSON files contain expert statistics gathered by decoding FLORES dev with NLLB.
