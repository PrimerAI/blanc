# BLANC
This is the reference implementation of BLANC-help and BLANC-tune as defined in [Fill in the BLANC: Human-free quality estimation of document summaries](https://www.aclweb.org/anthology/2020.eval4nlp-1.2/), originally [in arxiv](https://arxiv.org/abs/2002.09836). 

BLANC is a new approach to the automatic estimation of document summary quality. Our goal is to measure the functional performance of a summary with an objective, reproducible, and fully automated method. Our approach achieves this by measuring the performance boost gained by a pre-trained language model with access to a document summary while carrying out its language understanding task on the document's text. We present evidence that BLANC scores have at least as good correlation with human evaluations as do the ROUGE family of summary quality measurements. And unlike ROUGE, the BLANC method does not require human-written reference summaries, allowing for fully human-free summary quality estimation.

Two types of BLANC scores are introduced in the paper and available in this repo: BLANC-help and BLANC-tune. BLANC-help is faster to calculate (around 30% faster on CUDA with default settings), but BLANC-tune is more theoretically principled. They are around 90% correlated with each other, so either one can be used in most cases. We found that BLANC with gap=2 on average works the best [Sensitivity of BLANC to human-scored qualities of text summaries](https://arxiv.org/abs/2010.06716), it is now set as default. The original paper used gap=6. The datasets are in [data](https://github.com/PrimerAI/blanc/tree/master/data).

## Setup
1. Install Python 3.6 or higher
2. Install with `pip install blanc`

Note that we pinned the `transformers` requirement to 2.4.0 due to Rust installation issues that are introduced in future versions. We used `transformers` 2.5.1 in our experiments, so you may want to try upgrading `transformers` if you can.

## Python Usage
Basic usage:
```python
>>> from blanc import BlancHelp, BlancTune
>>> document = "Jack drove his minivan to the bazaar to purchase milk and honey for his large family."
>>> summary = "Jack bought milk and honey."
>>> blanc_help = BlancHelp()
>>> blanc_tune = BlancTune(finetune_mask_evenly=False, show_progress_bar=False)
>>> blanc_help.eval_once(document, summary)
0.2222222222222222
>>> blanc_tune.eval_once(document, summary)
0.3333333333333333
```

By default, BLANC is run on the CPU. Using CUDA with batching is much faster:
```python
blanc_help = BlancHelp(device='cuda', inference_batch_size=128)
blanc_tune = BlancTune(device='cuda', inference_batch_size=24, finetune_mask_evenly=False, finetune_batch_size=24)
```
With these batch sizes, BLANC-help takes around 1.4 sec per summary and BLANC-tune takes around 1.8 sec per summary on an NVIDIA V100. In addition to the parameters controlling device and batch sizes, BlancHelp and BlancTune take several other parameters controlling how the BLANC scores are calculated, and the default values for those parameters reproduce the results of the paper.

If you want to compute the BLANC scores of many documents and summaries at once, you can use `eval_pairs()` or `eval_summaries_for_docs()`. `eval_pairs()` is useful when you have many documents, each with a single summary:
```python
>>> documents = ["Jack drove his minivan to the bazaar to purchase milk and honey for his large family.", "As Jill started taking a walk in the park, she certainly noticed that the trees were extra green this year."]
>>> summaries = ["Jack bought milk and honey.", "Jill saw green trees in the park."]
>>> blanc_help.eval_pairs(documents, summaries)
[0.2222222222222222, 0.0]
```

`eval_summaries_for_docs()` is useful when you have many documents, each with many summaries:
```python
>>> doc_summaries = [["Jack bought milk and honey.", "Jack drove to the bazaar in a minivan"], ["Jill saw green trees in the park.", "The trees were green."]]
>>> blanc_tune.eval_summaries_for_docs(documents, doc_summaries)
[[0.2222222222222222, 0.2222222222222222], [-0.07142857142857142, -0.14285714285714285]]
```

## CLI Usage
A CLI for computing BLANC scores is provided for convenience.
```
$ blanc help --gap 6 --doc "Jack drove his minivan to the bazaar to purchase milk and honey for his large family." --summary "Jack bought milk and honey."
0.1111111111111111
```

Input data can also be provided in JSON format, with sample JSON input provided in `data/`
```
$ blanc help --single_json data/single.json --gap 6
0.1111111111111111
$ blanc tune --pairs_json data/pairs.json --gap 6 --finetune_mask_evenly False
[0.2222222222222222, 0.14285714285714285]
$ blanc tune --doc_summaries_json data/doc-summaries.json --gap 6 --finetune_mask_evenly False
[[0.2222222222222222, 0.2222222222222222], [0.14285714285714285, 0.07142857142857142]]
```

The `single_json` input format expects a single JSON blob with keys `document` and `summary`. The `pairs_json` input format expects a list of JSON blobs, each with a `document` and a `summary`. The `doc_summaries_json` input format expects a list of JSON blobs, each with keys `document` and `summaries`, where `summaries` is a list of strings. These keys are customizable with the `doc_key`, `summary_key`, and `summaries_key` arguments. By default, the output is printed to STDOUT, but it can be written to a JSON file provided with the `output_json` argument.

Full documentation is available with `blanc --help`:
```
required arguments:
  {help,tune}           BLANC-help or BLANC-tune

input arguments:
  --doc DOC             single input document (default: None)
  --summary SUMMARY     single input summary (default: None)
  --single_json FILENAME
                        filename for single document summary pair (default:
                        None)
  --pairs_json FILENAME
                        filename for list of document summary pairs (default:
                        None)
  --doc_summaries_json FILENAME
                        filename for list of documents, each with a list of
                        summaries (default: None)
  --doc_key KEY         json key for the input document (default: doc)
  --summary_key KEY     json key for the input summary (single_json or
                        pairs_json input) (default: summary)
  --summaries_key KEY   json key for the input summaries (doc_summaries_json
                        input) (default: summaries)

arguments for BLANC-help and BLANC-tune:
  --model_name NAME     BERT model type (default: bert-base-uncased)
  --measure {improve,relative}
                        measure improve or relative, as defined in the paper
                        (default: relative)
  --gap GAP             distance between words to mask during inference
                        (default: 2)
  --gap_mask NUM        number of tokens to mask during inference at each 
                        gap-defined position
                        (default: 1)
  --min_token_length_normal LEN
                        minimum number of chars in normal tokens to mask,
                        where a normal token is a whole word (default: 4)
  --min_token_length_lead LEN
                        minimum number of chars in lead token to mask, where a
                        lead token begins a word (default: 2)
  --min_token_length_followup LEN
                        minimum number of chars in followup token to mask,
                        where a followup token continues a word (default: 100)
  --device DEVICE       cpu or cuda device (default: cpu)
  --random_seed SEED    random seed for python and torch (default: 1)
  --inference_batch_size SIZE
                        batch size to use during inference (default: 1)
  --inference_mask_evenly MASK_EVENLY
                        when True, mask every `gap` tokens that are longer
                        than `min_token_length`during finetuning, when False
                        randomly mask tokens with probability 0.15 (default:
                        True)

BLANC-help arguments:
  --filler_token TOKEN  token to use as filler in lieu of summary (default: .)
  --help_sep SEP        token to use to separate the summary or filler from
                        the sentence, or '' for no separator (default: )

BLANC-tune arguments:
  --finetune_batch_size SIZE
                        batch size to use when finetuning on summary (default:
                        1)
  --finetune_epochs EPOCHS
                        number of epochs to train for when finetuning on
                        summary (default: 10)
  --finetune_mask_evenly MASK_EVENLY
                        when True, mask every `gap` tokens that are longer
                        than `min_token_length`during finetuning, when False
                        randomly mask tokens with probability 0.15 (default:
                        False)
  --finetune_chunk_size SIZE
                        number of summary tokens to use at a time when
                        finetuning (default: 64)
  --finetune_chunk_stride STRIDE
                        number of tokens between summary chunks for finetuning
                        (default: 32)
  --learning_rate LR    learning rate when finetuning on summary (default:
                        5e-05)
  --warmup_steps STEPS  warmup steps when finetuning on summary (default: 0)
  ```
