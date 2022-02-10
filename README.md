# Evaluation measures

This repositary contains reference implementations and explanations to accompany [Primer.ai](https://primer.ai) research and publications related to evaluation measures, mostly for the purpose of summary evaluation.

These evaluation measures include:

* BLANC-help (or simply 'BLANC'), BLANC-tune
  * blanc.py
  * All the info is in this page
* Shannon Score, Information Difference, BLANC-Shannon
  * shannon.py
  * Info: [Shannon Score and Information Difference](https://github.com/PrimerAI/blanc/tree/master/shannon)
* ESTIME, ESTIME-soft, ESTIME-coherence
  * estime.py
  * Info: [ESTIME (hard, soft and coherence)](https://github.com/PrimerAI/blanc/tree/master/estime)

Annotated summary quality datasets: [data](https://github.com/PrimerAI/blanc/tree/master/data)


## Setup
1. Install Python 3.6 or higher
2. Install with `pip install blanc`


## BLANC
This is the reference implementation of BLANC-help and BLANC-tune as defined in [Fill in the BLANC: Human-free quality estimation of document summaries](https://www.aclweb.org/anthology/2020.eval4nlp-1.2/).

BLANC is a reference-free approach to the automatic estimation of document summary quality. Our goal is to measure the functional performance of a summary with an objective, reproducible, and fully automated method. Our approach achieves this by measuring the performance boost gained by a pre-trained language model with access to a document summary while carrying out its language understanding task on the document's text. Unlike ROUGE, BLANC does not require human-written reference summaries, allowing for fully human-free summary quality estimation.

Two types of BLANC scores were introduced in the paper and are available in this repo: BLANC-help and BLANC-tune. BLANC-help is faster to calculate (around 30% faster on CUDA with default settings), but BLANC-tune is more theoretically principled. They are around 90% correlated with each other, so either one can be used in most cases.<br />
BLANC-help with gap=2 on average correlates the best with human scores [Sensitivity of BLANC to human-scored qualities of text summaries](https://arxiv.org/abs/2010.06716), it is now set as default. The original paper used gap=6. Optimal parameters for BLANC-help and for BLANC-tune are found by using 'max-help' criterion, without relying on human summaries or human scores, in [Is Human Scoring the Best Criteria for Summary Evaluation?](https://aclanthology.org/2021.findings-acl.192) (the paper points to the possible bias of human experts).


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
With these batch sizes, BLANC-help takes around 1.4 sec per summary and BLANC-tune takes around 1.8 sec per summary on an NVIDIA V100. In addition to the parameters controlling device and batch sizes, BlancHelp and BlancTune take several other parameters controlling how the BLANC scores are calculated, and the default values for those parameters reproduce the results of the paper. BlancTune results may vary if random_seed is not set.

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
                        than `min_token_length` during finetuning, when False
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

  ## BLANC on [SummEval](https://github.com/Yale-LILY/SummEval) dataset
  BLANC can run on top of any pretrained BERT or AlBERT model (more will be added). The table below lists correlations of BLANC with human scores on the human-annotated [SummEval](https://github.com/Yale-LILY/SummEval) dataset (described in [SummEval: Re-evaluating Summarization Evaluation](https://arxiv.org/abs/2007.12626v4)). The dataset contains 1600 text-summary pairs by 100 texts x 16 systems. We show correlation (Spearman and Kendall's Tau-c) between BLANC-help and experts-average scores for each quality of the summary (coherence, consistency, fluency, relevance):

  |quality|model|Spearman|Kendall|
|:---------------|:-----------|-----:|-----:|
|coherence|bbu|0.122|0.09|
|coherence|bbc|0.197|0.142|
|coherence|blu|0.116|0.085|
|coherence|blc|0.226|0.165|
|coherence|bluw|0.083|0.06|
|coherence|blcw|0.196|0.142|
|coherence|ab|0.168|0.125|
|coherence|al|0.152|0.111|
|coherence|axl|0.15|0.11|
|coherence|axxl|0.127|0.093|
|consistency|bbu|0.19|0.094|
|consistency|bbc|0.19|0.094|
|consistency|blu|0.207|0.102|
|consistency|blc|0.204|0.1|
|consistency|bluw|0.167|0.082|
|consistency|blcw|0.18|0.089|
|consistency|ab|0.192|0.095|
|consistency|al|0.199|0.098|
|consistency|axl|0.179|0.088|
|consistency|axxl|0.2|0.098|
|fluency|bbu|0.089|0.051|
|fluency|bbc|0.108|0.062|
|fluency|blu|0.112|0.065|
|fluency|blc|0.113|0.064|
|fluency|bluw|0.107|0.061|
|fluency|blcw|0.121|0.069|
|fluency|ab|0.124|0.072|
|fluency|al|0.132|0.076|
|fluency|axl|0.119|0.069|
|fluency|axxl|0.115|0.066|
|relevance|bbu|0.216|0.156|
|relevance|bbc|0.278|0.201|
|relevance|blu|0.217|0.156|
|relevance|blc|0.306|0.223|
|relevance|bluw|0.194|0.14|
|relevance|blcw|0.258|0.188|
|relevance|ab|0.27|0.193|
|relevance|al|0.267|0.192|
|relevance|axl|0.245|0.176|
|relevance|axxl|0.246|0.179|

The [transformers](https://huggingface.co/transformers/pretrained_models.html) models are: bert-base-uncased (bbu), bert-base-cased (bbc), bert-large-uncased (blu), bert-large-cased (blc), bert-large-uncased-whole-word-masking (bluw), bert-large-cased-whole-word-masking (blcw), albert-base-v2 (ab), albert-large-v2 (al), albert-xlarge-v2 (axl), albert-xxlarge-v2 (axxl). The BLANC-help was used with the current default settings (gap=2, min_token_length_normal=4, min_token_length_lead=2, min_token_length_followup=100). All the p-values above are of order 10^-5 or lower.

The system-level correlations (correlations between 16-dimensional scores after averaging each system scores over 100 texts) have too high p-values. The table below shows only the correlations with p-values <0.05:

 |quality|model|Spearman|p|Kendall|p|
|:---------------|:-----------|-----:|-----:|-----:|-----:|
|consistency|bbu|0.738|0.001|0.567|0.002|
|consistency|bbc|0.759|0.001|0.533|0.003|
|consistency|blu|0.724|0.002|0.567|0.002|
|consistency|blc|0.788|0.0|0.567|0.002|
|consistency|bluw|0.771|0.0|0.617|0.001|
|consistency|blcw|0.791|0.0|0.6|0.001|
|consistency|ab|0.724|0.002|0.583|0.001|
|consistency|al|0.774|0.0|0.6|0.001|
|consistency|axl|0.706|0.002|0.517|0.005|
|consistency|axxl|0.812|0.0|0.617|0.001|
|fluency|bbc|0.558|0.025|0.444|0.017|
|fluency|blc|0.549|0.028|0.444|0.017|
|fluency|bluw|0.525|0.037|0.377|0.043|
|fluency|blcw|0.595|0.015|0.477|0.01|
|fluency|al|0.518|0.04|0.393|0.034|
|fluency|axxl|0.534|0.033|0.41|0.027|
|relevance|bbc| | |0.467|0.011|
|relevance|blc| | |0.467|0.011|
|relevance|blcw|0.515|0.041|0.467|0.011|
