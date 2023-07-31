# ESTIME

ESTIME as the 'number of alarms' was defined in [ESTIME: Estimation of Summary-to-Text Inconsistency by Mismatched Embeddings](https://aclanthology.org/2021.eval4nlp-1.10/).
ESTIME-soft and ESTIME-coherence were defined in [Consistency and Coherence from Points of Contextual Similarity](https://arxiv.org/abs/2112.11638). Sourse: [estime](https://github.com/PrimerAI/blanc/blob/master/blanc/estime.py).

ESTIME is a reference-free estimator of summary quality with emphasis on factual consistency. It can be used for filtering generated summaries, or for estimating improvement of a generation system.

Usage is simple: create `Estime`, and use `evaluate_claims`. When creating Estime, specify the list of names of the measures to obtain for each claim. Basic usage:

```python
>>> from blanc import Estime
>>> estimator = Estime()
>>> text = """In Kander’s telling, Mandel called him up out of the blue a decade or so ago to pitch a project. It made sense why. The two men had similar profiles: Jewish combat veterans in their early 30s. New statewide officeholders in the Midwest."""
>>> summary = """Kander and Mandel had similar profiles, and it makes sense."""
>>> estimator.evaluate_claims(text, [summary])
[[5]]
```

Default `device` in Estime() is `device`='cpu'. It can be set `device`='cuda'.

In the example above only one summary is given to the text, and hence the list of results contains only one element [5] - the scores only for this summary. The scores list contains only single score =5, because by default the list of measures contains only one measure 'alarms'. More measures can be included: 'alarms', 'alarms_adjusted', 'alarms_alltokens', 'soft', 'coherence'. For example:

```
>>> estimator = Estime(output=['alarms', 'alarms_adjusted', 'soft', 'coherence'])
>>> estimator.evaluate_claims(text, [summary])
[[5, 7.5, 0.502, -0.25]]
```
The results appear in the same order as the names given in `output`. The measures 'alarms' (the original ESTIME), 'soft' and 'coherence' are as defined in the papers. The only difference is that when there are no any tokens overlap between the claim and the text, the 'alarms' is set to the number of the tokens in the summary. Unlike 'soft', the original ESTIME does not make good estimation for the cases where the number of overlap tokens is much less than the total number of summary tokens. Starting from the version 0.3.3, the measure 'alarms_adjusted' can be added. It is defined as `alarms_adjusted = alarms * N / M`, where M is the number of overlap tokens, and N is the total number of summary tokens. Thus, it serves as an extrapolation of the 'alarms' to the total number of summary tokens. When M=0, the 'alarms_adjusted' is set to N. For curiocity (not recommended), the 'alarms_alltokens' also can be added, it is defined as `alarms_alltokens = alarms + N - M`, meaning that any non-overlapping token is counted as an alarm. 

For more options, see comments in the source [estime](https://github.com/PrimerAI/blanc/blob/master/blanc/estime.py), or see [estime](https://github.com/PrimerAI/primer-research/tree/main/estime).

The table below is made in the same way as the Table 1 in [ESTIME](https://aclanthology.org/2021.eval4nlp-1.10/), except that the number of systems here is updated from 16 to 17, following the later version of [SummEval](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00373/100686/SummEval-Re-evaluating-Summarization-Evaluation). This means that the correlations are taken here between arrays of 1700-length (100 texts x 17 summary generation systems).

|model|consistency<br />Spearman|consistency<br />Kendall|relevance<br />Spearman|relevance<br />Kendall|coherence<br />Spearman|coherence<br />Kendall|fluency<br />Spearman|fluency<br />Kendall|
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
BLANC-AXXL|0.19|0.09|0.21|0.15|0.11|0.08|0.10|0.06|
BLANC-BLU|0.20|0.10|0.18|0.13|0.10|0.07|0.11|0.06|
BLANC|0.19|0.10|0.28|0.20|0.22|0.16|0.13|0.07|
ESTIME-12|0.36|0.18|0.10|0.07|0.20|0.14|0.32|0.19|
ESTIME-21|**0.39**|**0.19**|0.15|0.11|0.27|0.19|**0.38**|**0.22**|
ESTIME-24|0.34|0.17|0.08|0.06|0.16|0.11|0.34|0.20|
Jensen-Shannon|0.18|0.09|0.39|0.28|0.29|0.21|0.11|0.06|
SummaQA-F|0.17|0.08|0.14|0.10|0.08|0.06|0.12|0.07|
SummaQA-P|0.19|0.09|0.17|0.12|0.10|0.08|0.12|0.07|
SUPERT|0.28|0.14|0.26|0.19|0.20|0.15|0.17|0.10|
(r) BERTScore-F|0.10|0.05|0.38|0.28|**0.39**|**0.28**|0.13|0.07|
(r) BERTScore-P|0.05|0.03|0.29|0.21|0.34|0.25|0.11|0.06|
(r) BERTScore-R|0.15|0.08|**0.41**|**0.30**|0.34|0.249|0.11|0.06|
(r) BLEU|0.09|0.04|0.23|0.17|0.19|0.14|0.12|0.07|
(r) ROUGE-L|0.12|0.06|0.23|0.16|0.16|0.11|0.08|0.04|
(r) ROUGE-1|0.13|0.07|0.28|0.20|0.17|0.12|0.07|0.04|
(r) ROUGE-2|0.12|0.06|0.23|0.16|0.14|0.10|0.06|0.04|
(r) ROUGE-3|0.15|0.07|0.23|0.17|0.15|0.11|0.06|0.04|

(r): These measures need human-written reference summaries to evaluate a summary.<br />
The ESTIME and Jensen-Shannon scores are negated.
The third row is the default version of BLANC.

The numbers have slightly changed here compared to the 16-system data reported in [ESTIME](https://aclanthology.org/2021.eval4nlp-1.10/); the trends and the top correlations are the same.<br />
Notice that for consistency any reference-free measure outperforms all reference-needed measures.<br />









