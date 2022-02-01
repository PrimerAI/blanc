# ESTIME
This is the implementation of ESTIME. The 'soft ESTIME' and 'coherence' were defined in [Consistency and Coherence from Points of Contextual Similarity](https://arxiv.org/abs/2112.11638); the 'number of alarms' was defined in [ESTIME: Estimation of Summary-to-Text Inconsistency by Mismatched Embeddings](https://aclanthology.org/2021.eval4nlp-1.10/).

ESTIME is a reference-free estimator of summary quality with emphasis on factual consistency. It can be used for filtering generated summaries, or for estimating improvement of a generation system.

Usage is simple: create `Estime`, and use `evaluate_claims`. When creating Estime, specify the list of names of the measures to obtain for each claim.

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









