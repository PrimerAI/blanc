# ESTIME
This is the reference implementation of ESTIME as defined in [Estimation of Summary-to-Text Inconsistency by Mismatched Embeddings](https://arxiv.org/abs/2104.05156).

ESTIME is a reference-free estimator of summary quality with emphasis on factual consistency. It can be used for filtering generated summaries, or for estimating improvement of a generation system.

Usage is simple: create `Estime`, and use `evaluate_claims`.