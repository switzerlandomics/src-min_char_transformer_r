# Results and publication checklist

This document intentionally contains **no invented training measurements**.
The default configuration is an initial experiment whose runtime, learning
curves and generated text must be observed and recorded before publication.

Once a reference experiment has completed, record its experiment-directory
identifier, input checksum, R version, model dimensions, exact parameter
count, training update budget, characters processed, approximate corpus
passes, elapsed wall-clock time and peak memory where measured. Include the
initial, best-validation and final validation measurements, the train-only
uniform and bigram baseline losses and the untouched final test measurement.

Copy only selected, confirmed experiment figures into `docs/figures/`.
`training.png`, `training_mobile.png`, `validation_detail.png` and
`attention.png` can be rebuilt using `experiments/plot_results.R`; the latter
must be traceable to `attention_snapshot.rds` and a real model checkpoint.
Preserve generated samples verbatim and state the prompt, temperature and seed.

A decreasing training loss is not sufficient evidence of correct gradients or
of generalisation. Check the unit tests, numerical gradient checks, causal
prefix-invariance tests, independent held-out loss and reproduction of
stopped-and-resumed training. If the model fails to improve relative to the
bigram baseline or shows volatile validation loss, report those findings
rather than substituting an undocumented run.
