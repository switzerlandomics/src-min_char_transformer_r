#!/usr/bin/env Rscript
# A plot-only test: no neural-network training is performed.
args <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script <- if (length(args)) sub("^--file=", "", args[1L]) else "tests/test_monitoring.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
source(file.path(root, "R", "plots.R"))
directory <- tempfile("transformer-monitor-")
dir.create(directory)
metrics <- data.frame(iteration = c(0L, 10L, 20L),
  train_loss_nats_per_char = c(NA_real_, 3.5, 2.9),
  validation_loss_nats_per_char = c(4.1, 3.2, 3.4),
  elapsed_seconds = c(0, 1, 2), characters_processed = c(0, 80, 160),
  approximate_corpus_passes = c(0, 0.1, 0.2))
utils::write.csv(metrics, file.path(directory, "metrics.csv"), row.names = FALSE)
saveRDS(list(uniform_baseline_nats_per_char = 4.1,
             bigram_baseline_nats_per_char = 3.5), file.path(directory, "metadata.rds"))
stopifnot(nrow(read_experiment_metrics(directory)) == 3L,
          best_validation_row(metrics)$iteration == 10L)
if (requireNamespace("ggplot2", quietly = TRUE) &&
    utils::packageVersion("ggplot2") >= "3.4.0") {
  initial <- metrics[1L, , drop = FALSE]
  utils::write.csv(initial, file.path(directory, "metrics.csv"), row.names = FALSE)
  plot_experiment(directory) # Initial validation only: no training line yet.
  utils::write.csv(metrics, file.path(directory, "metrics.csv"), row.names = FALSE)
  result <- plot_experiment(directory)
  stopifnot(all(file.exists(result$files)), all(file.info(result$files)$size > 0))
  message("[PASS] Metric parsing, desktop/mobile PNG and browser monitor")
} else message("[PASS] Metric parsing (optional ggplot2 rendering skipped)")

unlink(directory, recursive = TRUE)
