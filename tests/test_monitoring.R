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

# The comparison is evidence from the model, not text embedded in plotting code.
# Check the exact initial-sample recovery and safe, readable HTML rendering.
initial <- list(iteration = 0L, validation_loss_nats_per_char = 4.1,
  prompt = "F", generated_text = "A <script> B & C\nSecond line",
  n_generated = nchar("A <script> B & C\nSecond line", type = "chars"),
  seed = 100042L, temperature = 1)
best <- initial
best$iteration <- 10L
best$validation_loss_nats_per_char <- 3.2
best$generated_text <- paste(rep("T", initial$n_generated), collapse = "")
writeLines(c("", "=== update 0 | validation 4.100000 | prompt \"F\" | temperature 1 ===",
             paste0(initial$prompt, initial$generated_text), "",
             "=== update 10 | validation 3.200000 | prompt \"F\" | temperature 1 ===",
             paste0(best$prompt, best$generated_text)),
           file.path(directory, "samples.txt"), useBytes = TRUE)
recovered <- read_initial_generation_sample(directory, "F", 4.1, 100042L)
stopifnot(identical(recovered$generated_text, initial$generated_text),
          identical(recovered$prompt, initial$prompt))
saveRDS(list(initial = initial, best = best),
        file.path(directory, "generation_comparison.rds"))
write_monitor_html(directory)
comparison_text <- paste(readLines(file.path(directory, "generation_comparison.txt"),
  warn = FALSE), collapse = "\n")
stopifnot(grepl(initial$generated_text, comparison_text, fixed = TRUE),
          grepl(best$generated_text, comparison_text, fixed = TRUE),
          grepl("update 10", comparison_text, fixed = TRUE))
html <- paste(readLines(file.path(directory, "training.html"), warn = FALSE), collapse = "\n")
stopifnot(grepl("Text generation: before and after training", html, fixed = TRUE),
          grepl("Best-validation checkpoint", html, fixed = TRUE),
          grepl("&lt;script&gt;", html, fixed = TRUE),
          grepl("A &lt;script&gt; B &amp; C\nSecond line", html, fixed = TRUE),
          !grepl("<script>", html, fixed = TRUE),
          grepl('href="generation_comparison.txt"', html, fixed = TRUE),
          grepl("white-space:pre-wrap", html, fixed = TRUE),
          grepl('meta http-equiv="refresh"', html, fixed = TRUE))
file.create(file.path(directory, "FINISHED"))
write_monitor_html(directory)
html <- paste(readLines(file.path(directory, "training.html"), warn = FALSE), collapse = "\n")
stopifnot(!grepl('meta http-equiv="refresh"', html, fixed = TRUE))
message("[PASS] Recorded initial text, plain-text export, escaped HTML, mobile layout, final page")
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
