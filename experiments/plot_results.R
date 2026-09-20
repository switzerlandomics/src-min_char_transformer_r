#!/usr/bin/env Rscript
# Recreate the monitor and optional figures from saved experiment evidence.
# Earlier runs can recover their recorded initial text without retraining.
arguments <- commandArgs(trailingOnly = FALSE)
file_argument <- grep("^--file=", arguments, value = TRUE)
script <- if (length(file_argument)) sub("^--file=", "", file_argument[1L]) else "experiments/plot_results.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
# These modules are used only if an older run needs its saved initial sample
# paired with a new sample from its genuine best-validation checkpoint.
for (module in c("data", "model", "progress", "plots")) {
  source(file.path(root, "R", paste0(module, ".R")))
}
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) || any(args %in% c("-h", "--help"))) {
  cat("Usage: Rscript experiments/plot_results.R EXPERIMENT_DIR [--watch] [--refresh=SECONDS]\n")
  quit(status = if (!length(args)) 1L else 0L)
}
directory <- normalizePath(args[1L], mustWork = TRUE)
setwd(root)  # Resolve saved, relative input-corpus paths against the repository.
watch <- "--watch" %in% args
refresh_options <- grep("^--refresh=", args, value = TRUE)
refresh <- if (length(refresh_options)) as.numeric(sub("^--refresh=", "", refresh_options[1L])) else 10
if (!is.finite(refresh) || refresh < 1) stop("Refresh interval must be >= 1 second.")
if (length(setdiff(args[-1L], c("--watch", refresh_options)))) stop("Unrecognised option.")
restore_earlier_results <- function() {
  comparison_path <- file.path(directory, "generation_comparison.rds")
  if (file.exists(comparison_path)) return(invisible(FALSE))
  metadata_path <- file.path(directory, "metadata.rds")
  vocab_path <- file.path(directory, "vocab.rds")
  if (!file.exists(metadata_path) || !file.exists(vocab_path)) {
    message("[SAMPLES] No metadata or vocabulary: older generation comparison unavailable.")
    return(invisible(FALSE))
  }
  metadata <- readRDS(metadata_path)
  corpus <- metadata$input_file
  if (is.null(corpus) || !file.exists(corpus)) {
    message("[SAMPLES] Original corpus missing; cannot establish the matching prompt.")
    return(invisible(FALSE))
  }
  prompt <- if (!is.null(metadata$generation_prompt)) metadata$generation_prompt else {
    substr(read_text_file(corpus), 1L, min(4L, metadata$config$context_length))
  }
  metrics <- read_experiment_metrics(directory)
  initial_row <- metrics[metrics$iteration == 0L, , drop = FALSE]
  best_row <- best_validation_row(metrics)
  if (!nrow(initial_row) || is.null(best_row)) return(invisible(FALSE))
  restored <- restore_generation_comparison(directory, load_vocab(vocab_path),
    prompt, metadata$config, initial_row$validation_loss_nats_per_char[1L],
    best_row$iteration[1L], best_row$validation_loss_nats_per_char[1L])
  if (is.null(restored)) {
    message("[SAMPLES] Earlier initial sample or best model missing; no comparison generated.")
    return(invisible(FALSE))
  }
  write_rds(restored, comparison_path)
  message("[SAMPLES] Recovered original iteration-0 text and generated from saved best_model.rds.")
  invisible(TRUE)
}

render <- function() {
  restore_earlier_results()
  if (requireNamespace("ggplot2", quietly = TRUE) &&
      utils::packageVersion("ggplot2") >= "3.4.0") {
    result <- plot_experiment(directory)
  } else {
    write_monitor_html(directory)
    result <- list(best = best_validation_row(read_experiment_metrics(directory)))
    message("[PLOT] ggplot2 >= 3.4 is unavailable; recreated the HTML and text comparison only.")
  }
  cat(sprintf("[PLOT] best validation %.6f at update %d | %s\n",
    result$best$validation_loss_nats_per_char, result$best$iteration,
    file.path(directory, "training.html")))
}
render()
if (watch) {
  path <- file.path(directory, "metrics.csv")
  previous <- file.info(path)$mtime
  repeat {
    if (file.exists(file.path(directory, "FINISHED"))) break
    Sys.sleep(refresh)
    current <- file.info(path)$mtime
    if (!is.na(current) && !identical(current, previous)) {
      tryCatch(render(), error = function(e) message("[PLOT] ", conditionMessage(e)))
      previous <- current
    }
  }
  cat("[PLOT] Experiment complete.\n")
}
