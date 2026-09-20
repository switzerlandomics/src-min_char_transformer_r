#!/usr/bin/env Rscript
# Render read-only figures from a completed or in-progress experiment directory.
arguments <- commandArgs(trailingOnly = FALSE)
file_argument <- grep("^--file=", arguments, value = TRUE)
script <- if (length(file_argument)) sub("^--file=", "", file_argument[1L]) else "experiments/plot_results.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
source(file.path(root, "R", "plots.R"))
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) || any(args %in% c("-h", "--help"))) {
  cat("Usage: Rscript experiments/plot_results.R EXPERIMENT_DIR [--watch] [--refresh=SECONDS]\n")
  quit(status = if (!length(args)) 1L else 0L)
}
directory <- normalizePath(args[1L], mustWork = TRUE)
watch <- "--watch" %in% args
refresh_options <- grep("^--refresh=", args, value = TRUE)
refresh <- if (length(refresh_options)) as.numeric(sub("^--refresh=", "", refresh_options[1L])) else 10
if (!is.finite(refresh) || refresh < 1) stop("Refresh interval must be >= 1 second.")
if (length(setdiff(args[-1L], c("--watch", refresh_options)))) stop("Unrecognised option.")
render <- function() {
  result <- plot_experiment(directory)
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
