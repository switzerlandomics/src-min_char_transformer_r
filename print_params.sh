#!/bin/bash

# One-time solution to print the full configuration for the last, largest, experiment. 
# Next time we can add a log file to do this directly rather than having params scattered.

Rscript -e '
cfg <- readRDS("output/20260920_190535_seed666/metadata.rds")$config

keys <- setdiff(names(cfg), c("plot", "smoke", "resume", "test"))
options <- vapply(keys, function(k) {
  paste0("--", gsub("_", "-", k), "=", shQuote(as.character(cfg[[k]])))
}, character(1))

flags <- c(
  if (isTRUE(cfg$smoke)) "--smoke",
  if (identical(cfg$plot, FALSE)) "--no-plot"
)

cat(paste(c("Rscript experiments/run.R", options, flags), collapse = " "), "\n")
'
