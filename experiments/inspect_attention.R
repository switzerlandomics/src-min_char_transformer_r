#!/usr/bin/env Rscript
# Inspect the actual attention weights for any prompt and saved model checkpoint.
arguments <- commandArgs(trailingOnly = FALSE)
file_argument <- grep("^--file=", arguments, value = TRUE)
script <- if (length(file_argument)) sub("^--file=", "", file_argument[1L]) else "experiments/inspect_attention.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
setwd(root)
source("R/data.R"); source("R/model.R"); source("R/plots.R")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || any(args %in% c("-h", "--help"))) {
  cat('Usage: Rscript experiments/inspect_attention.R EXPERIMENT_DIR [PROMPT] [--model=best_model.rds]\n')
  quit(status = if (!length(args)) 1L else 0L)
}
directory <- normalizePath(args[1L], mustWork = TRUE)
model_argument <- grep("^--model=", args, value = TRUE)
model_file <- if (length(model_argument)) sub("^--model=", "", model_argument[1L]) else "best_model.rds"
if (!model_file %in% c("best_model.rds", "model.rds")) stop("Choose best_model.rds or model.rds.")
prompt_args <- args[-1L][!grepl("^--model=", args[-1L])]
if (length(prompt_args) > 1L) stop("Pass at most one prompt argument.")
prompt <- if (length(prompt_args)) prompt_args[1L] else "The king"
model <- load_model(file.path(directory, model_file))
vocab <- load_vocab(file.path(directory, "vocab.rds"))
indices <- encode_text(prompt, vocab)
indices <- tail(indices, model$config$context_length)
result <- forward_transformer(model, indices, retain_cache = FALSE)
characters <- vocab$ix_to_char[indices]
output <- file.path(directory, "attention_inspection.png")
require_plots()
ggplot2::ggsave(output,
  plot_attention_matrix(result$attention, characters,
                        title = paste("Causal attention:", prompt)),
  width = 7.5, height = 6.8, dpi = 150, bg = "white")
cat("[ATTENTION] ", output, "\n", sep = "")
cat("[PREDICTION] Next-character probabilities for final prompt position:\n")
probabilities <- row_softmax(result$logits)
rank <- order(probabilities[nrow(probabilities), ], decreasing = TRUE)
for (i in head(rank, 10L)) {
  cat(sprintf("  %-12s %.4f\n", encodeString(vocab$ix_to_char[i], quote = '"'),
              probabilities[nrow(probabilities), i]))
}
