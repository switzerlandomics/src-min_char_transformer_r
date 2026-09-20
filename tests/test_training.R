#!/usr/bin/env Rscript
# Learn a simple predictable pattern; check deterministic model/optimiser state.
args <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script <- if (length(args)) sub("^--file=", "", args[1L]) else "tests/test_training.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
source(file.path(root, "R", "model.R"))
source(file.path(root, "R", "optimiser.R"))
source(file.path(root, "R", "data.R"))
source(file.path(root, "R", "progress.R"))
source(file.path(root, "R", "plots.R"))

train_step <- function(model, state, inputs, targets) {
  result <- loss_and_gradients(model, inputs, targets)
  adam_update(model, clip_gradients(result$grads, 1)$grads, state, 0.005)
}
set.seed(12)
initial <- initialise_model(2L, 4L, 8L, 12L)
inputs <- c(1L, 2L, 1L, 2L)
targets <- c(2L, 1L, 2L, 1L)
initial_loss <- sequence_loss(initial, inputs, targets)
model <- initial
state <- new_adam_state(model)
for (iteration in seq_len(100L)) {
  updated <- train_step(model, state, inputs, targets)
  model <- updated$model; state <- updated$state
}
final_loss <- sequence_loss(model, inputs, targets)
stopifnot(is.finite(final_loss), final_loss < initial_loss * 0.75)
cat(sprintf("[PASS] Tiny-pattern learning | loss %.4f -> %.4f\n",
            initial_loss, final_loss))

# Model and optimiser state produce identical outputs after a serialized resume.
model <- initial
state <- new_adam_state(model)
for (iteration in seq_len(50L)) {
  updated <- train_step(model, state, inputs, targets)
  model <- updated$model; state <- updated$state
}
checkpoint <- tempfile(fileext = ".rds")
saveRDS(list(model = model, state = state), checkpoint)
continued <- readRDS(checkpoint)
for (iteration in seq_len(50L)) {
  updated <- train_step(continued$model, continued$state, inputs, targets)
  continued$model <- updated$model; continued$state <- updated$state
}
stopifnot(identical(model$params$token_embedding, initial$params$token_embedding) == FALSE)
# Rebuild the uninterrupted 100-update reference from the same initial weights.
reference <- initial; reference_state <- new_adam_state(reference)
for (iteration in seq_len(100L)) {
  updated <- train_step(reference, reference_state, inputs, targets)
  reference <- updated$model; reference_state <- updated$state
}
stopifnot(identical(continued$model, reference),
          identical(continued$state, reference_state))
cat("[PASS] Serialised training continuation reproduces uninterrupted parameters\n")

# Exercise the actual CLI, full checkpoint and training-window RNG state.
# Runs without a download or ggplot2; compares 20 continuous updates with
# 10 saved updates followed by a resume to the same total of 20 updates.
runner <- file.path(root, "experiments", "run.R")
rscript <- file.path(R.home("bin"), "Rscript")
reference_output <- tempfile("continuous-")
resumed_output <- tempfile("resumed-")
invoke_runner <- function(arguments) {
  result <- suppressWarnings(system2(rscript, c(shQuote(runner), arguments),
                                      stdout = TRUE, stderr = TRUE))
  status <- attr(result, "status")
  if (!is.null(status) && status != 0L) {
    stop("Runner subprocess failed:\n", paste(result, collapse = "\n"))
  }
  invisible(result)
}
invoke_runner(c("--smoke", "--no-plot", "--iterations=20",
                paste0("--output=", shQuote(reference_output))))
invoke_runner(c("--smoke", "--no-plot", "--iterations=10",
                paste0("--output=", shQuote(resumed_output))))
resumed_directory <- list.files(resumed_output, full.names = TRUE)
stopifnot(length(resumed_directory) == 1L)
invoke_runner(c(paste0("--resume=", shQuote(resumed_directory)),
                "--no-plot", "--iterations=20"))
reference_directory <- list.files(reference_output, full.names = TRUE)
stopifnot(length(reference_directory) == 1L)
continuous <- readRDS(file.path(reference_directory, "latest_checkpoint.rds"))
resumed <- readRDS(file.path(resumed_directory, "latest_checkpoint.rds"))
stopifnot(identical(continuous$model, resumed$model),
          identical(continuous$optimiser, resumed$optimiser),
          identical(continuous$rng_state, resumed$rng_state),
          identical(continuous$window_order, resumed$window_order),
          identical(continuous$next_window, resumed$next_window),
          file.exists(file.path(resumed_directory, "FINISHED")),
          file.exists(file.path(reference_directory, "training.html")),
          file.exists(file.path(reference_directory, "generation_comparison.txt")),
          file.exists(file.path(resumed_directory, "training.html")))
cat("[PASS] Real runner: continuous and resumed smoke runs agree at update 20\n")

# A best-model sample uses exactly the same generation conditions as the saved
# iteration-0 sample. It is regenerated from best_model.rds, not selected by eye.
comparison <- readRDS(file.path(reference_directory, "generation_comparison.rds"))
resumed_comparison <- readRDS(file.path(resumed_directory, "generation_comparison.rds"))
metadata <- readRDS(file.path(reference_directory, "metadata.rds"))
stopifnot(identical(comparison, resumed_comparison),
          comparison$initial$iteration == 0L,
          comparison$best$iteration == metadata$best_iteration,
          identical(comparison$best$validation_loss_nats_per_char,
                    metadata$best_validation_nats_per_char),
          identical(comparison$initial$prompt, comparison$best$prompt),
          identical(comparison$initial$seed, comparison$best$seed),
          identical(comparison$initial$n_generated, comparison$best$n_generated),
          identical(comparison$initial$temperature, comparison$best$temperature))
vocab <- load_vocab(file.path(reference_directory, "vocab.rds"))
selected_model <- load_model(file.path(reference_directory, "best_model.rds"))
regenerated <- make_generation_sample(selected_model, vocab,
  comparison$initial$prompt, comparison$best$iteration,
  comparison$best$validation_loss_nats_per_char,
  comparison$best$n_generated, comparison$best$seed,
  comparison$best$temperature)
stopifnot(identical(regenerated, comparison$best))
recorded_initial <- read_initial_generation_sample(reference_directory,
  comparison$initial$prompt, comparison$initial$validation_loss_nats_per_char,
  comparison$initial$seed)
stopifnot(identical(recorded_initial, comparison$initial))
stopifnot(identical(restore_generation_comparison(reference_directory, vocab,
  comparison$initial$prompt, metadata$config,
  comparison$initial$validation_loss_nats_per_char,
  metadata$best_iteration, metadata$best_validation_nats_per_char), comparison))
html <- paste(readLines(file.path(reference_directory, "training.html"),
  warn = FALSE), collapse = "\n")
stopifnot(grepl("Best-validation checkpoint", html, fixed = TRUE),
          !grepl('meta http-equiv="refresh"', html, fixed = TRUE))
comparison_text <- paste(readLines(file.path(reference_directory,
  "generation_comparison.txt"), warn = FALSE), collapse = "\n")
stopifnot(grepl(comparison$initial$generated_text, comparison_text, fixed = TRUE),
          grepl(comparison$best$generated_text, comparison_text, fixed = TRUE))
cat("[PASS] Exact initial/best text, text-only HTML, real selected weights, legacy recovery, deterministic resume\n")

unlink(checkpoint)
unlink(c(reference_output, resumed_output), recursive = TRUE)
