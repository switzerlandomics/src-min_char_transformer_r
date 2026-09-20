#!/usr/bin/env Rscript
# Model and data invariants. Run from any working directory.
args <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script <- if (length(args)) sub("^--file=", "", args[1L]) else "tests/test_model.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
source(file.path(root, "R", "data.R")); source(file.path(root, "R", "model.R"))
source(file.path(root, "R", "optimiser.R")); source(file.path(root, "R", "evaluation.R"))

run_test <- function(name, fn) {
  started <- proc.time()[["elapsed"]]
  tryCatch({ fn(); cat(sprintf("[PASS] %-39s %.3fs\n", name,
                        proc.time()[["elapsed"]] - started)) },
           error = function(e) { cat(sprintf("[FAIL] %s: %s\n", name,
                                     conditionMessage(e))); quit(status = 1L) })
}
cat("min-char-transformer | model test suite\n\n")

run_test("UTF-8 character vocabulary round trip", function() {
  string <- "hello\nR!é"
  vocab <- build_vocab(string)
  indices <- encode_text(string, vocab)
  stopifnot(identical(decode_indices(indices, vocab), string),
            identical(decode_indices(encode_text("h", vocab), vocab), "h"),
            min(indices) == 1L,
            max(indices) == vocab$size)
})

run_test("contiguous train / validation / test", function() {
  text <- paste(rep("abcdefghij", 10L), collapse = "")
  split <- split_characters(text, 0.8, 0.1, minimum_size = 2L)
  stopifnot(identical(unname(split$sizes), c(80L, 10L, 10L)),
            identical(paste0(split$train, split$validation, split$test), text))
})

run_test("fixed passage plan and target alignment", function() {
  plan <- make_passage_plan(200L, 4L, 3L, 12L)
  stopifnot(nrow(plan) == 3L,
            all(diff(plan$start) > 12L), all(plan$start >= 1L))
  window <- sequence_window(seq_len(50L), 3L, 4L)
  stopifnot(identical(window$inputs, 3:6), identical(window$targets, 4:7))
})

run_test("parameter count and matrix dimensions", function() {
  set.seed(1)
  model <- initialise_model(65L, 32L, 32L, 64L)
  stopifnot(model_parameter_count(model) == 13729L,
            identical(dim(model$params$token_embedding), c(65L, 32L)),
            identical(dim(model$params$position_embedding), c(32L, 32L)),
            identical(dim(model$params$Wq), c(32L, 32L)),
            identical(dim(model$params$W1), c(32L, 64L)),
            identical(dim(model$params$Wout), c(32L, 65L)))
})

run_test("stable row-wise softmax and loss", function() {
  p <- row_softmax(matrix(c(1000, 1001, 999, -Inf, 1000, -Inf),
                          nrow = 2L, byrow = TRUE))
  stopifnot(all(is.finite(p)), all(abs(rowSums(p) - 1) < 1e-12),
            p[2L, 1L] == 0, p[2L, 3L] == 0)
  outcome <- logits_cross_entropy(matrix(c(1000, 1001, 999), nrow = 1L), 2L)
  stopifnot(is.finite(outcome$loss), outcome$loss > 0)
})

run_test("forward pass and causal attention", function() {
  set.seed(2)
  model <- initialise_model(5L, 8L, 8L, 12L)
  result <- forward_transformer(model, c(1L, 2L, 1L, 3L))
  stopifnot(identical(dim(result$logits), c(4L, 5L)),
            identical(dim(result$attention), c(4L, 4L)),
            all(result$attention[upper.tri(result$attention)] == 0),
            all(abs(rowSums(result$attention) - 1) < 1e-12))
})

run_test("future characters never affect prefix", function() {
  set.seed(3)
  model <- initialise_model(5L, 8L, 8L, 12L)
  first <- forward_transformer(model, c(1L, 2L, 3L, 4L), FALSE)$logits
  second <- forward_transformer(model, c(1L, 2L, 5L, 5L), FALSE)$logits
  stopifnot(isTRUE(all.equal(first[1:2, ], second[1:2, ], tolerance = 1e-12)))
})

run_test("forward loss and gradient dimensions", function() {
  set.seed(4)
  model <- initialise_model(5L, 6L, 6L, 8L)
  result <- loss_and_gradients(model, c(1L, 2L, 1L), c(2L, 1L, 3L))
  stopifnot(is.finite(result$loss),
            identical(names(result$grads), names(model$params)))
  for (name in names(model$params)) {
    stopifnot(identical(dim(result$grads[[name]]), dim(model$params[[name]])),
              all(is.finite(result$grads[[name]])))
  }
})

run_test("Adam changes model and can save/load", function() {
  set.seed(5)
  model <- initialise_model(5L, 6L, 6L, 8L)
  grads <- loss_and_gradients(model, c(1L, 2L, 1L), c(2L, 1L, 3L))$grads
  update <- adam_update(model, clip_gradients(grads, 1)$grads,
                        new_adam_state(model), 0.001)
  stopifnot(update$state$step == 1L,
            !identical(model$params$Wout, update$model$params$Wout))
  file <- tempfile(fileext = ".rds")
  on.exit(unlink(file), add = TRUE)
  save_model(update$model, file)
  stopifnot(identical(update$model, load_model(file)))
})

run_test("autoregressive generation returns valid indices", function() {
  set.seed(6)
  model <- initialise_model(5L, 6L, 6L, 8L)
  generated <- sample_indices(model, c(1L, 2L), 20L)
  stopifnot(length(generated) == 20L, all(generated >= 1L & generated <= 5L))
})
cat("\nAll model tests passed.\n")
