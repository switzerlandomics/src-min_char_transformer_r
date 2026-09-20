#!/usr/bin/env Rscript
# Central-difference checks of EVERY trainable parameter on a tiny model.
args <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script <- if (length(args)) sub("^--file=", "", args[1L]) else "tests/test_gradients.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
source(file.path(root, "R", "model.R"))
cat("min-char-transformer | finite-difference gradient checks\n")
set.seed(17)
model <- initialise_model(4L, 4L, 4L, 5L, weight_sd = 0.06)
inputs <- c(1L, 2L, 1L, 3L)
targets <- c(2L, 1L, 3L, 4L)
started <- proc.time()[["elapsed"]]
checks <- gradient_check(model, inputs, targets, epsilon = 1e-5,
                         max_checks_per_parameter = Inf)
for (name in unique(checks$parameter)) {
  rows <- checks[checks$parameter == name, , drop = FALSE]
  cat(sprintf("[CHECK] %-20s %4d elements | max abs %.3e | max relative %.3e\n",
     name, nrow(rows), max(rows$absolute_error), max(rows$relative_error)))
}
# Relative errors for tiny derivatives can be dominated by floating-point noise;
# an independent absolute tolerance handles those without excusing large errors.
failed <- checks[!is.finite(checks$absolute_error) |
                 (checks$absolute_error > 2e-6 & checks$relative_error > 2e-4), , drop = FALSE]
if (nrow(failed)) {
  print(utils::head(failed[order(-failed$absolute_error), ], 10L))
  stop(sprintf("Gradient check failed for %d parameter elements.", nrow(failed)))
}
cat(sprintf("[PASS] %d elements across %d matrices | %.2fs\n",
  nrow(checks), length(unique(checks$parameter)),
  proc.time()[["elapsed"]] - started))
