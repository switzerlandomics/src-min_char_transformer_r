# Evaluation only: fixed held-out windows, loss and simple train-only baselines.
# No model parameters or optimiser state are modified by these functions.

evaluate_passages <- function(model, indices, plan, context_length) {
  rows <- vector("list", nrow(plan))
  for (passage in seq_len(nrow(plan))) {
    first <- plan$start[passage]
    steps <- seq.int(0L, plan$transitions[passage] - context_length,
                     by = context_length)
    window_losses <- vapply(steps, function(offset) {
      window <- sequence_window(indices, first + offset, context_length)
      sequence_loss(model, window$inputs, window$targets)
    }, numeric(1L))
    rows[[passage]] <- data.frame(passage_id = plan$passage_id[passage],
       start = first, transitions = plan$transitions[passage],
       loss_nats_per_char = mean(window_losses))
  }
  measurements <- do.call(rbind, rows)
  list(loss = weighted.mean(measurements$loss_nats_per_char,
                            measurements$transitions), passages = measurements)
}

# Return exact (previous, next) index pairs used by the selected evaluation plan.
passage_transitions <- function(indices, plan) {
  positions <- unlist(lapply(seq_len(nrow(plan)), function(i) {
    seq.int(plan$start[i], length.out = plan$transitions[i])
  }), use.names = FALSE)
  list(previous = indices[positions], next = indices[positions + 1L])
}

bigram_baseline <- function(training_indices, heldout_indices, plan, vocab_size) {
  positions <- seq_len(length(training_indices) - 1L)
  previous <- training_indices[positions]
  following <- training_indices[positions + 1L]
  counts <- matrix(tabulate((previous - 1L) * vocab_size + following,
                            nbins = vocab_size^2), nrow = vocab_size, byrow = TRUE)
  totals <- rowSums(counts)
  pairs <- passage_transitions(heldout_indices, plan)
  -mean(log((counts[cbind(pairs$previous, pairs$next)] + 1) /
              (totals[pairs$previous] + vocab_size)))
}
