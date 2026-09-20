# One-block, one-head, causal character Transformer with manual differentiation.
# Rows are sequence positions; columns are features. All indices are one-based.
# All trainable arrays live in model$params; model$config contains dimensions.

assert_positive_integer <- function(x, name) {
  if (length(x) != 1L || is.na(x) || !is.finite(x) || x < 1 || x != as.integer(x)) {
    stop(name, " must be a positive integer.")
  }
  invisible(TRUE)
}

initialise_model <- function(vocab_size, context_length = 32L,
                             embedding_size = 32L, feedforward_size = 64L,
                             weight_sd = 0.02, norm_epsilon = 1e-5) {
  for (name in c("vocab_size", "context_length", "embedding_size", "feedforward_size")) {
    assert_positive_integer(get(name), name)
  }
  if (!is.finite(weight_sd) || weight_sd <= 0 ||
      !is.finite(norm_epsilon) || norm_epsilon <= 0) stop("Invalid numerical configuration.")
  if (embedding_size < 2L) stop("embedding_size must be at least two for layer normalisation.")
  v <- as.integer(vocab_size); context <- as.integer(context_length)
  d <- as.integer(embedding_size); f <- as.integer(feedforward_size)
  weights <- function(rows, cols) matrix(rnorm(rows * cols, sd = weight_sd), rows, cols)
  zeros <- function(width) matrix(0, 1L, width)
  ones <- function(width) matrix(1, 1L, width)
  params <- list(
    token_embedding = weights(v, d), position_embedding = weights(context, d),
    ln1_gain = ones(d), ln1_bias = zeros(d),
    Wq = weights(d, d), Wk = weights(d, d), Wv = weights(d, d), Wo = weights(d, d),
    ln2_gain = ones(d), ln2_bias = zeros(d),
    W1 = weights(d, f), b1 = zeros(f), W2 = weights(f, d), b2 = zeros(d),
    ln3_gain = ones(d), ln3_bias = zeros(d),
    Wout = weights(d, v), bout = zeros(v)
  )
  list(config = list(vocab_size = v, context_length = context, embedding_size = d,
                     feedforward_size = f, norm_epsilon = norm_epsilon),
       params = params)
}

model_parameter_count <- function(model) sum(vapply(model$params, length, integer(1L)))

assert_model_finite <- function(model) {
  if (!is.list(model) || !is.list(model$params) || !length(model$params) ||
      !all(vapply(model$params, function(x) is.matrix(x) && all(is.finite(x)), logical(1L)))) {
    stop("Model has missing or non-finite parameter matrices.")
  }
  invisible(TRUE)
}

# Unlike implicit R vector recycling, this addition works for any batch length.
add_row_bias <- function(x, bias) {
  stopifnot(is.matrix(x), is.matrix(bias), nrow(bias) == 1L,
            ncol(x) == ncol(bias))
  sweep(x, 2L, as.numeric(bias), "+")
}

row_softmax <- function(logits) {
  if (!is.matrix(logits) || !is.numeric(logits) || anyNA(logits) ||
      !all(is.finite(logits) | logits == -Inf)) {
    stop("Softmax requires a numeric matrix with finite values or masked -Inf.")
  }
  shifted <- sweep(logits, 1L, apply(logits, 1L, max), "-")
  if (anyNA(shifted) || any(!is.finite(shifted) & shifted != -Inf)) {
    stop("Every softmax row needs at least one unmasked finite value.")
  }
  numerators <- exp(shifted)
  sweep(numerators, 1L, rowSums(numerators), "/")
}

softmax_backward <- function(probabilities, upstream) {
  stopifnot(identical(dim(probabilities), dim(upstream)))
  probabilities * sweep(upstream, 1L,
                        rowSums(probabilities * upstream), "-")
}

layer_norm_forward <- function(x, gain, bias, epsilon = 1e-5) {
  stopifnot(is.matrix(x), ncol(x) > 1L, ncol(gain) == ncol(x),
            nrow(gain) == 1L, identical(dim(gain), dim(bias)))
  centred <- sweep(x, 1L, rowMeans(x), "-")
  inv_sd <- 1 / sqrt(rowMeans(centred^2) + epsilon)
  normalized <- sweep(centred, 1L, inv_sd, "*")
  output <- add_row_bias(sweep(normalized, 2L, as.numeric(gain), "*"), bias)
  list(value = output, cache = list(normalized = normalized, inv_sd = inv_sd,
                                   gain = gain))
}

layer_norm_backward <- function(upstream, cache) {
  normalized <- cache$normalized
  stopifnot(identical(dim(upstream), dim(normalized)))
  d_gain <- matrix(colSums(upstream * normalized), 1L)
  d_bias <- matrix(colSums(upstream), 1L)
  dz <- sweep(upstream, 2L, as.numeric(cache$gain), "*")
  centred_gradient <- sweep(dz, 1L, rowMeans(dz), "-")
  centred_gradient <- centred_gradient - sweep(normalized, 1L,
                                               rowMeans(dz * normalized), "*")
  dx <- sweep(centred_gradient, 1L, cache$inv_sd, "*")
  list(dx = dx, d_gain = d_gain, d_bias = d_bias)
}

# A row at position i may only use positions 1,...,i.
causal_attention_forward <- function(x, Wq, Wk, Wv, Wo) {
  q <- x %*% Wq; k <- x %*% Wk; v <- x %*% Wv
  scale <- sqrt(ncol(q))
  scores <- q %*% t(k) / scale
  scores[upper.tri(scores)] <- -Inf
  attention <- row_softmax(scores)
  weighted_values <- attention %*% v
  output <- weighted_values %*% Wo
  list(value = output, attention = attention,
       cache = list(x = x, q = q, k = k, v = v, attention = attention,
                    weighted_values = weighted_values, scale = scale))
}

causal_attention_backward <- function(upstream, cache, Wq, Wk, Wv, Wo) {
  dz <- upstream %*% t(Wo)
  dWo <- t(cache$weighted_values) %*% upstream
  da <- dz %*% t(cache$v)
  dv <- t(cache$attention) %*% dz
  ds <- softmax_backward(cache$attention, da)
  ds[upper.tri(ds)] <- 0
  dq <- ds %*% cache$k / cache$scale
  dk <- t(ds) %*% cache$q / cache$scale
  list(dx = dq %*% t(Wq) + dk %*% t(Wk) + dv %*% t(Wv),
       dWq = t(cache$x) %*% dq, dWk = t(cache$x) %*% dk,
       dWv = t(cache$x) %*% dv, dWo = dWo)
}

validate_model_inputs <- function(model, inputs) {
  conf <- model$config
  if (!is.numeric(inputs) || !length(inputs) || anyNA(inputs) ||
      !all(is.finite(inputs)) ||
      length(inputs) > conf$context_length ||
      any(inputs != as.integer(inputs)) ||
      any(inputs < 1L | inputs > conf$vocab_size)) {
    stop("Input must contain valid one-based indices within the model context.")
  }
  invisible(TRUE)
}

# Pre-norm block: x + Attention(LN1(x)); r + FFN(LN2(r)); LN3; logits.
forward_transformer <- function(model, inputs, retain_cache = TRUE) {
  validate_model_inputs(model, inputs)
  p <- model$params; conf <- model$config
  n <- length(inputs)
  x <- p$token_embedding[inputs, , drop = FALSE] +
       p$position_embedding[seq_len(n), , drop = FALSE]
  ln1 <- layer_norm_forward(x, p$ln1_gain, p$ln1_bias, conf$norm_epsilon)
  att <- causal_attention_forward(ln1$value, p$Wq, p$Wk, p$Wv, p$Wo)
  residual1 <- x + att$value
  ln2 <- layer_norm_forward(residual1, p$ln2_gain, p$ln2_bias, conf$norm_epsilon)
  ff_pre <- add_row_bias(ln2$value %*% p$W1, p$b1)
  ff_hidden <- pmax(ff_pre, 0)
  ff_output <- add_row_bias(ff_hidden %*% p$W2, p$b2)
  residual2 <- residual1 + ff_output
  ln3 <- layer_norm_forward(residual2, p$ln3_gain, p$ln3_bias, conf$norm_epsilon)
  logits <- add_row_bias(ln3$value %*% p$Wout, p$bout)
  result <- list(logits = logits, attention = att$attention)
  if (retain_cache) {
    result$cache <- list(inputs = as.integer(inputs), ln1 = ln1,
                         attention = att$cache, ln2 = ln2, ff_pre = ff_pre,
                         ff_hidden = ff_hidden, ln3 = ln3)
  }
  result
}

# Average negative log likelihood in nats per target position.
logits_cross_entropy <- function(logits, targets) {
  if (!is.matrix(logits) || length(targets) != nrow(logits) ||
      anyNA(targets) || any(targets < 1L | targets > ncol(logits)) ||
      any(targets != as.integer(targets)) || !all(is.finite(logits))) {
    stop("Invalid next-character logits or targets.")
  }
  shifted <- sweep(logits, 1L, apply(logits, 1L, max), "-")
  log_denominator <- log(rowSums(exp(shifted)))
  loss <- mean(log_denominator - shifted[cbind(seq_len(nrow(logits)), targets)])
  list(loss = as.numeric(loss), probabilities = sweep(exp(shifted), 1L,
                                                   exp(log_denominator), "/"))
}

sequence_loss <- function(model, inputs, targets) {
  logits_cross_entropy(forward_transformer(model, inputs, FALSE)$logits, targets)$loss
}

loss_and_gradients <- function(model, inputs, targets) {
  p <- model$params
  forward <- forward_transformer(model, inputs, TRUE)
  loss <- logits_cross_entropy(forward$logits, targets)
  cache <- forward$cache
  grads <- lapply(p, function(parameter) matrix(0, nrow(parameter), ncol(parameter)))
  n <- length(inputs)

  # Derivative of mean cross entropy with respect to output logits.
  dy <- loss$probabilities
  dy[cbind(seq_len(n), targets)] <- dy[cbind(seq_len(n), targets)] - 1
  dy <- dy / n
  grads$Wout <- t(cache$ln3$value) %*% dy
  grads$bout <- matrix(colSums(dy), 1L)
  ln3_back <- layer_norm_backward(dy %*% t(p$Wout), cache$ln3$cache)
  grads$ln3_gain <- ln3_back$d_gain; grads$ln3_bias <- ln3_back$d_bias

  # Second residual branch, feed-forward network and second layer norm.
  du <- ln3_back$dx
  grads$W2 <- t(cache$ff_hidden) %*% du
  grads$b2 <- matrix(colSums(du), 1L)
  dff_pre <- (du %*% t(p$W2)) * (cache$ff_pre > 0)
  grads$W1 <- t(cache$ln2$value) %*% dff_pre
  grads$b1 <- matrix(colSums(dff_pre), 1L)
  ln2_back <- layer_norm_backward(dff_pre %*% t(p$W1), cache$ln2$cache)
  grads$ln2_gain <- ln2_back$d_gain; grads$ln2_bias <- ln2_back$d_bias
  dr <- du + ln2_back$dx

  # First residual branch, causal attention and first layer norm.
  att_back <- causal_attention_backward(dr, cache$attention, p$Wq, p$Wk,
                                        p$Wv, p$Wo)
  grads$Wq <- att_back$dWq; grads$Wk <- att_back$dWk
  grads$Wv <- att_back$dWv; grads$Wo <- att_back$dWo
  ln1_back <- layer_norm_backward(att_back$dx, cache$ln1$cache)
  grads$ln1_gain <- ln1_back$d_gain; grads$ln1_bias <- ln1_back$d_bias
  dx <- dr + ln1_back$dx

  # The same character may occur at several positions; accumulate its gradient.
  for (position in seq_len(n)) {
    index <- inputs[position]
    grads$token_embedding[index, ] <- grads$token_embedding[index, ] + dx[position, ]
    grads$position_embedding[position, ] <-
      grads$position_embedding[position, ] + dx[position, ]
  }
  list(loss = loss$loss, grads = grads, attention = forward$attention)
}

# Generation recalculates only the last context_length characters. No recurrent
# state is carried between calls, and no future generated characters are read.
sample_indices <- function(model, prompt_indices, n, temperature = 1) {
  validate_model_inputs(model, tail(prompt_indices, model$config$context_length))
  if (length(n) != 1L || is.na(n) || n < 0L || n != as.integer(n) ||
      !is.finite(temperature) || temperature <= 0) stop("Invalid sampling options.")
  history <- as.integer(prompt_indices)
  generated <- integer(n)
  for (step in seq_len(n)) {
    context <- tail(history, model$config$context_length)
    logits <- forward_transformer(model, context, FALSE)$logits
    probabilities <- row_softmax(matrix(logits[nrow(logits), ] / temperature, nrow = 1L))
    index <- sample.int(model$config$vocab_size, 1L, prob = as.numeric(probabilities))
    generated[step] <- index
    history <- c(history, index)
  }
  generated
}

save_model <- function(model, path) {
  assert_model_finite(model)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(model, path)
  invisible(path)
}
load_model <- function(path) {
  model <- readRDS(path)
  assert_model_finite(model)
  model
}

# Central differences are intentionally expensive: use tiny test models only.
gradient_check <- function(model, inputs, targets, epsilon = 1e-5,
                           max_checks_per_parameter = Inf) {
  if (!is.finite(epsilon) || epsilon <= 0) stop("epsilon must be positive.")
  analytical <- loss_and_gradients(model, inputs, targets)$grads
  results <- vector("list", 0L)
  for (name in names(model$params)) {
    count <- length(model$params[[name]])
    indices <- if (is.finite(max_checks_per_parameter) &&
                   max_checks_per_parameter < count) {
      unique(as.integer(round(seq(1, count, length.out = max_checks_per_parameter))))
    } else seq_len(count)
    for (index in indices) {
      plus <- model; minus <- model
      plus$params[[name]][index] <- plus$params[[name]][index] + epsilon
      minus$params[[name]][index] <- minus$params[[name]][index] - epsilon
      numerical <- (sequence_loss(plus, inputs, targets) -
                    sequence_loss(minus, inputs, targets)) / (2 * epsilon)
      actual <- analytical[[name]][index]
      abs_error <- abs(actual - numerical)
      relative_error <- abs_error / max(1e-7, abs(actual) + abs(numerical))
      results[[length(results) + 1L]] <- data.frame(parameter = name,
        index = index, analytical = actual, numerical = numerical,
        absolute_error = abs_error, relative_error = relative_error)
    }
  }
  do.call(rbind, results)
}
