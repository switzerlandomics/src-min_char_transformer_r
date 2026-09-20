# Optimiser and clipping, independent of any particular model architecture.
# Adam is implemented directly; no external optimizer or autodiff is used.

new_adam_state <- function(model) {
  zero <- function(parameter) matrix(0, nrow(parameter), ncol(parameter))
  list(step = 0L, first = lapply(model$params, zero),
       second = lapply(model$params, zero))
}

gradient_global_norm <- function(grads) {
  sqrt(sum(vapply(grads, function(x) sum(x^2), numeric(1L))))
}

clip_gradients <- function(grads, max_norm = 1) {
  if (!is.finite(max_norm) || max_norm <= 0) stop("max_norm must be positive.")
  norm <- gradient_global_norm(grads)
  if (!is.finite(norm)) stop("Non-finite gradients; refusing to update the model.")
  if (norm > max_norm) grads <- lapply(grads, function(x) x * (max_norm / norm))
  list(grads = grads, norm = norm, clipped = norm > max_norm)
}

adam_update <- function(model, grads, state, learning_rate = 0.001,
                        beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
  if (!is.finite(learning_rate) || learning_rate <= 0 ||
      !(beta1 > 0 && beta1 < 1 && beta2 > 0 && beta2 < 1) ||
      !is.finite(epsilon) || epsilon <= 0) stop("Invalid Adam configuration.")
  if (!identical(names(model$params), names(grads)) ||
      !identical(names(grads), names(state$first)) ||
      !identical(names(grads), names(state$second))) {
    stop("Gradient and optimiser parameter names must match the model.")
  }
  state$step <- state$step + 1L
  correction1 <- 1 - beta1^state$step
  correction2 <- 1 - beta2^state$step
  for (name in names(model$params)) {
    parameter <- model$params[[name]]
    gradient <- grads[[name]]
    if (!identical(dim(parameter), dim(gradient))) stop("Gradient dimension mismatch: ", name)
    state$first[[name]] <- beta1 * state$first[[name]] + (1 - beta1) * gradient
    state$second[[name]] <- beta2 * state$second[[name]] + (1 - beta2) * gradient^2
    first_hat <- state$first[[name]] / correction1
    second_hat <- state$second[[name]] / correction2
    model$params[[name]] <- parameter - learning_rate * first_hat / (sqrt(second_hat) + epsilon)
  }
  list(model = model, state = state)
}
