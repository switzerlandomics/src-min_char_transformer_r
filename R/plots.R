# Publication and browser-monitor plots. Only this module needs ggplot2.

require_plots <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE) ||
      utils::packageVersion("ggplot2") < "3.4.0") {
    stop("Plotting requires ggplot2 >= 3.4; install.packages('ggplot2').")
  }
}

read_experiment_metrics <- function(directory) {
  path <- file.path(directory, "metrics.csv")
  if (!file.exists(path)) stop("Missing metrics.csv: ", path)
  metrics <- utils::read.csv(path, stringsAsFactors = FALSE)
  required <- c("iteration", "train_loss_nats_per_char", "validation_loss_nats_per_char")
  if (!all(required %in% names(metrics))) stop("metrics.csv is missing required columns.")
  metrics <- metrics[order(metrics$iteration), , drop = FALSE]
  metrics[!duplicated(metrics$iteration, fromLast = TRUE), , drop = FALSE]
}

best_validation_row <- function(metrics) {
  valid <- which(is.finite(metrics$validation_loss_nats_per_char))
  if (!length(valid)) return(NULL)
  metrics[valid[which.min(metrics$validation_loss_nats_per_char[valid])], , drop = FALSE]
}

plot_theme <- function(base_size = 13) {
  ggplot2::theme_bw(base_size = base_size) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(colour = "#E7EBEE", linewidth = 0.3),
      plot.title.position = "plot", plot.title = ggplot2::element_text(face = "bold"),
      plot.subtitle = ggplot2::element_text(colour = "#4F606B"),
      plot.caption = ggplot2::element_text(hjust = 0, size = 10, colour = "#53646F"),
      legend.position = "top", legend.justification = "left",
      legend.title = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(12, 14, 12, 12))
}

plot_learning_curves <- function(metrics, metadata, mobile = FALSE) {
  require_plots()
  train <- metrics[is.finite(metrics$train_loss_nats_per_char), , drop = FALSE]
  valid <- metrics[is.finite(metrics$validation_loss_nats_per_char), , drop = FALSE]
  if (!nrow(valid)) stop("No validation measurements are available.")
  data <- rbind(data.frame(iteration = train$iteration, loss = train$train_loss_nats_per_char,
                           series = rep("Training (smoothed)", nrow(train))),
                data.frame(iteration = valid$iteration, loss = valid$validation_loss_nats_per_char,
                           series = rep("Validation (held out)", nrow(valid))))
  data$series <- factor(data$series,
                        levels = c("Training (smoothed)", "Validation (held out)"))
  best <- best_validation_row(metrics)
  p <- ggplot2::ggplot(data, ggplot2::aes(x = iteration, y = loss, colour = series)) +
    ggplot2::scale_colour_manual(values = c("Training (smoothed)" = "#687984",
                                              "Validation (held out)" = "#087A92")) +
    ggplot2::labs(title = "Training and validation",
      subtitle = if (mobile) sprintf("Best validation: %.3f nats/char", best$validation_loss_nats_per_char)
                 else sprintf("%s updates  |  best validation %.3f at %s updates",
                              format(tail(metrics$iteration, 1L), big.mark = ","),
                              best$validation_loss_nats_per_char,
                              format(best$iteration, big.mark = ",")),
      x = "Parameter updates", y = if (mobile) "Loss (nats/char)" else "Cross-entropy loss (nats/char)",
      caption = "Training: smoothed window loss. Validation: fixed held-out passages. Lower is better.") +
    ggplot2::scale_x_continuous(labels = function(x) format(x, big.mark = ",", trim = TRUE)) +
    plot_theme(if (mobile) 14 else 13)
  if (nrow(data) > 1L) p <- p + ggplot2::geom_line(linewidth = 0.85, na.rm = TRUE)
  p <- p + ggplot2::geom_point(data = valid, ggplot2::aes(x = iteration,
    y = validation_loss_nats_per_char), inherit.aes = FALSE,
    colour = "#087A92", size = if (mobile) 1.5 else 1.2)
  for (baseline in c("uniform_baseline_nats_per_char", "bigram_baseline_nats_per_char")) {
    value <- metadata[[baseline]]
    if (length(value) == 1L && is.finite(value)) {
      p <- p + ggplot2::geom_hline(yintercept = value,
        linetype = if (baseline == "uniform_baseline_nats_per_char") "dashed" else "dotted",
        colour = "#B5BDC3", linewidth = 0.5)
    }
  }
  p + ggplot2::geom_point(data = data.frame(iteration = best$iteration,
       loss = best$validation_loss_nats_per_char), ggplot2::aes(x = iteration, y = loss),
       inherit.aes = FALSE, colour = "#B54E32", size = 2.7)
}

plot_validation_passages <- function(passage_metrics) {
  require_plots()
  if (!nrow(passage_metrics)) stop("No passage metrics to plot.")
  ggplot2::ggplot(passage_metrics, ggplot2::aes(x = iteration,
       y = loss_nats_per_char, group = factor(passage_id))) +
    ggplot2::geom_line(colour = "#93B5C2", linewidth = 0.6, na.rm = TRUE) +
    ggplot2::geom_point(colour = "#087A92", size = 1.1, na.rm = TRUE) +
    ggplot2::labs(title = "Held-out passages", subtitle = "Each line is one fixed text passage",
                  x = "Parameter updates", y = "Loss (nats/character)",
                  caption = "Passages use the same fixed context-window protocol at each check.") +
    plot_theme()
}

plot_attention_matrix <- function(attention, characters, title = "Causal self-attention") {
  require_plots()
  if (!is.matrix(attention) || nrow(attention) != ncol(attention) ||
      nrow(attention) != length(characters)) stop("Attention and character dimensions differ.")
  n <- nrow(attention)
  data <- expand.grid(query_position = seq_len(n), key_position = seq_len(n))
  data$weight <- attention[cbind(data$query_position, data$key_position)]
  data$weight[data$key_position > data$query_position] <- NA_real_
  character_labels <- ifelse(characters == " ", "space", ifelse(characters == "\n", "\\n", characters))
  ggplot2::ggplot(data, ggplot2::aes(x = key_position, y = query_position, fill = weight)) +
    ggplot2::geom_tile(colour = "#E4E9EC", linewidth = 0.35) +
    ggplot2::scale_fill_gradient(low = "#F1F8FA", high = "#087A92", na.value = "#F4F5F6",
                                 limits = c(0, 1), name = "Weight") +
    ggplot2::scale_x_continuous(breaks = seq_len(n), labels = character_labels,
                                expand = ggplot2::expansion(add = 0.5)) +
    ggplot2::scale_y_reverse(breaks = seq_len(n), labels = character_labels,
                             expand = ggplot2::expansion(add = 0.5)) +
    ggplot2::coord_fixed() +
    ggplot2::labs(title = title, subtitle = "Rows attend only to the current and earlier positions",
      x = "Attended character (key)", y = "Predicting position (query)",
      caption = "Shaded cells show model weights; grey cells are causally masked.") + plot_theme(12) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}

# Plot files are generated from saved experiment artifacts, not fabricated curves.
plot_experiment <- function(directory) {
  require_plots()
  metrics <- read_experiment_metrics(directory)
  metadata_path <- file.path(directory, "metadata.rds")
  metadata <- if (file.exists(metadata_path)) readRDS(metadata_path) else list()
  desktop <- file.path(directory, "training.png")
  mobile <- file.path(directory, "training_mobile.png")
  ggplot2::ggsave(desktop, plot_learning_curves(metrics, metadata), width = 10,
                   height = 6, dpi = 135, bg = "white")
  ggplot2::ggsave(mobile, plot_learning_curves(metrics, metadata, mobile = TRUE),
                   width = 5.5, height = 5.2, dpi = 135, bg = "white")
  passage_path <- file.path(directory, "validation_passages.csv")
  if (file.exists(passage_path)) {
    passages <- utils::read.csv(passage_path)
    if (nrow(passages)) ggplot2::ggsave(file.path(directory, "validation_detail.png"),
       plot_validation_passages(passages), width = 10, height = 6, dpi = 135, bg = "white")
  }
  initial_path <- file.path(directory, "attention_initial.rds")
  if (file.exists(initial_path)) {
    initial <- readRDS(initial_path)
    ggplot2::ggsave(file.path(directory, "attention_initial.png"),
      plot_attention_matrix(initial$attention, initial$characters,
                            title = "Attention before training"),
      width = 7.5, height = 6.8, dpi = 150, bg = "white")
  }
  snapshot_path <- file.path(directory, "attention_snapshot.rds")
  if (file.exists(snapshot_path)) {
    snapshot <- readRDS(snapshot_path)
    ggplot2::ggsave(file.path(directory, "attention.png"),
      plot_attention_matrix(snapshot$attention, snapshot$characters),
      width = 7.5, height = 6.8, dpi = 150, bg = "white")
  }
  html <- '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="10"><title>Transformer training</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;color:#23323c}img{display:block;max-width:100%;height:auto;margin:20px auto 28px}h1{font-size:1.4rem}small{color:#52646f}@media(max-width:600px){body{padding:8px}h1{font-size:1.15rem}}</style></head><body><h1>Transformer training</h1><small>Refreshes every 10 seconds. Training runs independently of this page.</small><picture><source media="(max-width:600px)" srcset="training_mobile.png"><img src="training.png" alt="Training and validation loss"></picture><img src="validation_detail.png" alt="Loss by held-out passage" onerror="this.style.display=\'none\'"><img src="attention.png" alt="Learned causal attention" onerror="this.style.display=\'none\'"></body></html>'
  writeLines(html, file.path(directory, "training.html"), useBytes = TRUE)
  list(files = c(desktop, mobile, file.path(directory, "training.html")),
       best = best_validation_row(metrics))
}
