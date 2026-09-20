# Publication and browser-monitor plots. Only numerical figures need ggplot2.

# Model-derived generation evidence. These helpers do not require ggplot2.
# The fixed comparison seed makes the initial and selected-model samples directly
# comparable as demonstrations, without selecting the most readable random sample.
make_generation_sample <- function(model, vocab, prompt, iteration, validation_loss,
                                   n, seed, temperature = 1) {
  if (!is.character(prompt) || length(prompt) != 1L || is.na(prompt) ||
      !nzchar(prompt) || length(validation_loss) != 1L ||
      !is.finite(validation_loss) || length(temperature) != 1L ||
      !is.finite(temperature) || temperature <= 0) {
    stop("Invalid generation prompt, validation loss, or temperature.")
  }
  generated <- with_sampling_seed(seed, sample_indices(
    model, encode_text(prompt, vocab), n = n, temperature = temperature))
  list(iteration = as.integer(iteration),
       validation_loss_nats_per_char = as.numeric(validation_loss),
       prompt = prompt, generated_text = decode_indices(generated, vocab),
       n_generated = as.integer(n), seed = as.integer(seed),
       temperature = as.numeric(temperature))
}

# Before this feature existed, the initial sample was recorded only in samples.txt.
# Read the actual recorded text rather than inventing a replacement initial model.
read_initial_generation_sample <- function(directory, prompt, initial_loss,
                                           seed, temperature = 1) {
  sample_file <- file.path(directory, "samples.txt")
  if (!file.exists(sample_file)) return(NULL)
  size <- file.info(sample_file)$size
  if (!is.finite(size) || size < 1L) return(NULL)
  content <- rawToChar(readBin(sample_file, what = "raw", n = size))
  start <- regexpr("=== update 0 |", content, fixed = TRUE)[1L]
  if (start < 1L) return(NULL)
  remaining <- substring(content, start)
  header_end <- regexpr("\n", remaining, fixed = TRUE)[1L]
  if (header_end < 1L) return(NULL)
  header <- substr(remaining, 1L, header_end - 1L)
  if (!startsWith(header, "=== update 0 | validation ") ||
      !grepl(" | prompt ", header, fixed = TRUE) ||
      !endsWith(header, " | temperature 1 ===")) return(NULL)
  body <- substring(remaining, header_end + 1L)
  next_sample <- regexpr("\n=== update ", body, fixed = TRUE)[1L]
  if (next_sample > 0L) body <- substr(body, 1L, next_sample - 1L)
  body <- sub("\n$", "", body)
  prefix_size <- nchar(prompt, type = "chars")
  if (!startsWith(body, prompt)) return(NULL)
  generated_text <- substring(body, prefix_size + 1L)
  list(iteration = 0L, validation_loss_nats_per_char = as.numeric(initial_loss),
       prompt = prompt, generated_text = generated_text,
       n_generated = as.integer(nchar(generated_text, type = "chars")),
       seed = as.integer(seed), temperature = as.numeric(temperature))
}

# Migrate an earlier completed run using its real recorded initial sample and the
# genuine selected model. Returns NULL if the required evidence is unavailable.
restore_generation_comparison <- function(directory, vocab, prompt, config,
                                          initial_loss, best_iteration, best_loss) {
  fixed_seed <- as.integer(config$seed + 100000L)
  initial <- read_initial_generation_sample(directory, prompt, initial_loss, fixed_seed)
  selected_path <- file.path(directory, "best_model.rds")
  if (is.null(initial) || !file.exists(selected_path)) return(NULL)
  best <- make_generation_sample(load_model(selected_path), vocab, prompt,
    best_iteration, best_loss, n = initial$n_generated,
    seed = fixed_seed, temperature = initial$temperature)
  list(initial = initial, best = best)
}

# This content is inserted as text, not trusted markup, in a local HTML monitor.
html_escape <- function(value) {
  value <- gsub("&", "&amp;", value, fixed = TRUE)
  value <- gsub("<", "&lt;", value, fixed = TRUE)
  value <- gsub(">", "&gt;", value, fixed = TRUE)
  value <- gsub('"', "&quot;", value, fixed = TRUE)
  gsub("'", "&#39;", value, fixed = TRUE)
}

generation_comparison_html <- function(directory) {
  comparison_path <- file.path(directory, "generation_comparison.rds")
  if (!file.exists(comparison_path)) return("")
  result <- readRDS(comparison_path)
  samples <- list(result$initial, result$best)
  if (!all(vapply(samples, function(item) is.list(item) &&
       all(c("iteration", "validation_loss_nats_per_char", "prompt",
             "generated_text", "n_generated", "seed", "temperature") %in%
             names(item)), logical(1L)))) {
    stop("Invalid generation_comparison.rds.")
  }
  if (!identical(samples[[1L]]$prompt, samples[[2L]]$prompt) ||
      !identical(samples[[1L]]$seed, samples[[2L]]$seed) ||
      !identical(samples[[1L]]$n_generated, samples[[2L]]$n_generated) ||
      !identical(samples[[1L]]$temperature, samples[[2L]]$temperature)) {
    stop("Generation comparison must use an identical prompt and sampling setup.")
  }
  sample_card <- function(sample, title) {
    heading <- sprintf("Update %s | validation %.6f nats/char",
      format(sample$iteration, big.mark = ","),
      sample$validation_loss_nats_per_char)
    paste0('<article class="sample"><h3>', html_escape(title), '</h3><p class="sample-meta">',
      html_escape(heading), '</p><pre class="generated">',
      html_escape(sample$generated_text), '</pre></article>')
  }
  paste0('<section aria-labelledby="generation-heading"><h2 id="generation-heading">',
    'Text generation: before and after training</h2>',
    '<p>The selected checkpoint has the lowest measured validation loss, ',
    'not the most appealing sample. Both outputs use the same prompt and random ',
    'sampling settings; the text below is the unedited model output.</p>',
    '<p class="sample-meta">Prompt: <code>', html_escape(samples[[1L]]$prompt),
    '</code> | ', samples[[1L]]$n_generated,
    ' generated characters | temperature ',
    format(samples[[1L]]$temperature, trim = TRUE), ' | sample seed ',
    samples[[1L]]$seed, '</p>',
    '<div class="sample-grid">',
    sample_card(samples[[1L]], "Before training"),
    sample_card(samples[[2L]], "Best-validation checkpoint"),
    '</div><p class="sample-meta">See <a href="samples.txt">all periodic samples</a>',
    ' and <a href="generation_comparison.txt">the complete comparison text</a>',
    ' (<a href="generation_comparison.rds">structured data</a>).</p></section>')
}

# The companion plain-text file is useful for copying genuine results into a blog.
# The structured RDS remains the exact source of truth for prompt and output text.
write_generation_comparison_text <- function(directory) {
  record_path <- file.path(directory, "generation_comparison.rds")
  if (!file.exists(record_path)) return(invisible(NULL))
  comparison <- readRDS(record_path)
  format_sample <- function(sample, label) {
    paste0("=== ", label, " | update ", sample$iteration,
      " | validation ", sprintf("%.6f", sample$validation_loss_nats_per_char),
      " nats/char | prompt ", encodeString(sample$prompt, quote = '"'),
      " | temperature ", sample$temperature, " | sampling seed ",
      sample$seed, " ===\n", sample$prompt, sample$generated_text)
  }
  content <- paste0("Text generation: before and after training\n",
    "Both samples use the same prompt, sampling seed, temperature and length.\n",
    "The checkpoint is selected by validation loss, not by sample readability.\n\n",
    format_sample(comparison$initial, "Before training"), "\n\n",
    format_sample(comparison$best, "Best-validation checkpoint"), "\n")
  destination <- file.path(directory, "generation_comparison.txt")
  writeChar(content, destination, eos = NULL, useBytes = TRUE)
  invisible(destination)
}

# Rebuild only the monitor HTML after FINISHED appears, so its automatic refresh
# no longer interrupts reading/copying the generated text from a completed run.
write_monitor_html <- function(directory) {
  active <- !file.exists(file.path(directory, "FINISHED"))
  refresh <- if (active) '<meta http-equiv="refresh" content="10">' else ''
  comparison <- generation_comparison_html(directory)
  write_generation_comparison_text(directory)
  image <- function(filename, alt) {
    if (!file.exists(file.path(directory, filename))) return("")
    paste0('<img src="', filename, '" alt="', html_escape(alt), '">')
  }
  html <- paste0('<!doctype html><html lang="en"><head><meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width,initial-scale=1">', refresh,
    '<title>Transformer training</title><style>',
    'body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:auto;',
    'padding:16px;color:#23323c}h1{font-size:1.4rem}h2{font-size:1.2rem;',
    'margin-top:32px}h3{font-size:1rem;margin:0}small,.sample-meta{color:#52646f}',
    'img{display:block;max-width:100%;height:auto;margin:20px auto 28px}',
    '.sample-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}',
    '.sample{min-width:0;border:1px solid #dbe4e9;border-radius:9px;padding:16px}',
    '.sample-meta{font-size:.88rem;margin:8px 0 12px}',
    '.generated{font:13px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;',
    'white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;',
    'max-width:100%;margin:0;tab-size:4}',
    '@media(max-width:700px){.sample-grid{grid-template-columns:minmax(0,1fr)}',
    'body{padding:12px}h1{font-size:1.2rem}.sample{padding:12px}}',
    '</style></head><body><h1>Transformer training</h1>',
    '<small>', if (active) 'Refreshes every 10 seconds. Training runs independently of this page.' else 'Completed experiment. Results are saved locally.', '</small>',
    if (file.exists(file.path(directory, "training.png"))) {
      if (file.exists(file.path(directory, "training_mobile.png")))
        paste0('<picture><source media="(max-width:600px)" srcset="training_mobile.png">',
               '<img src="training.png" alt="Training and validation loss"></picture>')
      else '<img src="training.png" alt="Training and validation loss">'
    } else '<p class="sample-meta">Training curves are not available for this run. Text samples and numerical metrics are saved independently of plotting.</p>',

    comparison,
    image("validation_detail.png", "Loss by held-out passage"),
    image("attention.png", "Learned causal attention"),
    '</body></html>')
  writeLines(html, file.path(directory, "training.html"), useBytes = TRUE)
  invisible(file.path(directory, "training.html"))
}

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
  write_monitor_html(directory)
  list(files = c(desktop, mobile, file.path(directory, "training.html")),
       best = best_validation_row(metrics))
}
