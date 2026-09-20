
# Figure 5: measured attention from the saved best-validation Transformer.
# From anywhere: Rscript experiments/plot_blog_attention.R
# Optional prompt: Rscript experiments/plot_blog_attention.R "The king"
# Requires ggplot2 and svglite; the model itself uses base R only.

library(ggplot2)
library(svglite)

# Resolve the repository root from this script, not the working directory.
# file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
# if (!length(file_arg)) stop("Run this script using Rscript.")
# script_path <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
root <- "../"

run_dir <- file.path(root, "output", "20260920_190535_seed666")
output_svg <- file.path(root, "docs", "figures", "05_learned_attention_map.svg")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 1L) stop('Supply at most one prompt, e.g. "The king".')
prompt <- if (length(args)) args[[1L]] else "The king"

# Reuse exactly the project's encoding and forward-pass implementation.
source(file.path(root, "R", "data.R"))
source(file.path(root, "R", "model.R"))
model <- load_model(file.path(run_dir, "best_model.rds"))
vocab <- load_vocab(file.path(run_dir, "vocab.rds"))
metadata <- readRDS(file.path(run_dir, "metadata.rds"))
metrics <- read.csv(file.path(run_dir, "metrics.csv"))

# Record the selected checkpoint, rather than using the latest attention snapshot.
valid <- which(is.finite(metrics$validation_loss_nats_per_char))
if (!length(valid)) stop("No finite validation losses in metrics.csv.")
best <- metrics[valid[which.min(metrics$validation_loss_nats_per_char[valid])], ]
if (!isTRUE(all.equal(as.numeric(metadata$best_iteration),
                      as.numeric(best$iteration)))) {
  stop("Metadata and metrics disagree on the best checkpoint; inspect the run.")
}
if (!isTRUE(all.equal(as.numeric(metadata$best_validation_nats_per_char),
                      as.numeric(best$validation_loss_nats_per_char),
                      tolerance = 1e-7))) {
  stop("Metadata and metrics disagree on the best validation loss.")
}

indices <- encode_text(prompt, vocab)
if (length(indices) < 2L || length(indices) > model$config$context_length) {
  stop("Prompt must contain 2 to ", model$config$context_length, " characters.")
}
characters <- vocab$ix_to_char[indices]
labels <- ifelse(characters == " ", "␠",
                 ifelse(characters == "\n", "↵",
                        ifelse(characters == "\t", "⇥", characters)))

# Each matrix row is a query position and each column is a key position.
attention <- forward_transformer(model, indices, retain_cache = FALSE)$attention
n <- length(indices)
if (!is.matrix(attention) || !identical(dim(attention), c(n, n)) ||
    any(!is.finite(attention)) || any(attention < -1e-12) ||
    any(abs(rowSums(attention) - 1) > 1e-7) ||
    any(abs(attention[upper.tri(attention)]) > 1e-12)) {
  stop("Attention matrix is not a valid causal, row-normalised matrix.")
}

tiles <- expand.grid(query = seq_len(n), key = seq_len(n))
tiles$permitted <- tiles$key <= tiles$query
tiles$weight <- attention[cbind(tiles$query, tiles$key)]
measured <- tiles[tiles$permitted, ]

p <- ggplot() +
  # The masked future positions have no attention weights.
  geom_tile(data = tiles, aes(x = key, y = query),
            width = 1, height = 1, fill = "#F3F2F2",
            colour = "#FFFFFF", linewidth = 0.8) +
  # Only permitted positions receive colours derived from measured weights.
  geom_tile(data = measured, aes(x = key, y = query, fill = weight),
            width = 1, height = 1, colour = "#FFFFFF", linewidth = 0.8) +
  scale_fill_gradientn(
    colours = c("#FDEBED", "#F88379", "#E5262F"),
    limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1),
    name = "Attention weight",
    guide = guide_colorbar(
      barwidth = grid::unit(6, "mm"),
      barheight = grid::unit(55, "mm"),
      frame.colour = "#000000", ticks.colour = "#000000"
    )
  ) +
  scale_x_continuous(
    position = "top", breaks = seq_len(n), labels = labels,
    limits = c(0.5, n + 0.5), expand = expansion(mult = 0)
  ) +
  scale_y_reverse(
    breaks = seq_len(n), labels = labels,
    limits = c(n + 0.5, 0.5), expand = expansion(mult = 0)
  ) +
  coord_fixed() +
  labs(
    title = "Learned attention weights",
    subtitle = sprintf('Input: "%s"  |  Best checkpoint: %s updates',
                       prompt,
                       format(best$iteration, big.mark = ",", scientific = FALSE)),
    x = "Key position (available input)",
    y = "Query position (current character)",
    caption = "Off-white cells: masked future positions. ␠ represents a space."
  ) +
  theme_minimal(base_family = "Helvetica Neue", base_size = 12) +
  theme(
    plot.background = element_rect(fill = "#FFFFFF", colour = NA),
    panel.background = element_rect(fill = "#FFFFFF", colour = NA),
    panel.grid = element_blank(),
    axis.ticks = element_blank(),
    text = element_text(colour = "#000000"),
    plot.title = element_text(size = 16, face = "bold", colour = "#000000",
                              margin = margin(b = 10)),
    plot.subtitle = element_text(size = 12, colour = "#000000",
                                 margin = margin(b = 18)),
    plot.caption = element_text(size = 12, colour = "#000000",
                                hjust = 0, margin = margin(t = 12)),
    axis.title = element_text(size = 12, colour = "#000000"),
    axis.text = element_text(size = 12, colour = "#000000"),
    legend.title = element_text(size = 12, colour = "#000000"),
    legend.text = element_text(size = 12, colour = "#000000"),
    plot.margin = margin(20, 24, 20, 20)
  )

p

# dir.create(dirname(output_svg), recursive = TRUE, showWarnings = FALSE)
ggsave(output_svg, plot = p, device = svglite::svglite,
       width = 8.5, height = 7.7, units = "in", bg = "#FFFFFF")

cat("SVG:", output_svg, "\n")
cat("Prompt:", encodeString(prompt, quote = '"'), "\n")
cat("Checkpoint:", format(best$iteration, big.mark = ",", scientific = FALSE),
    "updates | validation:",
    sprintf("%.6f", best$validation_loss_nats_per_char), "nats/character\n")
