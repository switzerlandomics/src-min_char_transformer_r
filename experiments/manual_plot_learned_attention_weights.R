# Figure 5: attention for the exact initial prompt in the saved generation comparison.
# Open in RStudio and Source. Working directory: repository root OR experiments/.
# Requires ggplot2 and svglite for plotting; model inference uses base R only.

library(ggplot2)
library(svglite)

# Locate the repository without relying on Rscript arguments (which do not
# describe a script sourced interactively in RStudio).
if (file.exists(file.path("R", "model.R"))) {
  root <- normalizePath(".")
} else if (file.exists(file.path("..", "R", "model.R"))) {
  root <- normalizePath("..")
} else {
  stop("Set the RStudio working directory to the repository root or experiments/.")
}

run_dir <- file.path(root, "output", "20260920_190535_seed666")
output_svg <- file.path(root, "docs", "figures", "05_learned_attention_map.svg")
required <- c("best_model.rds", "vocab.rds", "metadata.rds", "metrics.csv",
              "generation_comparison.rds")
missing <- required[!file.exists(file.path(run_dir, required))]
if (length(missing)) stop("Missing files in selected run: ", paste(missing, collapse = ", "))

source(file.path(root, "R", "data.R"))
source(file.path(root, "R", "model.R"))
source(file.path(root, "R", "progress.R"))

model <- load_model(file.path(run_dir, "best_model.rds"))
vocab <- load_vocab(file.path(run_dir, "vocab.rds"))
metadata <- readRDS(file.path(run_dir, "metadata.rds"))
metrics <- read.csv(file.path(run_dir, "metrics.csv"))
comparison <- readRDS(file.path(run_dir, "generation_comparison.rds"))

# The recorded generation comparison, not a hard-coded example or RStudio
# command-line arguments, determines which prompt this figure describes.
if (!is.list(comparison) || !is.list(comparison$initial) ||
    !is.list(comparison$best)) {
  stop("Unexpected generation_comparison.rds structure: expected $initial and $best.")
}
prompt <- comparison$initial$prompt
if (!is.character(prompt) || length(prompt) != 1L || is.na(prompt) ||
    !nzchar(prompt) || !identical(prompt, comparison$best$prompt)) {
  stop("Initial and best-generation prompts are missing or different.")
}
if (!is.null(metadata$generation_prompt) &&
    !identical(prompt, metadata$generation_prompt)) {
  stop("Generation-comparison prompt disagrees with metadata$generation_prompt.")
}

# Check the recorded selected update against all available provenance.
required_columns <- c("iteration", "validation_loss_nats_per_char")
if (!all(required_columns %in% names(metrics))) stop("Missing metrics columns.")
valid <- which(is.finite(metrics$validation_loss_nats_per_char))
if (!length(valid)) stop("No finite validation losses in metrics.csv.")
best <- metrics[valid[which.min(metrics$validation_loss_nats_per_char[valid])],
                , drop = FALSE]

same_number <- function(actual, expected, tolerance = 1e-7) {
  is.numeric(actual) && length(actual) == 1L && is.finite(actual) &&
    is.numeric(expected) && length(expected) == 1L && is.finite(expected) &&
    isTRUE(all.equal(as.numeric(actual), as.numeric(expected),
                     tolerance = tolerance))
}
if (!same_number(metadata$best_iteration, best$iteration, tolerance = 0) ||
    !same_number(comparison$best$iteration, best$iteration, tolerance = 0)) {
  stop("Selected update disagrees between metrics, metadata and generation comparison.")
}
if (!same_number(metadata$best_validation_nats_per_char,
                 best$validation_loss_nats_per_char) ||
    !same_number(comparison$best$validation_loss_nats_per_char,
                 best$validation_loss_nats_per_char)) {
  stop("Selected validation loss disagrees between saved records.")
}

indices <- encode_text(prompt, vocab)
if (length(indices) < 2L || length(indices) > model$config$context_length ||
    vocab$size != model$config$vocab_size) {
  stop("Recorded prompt or vocabulary is incompatible with the selected model.")
}
# Reproduce the recorded best-checkpoint continuation before making the figure.
# This confirms that the currently saved model, vocabulary, prompt and sampling
# setup really reproduce the text used in the blog, rather than merely sharing
# the same run-directory name and recorded checkpoint number.
record <- comparison$best
if (!is.numeric(record$seed) || length(record$seed) != 1L ||
    !is.finite(record$seed) || !is.numeric(record$n_generated) ||
    length(record$n_generated) != 1L || !is.finite(record$n_generated) ||
    record$n_generated < 1 ||
    !is.numeric(record$temperature) || length(record$temperature) != 1L ||
    !is.finite(record$temperature) || record$temperature <= 0 ||
    !is.character(record$generated_text) || length(record$generated_text) != 1L) {
  stop("Missing or invalid recorded best-generation sampling settings/text.")
}
replayed_indices <- with_sampling_seed(
  record$seed,
  sample_indices(model, indices, n = record$n_generated,
                 temperature = record$temperature)
)
if (!identical(decode_indices(replayed_indices, vocab), record$generated_text)) {
  stop(paste(
    "The saved model/vocabulary does not reproduce the recorded best-generation text.",
    "Check the checkpoint, code version and R sampling/RNG configuration before",
    "claiming that the plot matches the generation comparison."
  ))
}

characters <- vocab$ix_to_char[indices]
labels <- vapply(characters, function(ch) {
  if (ch == " ") "␠" else if (ch == "\n") "↵" else if (ch == "\t") "⇥" else ch
}, character(1L))

# These are the actual weights for the *initial prompt only*, before sampling
# the first generated character. Rows = queries; columns = keys.
attention <- forward_transformer(model, indices, retain_cache = FALSE)$attention
n <- length(indices)
if (!is.matrix(attention) || !identical(dim(attention), c(n, n)) ||
    any(!is.finite(attention)) || any(attention < -1e-10) ||
    any(attention > 1 + 1e-10) ||
    any(abs(rowSums(attention) - 1) > 1e-7) ||
    any(abs(attention[upper.tri(attention)]) > 1e-12)) {
  stop("Model output is not a valid row-normalised causal attention matrix.")
}

tiles <- expand.grid(query = seq_len(n), key = seq_len(n))
tiles$permitted <- tiles$key <= tiles$query
tiles$weight <- attention[cbind(tiles$query, tiles$key)]
measured <- tiles[tiles$permitted, , drop = FALSE]

p <- ggplot() +
  geom_tile(data = tiles, aes(x = key, y = query), width = 1, height = 1,
            fill = "#F3F2F2", colour = "#D9D9D9", linewidth = 0.5) +
  geom_tile(data = measured, aes(x = key, y = query, fill = weight),
            width = 1, height = 1, colour = "#D9D9D9", linewidth = 0.5) +
  scale_fill_gradientn(
    colours = c("#FDEBED", "#F88379", "#E5262F"),
    limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1),
    name = "Attention weight",
    guide = guide_colorbar(barwidth = grid::unit(6, "mm"),
                           barheight = grid::unit(52, "mm"),
                           frame.colour = "#000000", ticks.colour = "#000000")
  ) +
  scale_x_continuous(position = "top", breaks = seq_len(n), labels = labels,
                     limits = c(0.5, n + 0.5), expand = expansion(mult = 0)) +
  scale_y_reverse(breaks = seq_len(n), labels = labels,
                  limits = c(n + 0.5, 0.5), expand = expansion(mult = 0)) +
  coord_fixed() +
  labs(
    title = "Learned attention weights",
    subtitle = sprintf("Initial prompt: %s  |  Best checkpoint: %s updates",
                       encodeString(prompt, quote = '"'),
                       format(best$iteration, big.mark = ",", scientific = FALSE)),
    x = "Key position (supplied input)",
    y = "Query position (current input)",
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
    plot.caption = element_text(size = 12, colour = "#000000", hjust = 0,
                                margin = margin(t = 12)),
    axis.title = element_text(size = 12, colour = "#000000"),
    axis.text = element_text(size = 12, colour = "#000000"),
    legend.title = element_text(size = 12, colour = "#000000"),
    legend.text = element_text(size = 12, colour = "#000000"),
    plot.margin = margin(20, 24, 20, 20)
  )

print(p)  # Display in the RStudio Plots pane.
# dir.create(dirname(output_svg), recursive = TRUE, showWarnings = FALSE)
ggsave(output_svg, plot = p, device = svglite::svglite,
       width = 8.5, height = 7.7, units = "in", bg = "#FFFFFF")

cat("Saved:", output_svg, "\n")
cat("Source: best_model.rds and generation_comparison.rds\n")
cat("Recorded best-generation continuation reproduced exactly.\n")
cat("Initial prompt:", encodeString(prompt, quote = '"'), "\n")
cat("Selected checkpoint:", best$iteration, "updates; validation loss:",
    sprintf("%.6f", best$validation_loss_nats_per_char), "nats/character\n")
cat("Measured attention matrix (queries in rows, keys in columns):\n")
dimnames(attention) <- list(paste0(seq_len(n), ":", labels),
                            paste0(seq_len(n), ":", labels))
print(round(attention, 4))
