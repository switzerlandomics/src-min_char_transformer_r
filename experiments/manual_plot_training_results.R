
# Figure 4: measured Transformer training and validation loss.
# Run from the repository root:
# Rscript experiments/plot_blog_training.R

library(ggplot2)
library(svglite)

run_dir <- "../output/20260920_190535_seed666"
metrics <- read.csv(file.path(run_dir, "metrics.csv"))

# Recorded for this experiment in experiment.log.
bigram_baseline <- 2.469208
uniform_baseline <- 4.174387

# Select the checkpoint using measured validation loss.
best <- metrics[which.min(metrics$validation_loss_nats_per_char), ]

# The plotted range begins at update 5,000. Update 0 and the
# uniform baseline are reported in the blog caption.
plot_data <- subset(metrics, iteration > 0)

series <- rbind(
  data.frame(
    iteration = plot_data$iteration,
    loss = plot_data$train_loss_nats_per_char,
    measurement = "Training (smoothed)"
  ),
  data.frame(
    iteration = plot_data$iteration,
    loss = plot_data$validation_loss_nats_per_char,
    measurement = "Validation (held-out)"
  )
)

series$measurement <- factor(
  series$measurement,
  levels = c("Training (smoothed)", "Validation (held-out)")
)

p <- ggplot(series, aes(iteration, loss, colour = measurement)) +
  geom_hline(
    yintercept = bigram_baseline,
    colour = "#929292",
    linetype = "dotted",
    linewidth = 0.65
  ) +
  geom_line(linewidth = 0.85) +
  geom_point(
    data = best,
    aes(
      x = iteration,
      y = validation_loss_nats_per_char
    ),
    inherit.aes = FALSE,
    colour = "#E5262F",
    fill = "#FFFFFF",
    shape = 21,
    size = 3.8,
    stroke = 1.4
  ) +
  annotate(
    "text",
    x = 105000,
    y = 2.12,
    label = sprintf(
      "Best validation: %.4f at %s updates",
      best$validation_loss_nats_per_char,
      format(best$iteration, big.mark = ",", scientific = FALSE)
    ),
    hjust = 0,
    colour = "#000000",
    family = "Helvetica Neue",
    size = 4
  ) +
  annotate(
    "text",
    x = 295000,
    y = bigram_baseline + 0.045,
    label = sprintf("Bigram baseline: %.3f", bigram_baseline),
    hjust = 1,
    colour = "#000000",
    family = "Helvetica Neue",
    size = 3.6
  ) +
  scale_colour_manual(
    values = c(
      "Training (smoothed)" = "#000000",
      "Validation (held-out)" = "#E5262F"
    )
  ) +
  scale_x_continuous(
    breaks = seq(0, 300000, by = 50000),
    labels = function(x) format(x, big.mark = ",", scientific = FALSE),
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_y_continuous(
    breaks = seq(1.6, 2.6, by = 0.2)
  ) +
  coord_cartesian(ylim = c(1.55, 2.62)) +
  labs(
    title = "Training and validation loss",
    x = "Parameter updates",
    y = "Loss (nats/character)",
    colour = NULL
  ) +
  theme_minimal(base_family = "Helvetica Neue", base_size = 12) +
  theme(
    plot.background = element_rect(fill = "#FFFFFF", colour = NA),
    panel.background = element_rect(fill = "#FFFFFF", colour = NA),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(
      colour = "#E5E5E5",
      linewidth = 0.35
    ),
    text = element_text(colour = "#000000"),
    plot.title = element_text(
      colour = "#000000",
      face = "bold",
      size = 16,
      margin = margin(b = 16)
    ),
    axis.title = element_text(colour = "#000000", size = 12),
    axis.text = element_text(colour = "#000000", size = 12),
    legend.text = element_text(colour = "#000000", size = 12),
    legend.position = "bottom",
    plot.margin = margin(18, 22, 18, 18)
  )
p

# dir.create("docs/figures", recursive = TRUE, showWarnings = FALSE)

ggsave(
  filename = "../docs/figures/04_training_curve.svg",
  plot = p,
  device = svglite::svglite,
  width = 11,
  height = 5.5,
  units = "in",
  bg = "#FFFFFF"
  # family = "Helvetica Neue"
)

cat(
  "Best validation:",
  sprintf("%.6f", best$validation_loss_nats_per_char),
  "at update", best$iteration, "\n"
)
