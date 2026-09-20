#!/usr/bin/env Rscript
# Experiment orchestration only. Neural-network mathematics is in R/model.R.

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(file_arg)) return(normalizePath("experiments/run.R", mustWork = FALSE))
  normalizePath(sub("^--file=", "", file_arg[1L]), mustWork = TRUE)
}
PROJECT_ROOT <- normalizePath(file.path(dirname(script_path()), ".."), mustWork = TRUE)
setwd(PROJECT_ROOT)
for (module in c("data", "model", "optimiser", "evaluation", "progress", "plots")) {
  source(file.path("R", paste0(module, ".R")))
}

REFERENCE_INPUT_URL <- paste0("https://raw.githubusercontent.com/karpathy/char-rnn/",
                              "master/data/tinyshakespeare/input.txt")

default_config <- function() {
  list(input = "data/tiny_shakespeare.txt", output = "output", iterations = 2000L,
       context_length = 32L, embedding_size = 32L, feedforward_size = 64L,
       lr = 0.001, seed = 42L, clip_norm = 1, log_interval = 100L,
       validation_interval = 500L, checkpoint_interval = 500L,
       sample_interval = 500L, sample_length = 180L,
       n_passages = 8L, passage_chars = 256L, train_fraction = 0.8,
       validation_fraction = 0.1, plot = TRUE, smoke = FALSE, resume = "",
       test = FALSE)
}

print_help <- function() {
  cat(paste(c(
    "min-char-transformer | base R | CPU-only | manually differentiated",
    "Usage: Rscript experiments/run.R [options]", "",
    "--input=PATH                UTF-8 corpus (download Tiny Shakespeare if absent)",
    "--output=DIR                Experiment parent directory",
    "--iterations=N              Total target parameter updates, including resume",
    "--context-length=N          Context window (new runs)",
    "--embedding-size=N          Model width (new runs)",
    "--feedforward-size=N        Feed-forward width (new runs)",
    "--lr=VALUE                  Adam learning rate (new runs)",
    "--clip-norm=VALUE           Global gradient norm threshold (new runs)",
    "--seed=N                    RNG seed (new runs)",
    "--log-interval=N            Progress reporting frequency",
    "--validation-interval=N     Fixed held-out checks",
    "--checkpoint-interval=N     Full resumable checkpoint frequency",
    "--sample-interval=N         Generated-text frequency",
    "--sample-length=N           Generated characters per sample",
    "--n-passages=N              Fixed validation passage count (new runs)",
    "--passage-chars=N           Transitions per passage; multiple of context",
    "--no-plot                   Train without optional ggplot2",
    "--smoke                     Small, offline training-pipeline test",
    "--resume=DIR                Resume DIR/latest_checkpoint.rds",
    "--test                      Evaluate a completed run's selected model once",
    "--help                      Show these options", "",
    "For a clean stop, create a STOP file in the experiment directory.",
    "Remove STOP before resuming. model.rds alone cannot resume training."
  ), collapse = "\n"), "\n")
}

parse_args <- function(args, config) {
  supplied <- character()
  for (arg in args) {
    if (arg %in% c("--help", "-h")) { print_help(); quit(status = 0L) }
    if (arg %in% c("--smoke", "--no-plot", "--test")) {
      key <- switch(arg, "--smoke" = "smoke", "--no-plot" = "plot", "--test" = "test")
      config[[key]] <- arg != "--no-plot"
      supplied <- c(supplied, key)
      next
    }
    parts <- regmatches(arg, regexec("^--([^=]+)=(.*)$", arg))[[1L]]
    if (length(parts) != 3L) stop("Unknown argument: ", arg)
    key <- gsub("-", "_", parts[2L], fixed = TRUE)
    if (!key %in% names(config) || key %in% c("smoke", "plot", "test")) {
      stop("Unknown option: ", arg)
    }
    value <- parts[3L]
    if (is.integer(config[[key]])) {
      value <- suppressWarnings(as.integer(value))
      if (is.na(value)) stop("Expected integer: ", arg)
    } else if (is.numeric(config[[key]])) {
      value <- suppressWarnings(as.numeric(value))
      if (!is.finite(value)) stop("Expected finite number: ", arg)
    }
    config[[key]] <- value
    supplied <- c(supplied, key)
  }
  list(config = config, supplied = unique(supplied))
}

apply_smoke_defaults <- function(config, supplied) {
  if (!config$smoke) return(config)
  defaults <- list(input = "data/input.txt", iterations = 30L, context_length = 8L,
    embedding_size = 8L, feedforward_size = 16L, log_interval = 10L,
    validation_interval = 10L, checkpoint_interval = 15L,
    sample_interval = 15L, sample_length = 60L, n_passages = 1L,
    passage_chars = 16L)
  for (name in names(defaults)) if (!name %in% supplied) config[[name]] <- defaults[[name]]
  config
}

validate_config <- function(config) {
  for (name in c("iterations", "context_length", "embedding_size", "feedforward_size",
                 "seed", "log_interval", "validation_interval", "checkpoint_interval",
                 "sample_interval", "sample_length", "n_passages", "passage_chars")) {
    assert_positive_integer(config[[name]], name)
  }
  if (config$embedding_size < 2L || config$passage_chars %% config$context_length != 0L ||
      config$passage_chars < config$context_length || config$lr <= 0 ||
      config$clip_norm <= 0 || config$train_fraction <= 0 ||
      config$validation_fraction <= 0 ||
      config$train_fraction + config$validation_fraction >= 1) {
    stop("Invalid model, evaluation or optimiser configuration.")
  }
  invisible(TRUE)
}

ensure_input_file <- function(config, supplied) {
  if (file.exists(config$input)) return(invisible(config$input))
  if (!"input" %in% supplied && identical(config$input, "data/tiny_shakespeare.txt")) {
    dir.create(dirname(config$input), showWarnings = FALSE, recursive = TRUE)
    message("[setup] Downloading Tiny Shakespeare ...")
    status <- tryCatch(utils::download.file(REFERENCE_INPUT_URL, config$input,
                                            mode = "wb", quiet = TRUE),
                       error = function(e) stop("Corpus download failed: ", conditionMessage(e)))
    if (status != 0L || !file.exists(config$input)) stop("Corpus download failed.")
    return(invisible(config$input))
  }
  stop("Input file does not exist: ", config$input)
}

make_experiment_dir <- function(parent, seed) {
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  stamp <- paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_seed", seed)
  directory <- file.path(parent, stamp)
  suffix <- 1L
  while (file.exists(directory)) {
    directory <- file.path(parent, paste0(stamp, "_", suffix))
    suffix <- suffix + 1L
  }
  dir.create(directory)
  directory
}

append_sample <- function(path, model, vocab, prompt, iteration, validation_loss, config) {
  prompt_indices <- encode_text(prompt, vocab)
  generated <- with_sampling_seed(config$seed + 100000L + iteration,
    sample_indices(model, prompt_indices, config$sample_length))
  text <- decode_indices(generated, vocab)
  cat(sprintf("\n=== update %d | validation %.6f | prompt %s | temperature 1 ===\n",
              iteration, validation_loss, encodeString(prompt, quote = '"')),
      prompt, text, "\n", file = path, append = TRUE, sep = "")
}

make_attention_snapshot <- function(model, vocab, prompt, iteration) {
  indices <- encode_text(prompt, vocab)
  if (length(indices) > model$config$context_length) {
    indices <- tail(indices, model$config$context_length)
  }
  result <- forward_transformer(model, indices, retain_cache = FALSE)
  list(iteration = iteration, prompt = decode_indices(indices, vocab),
       characters = vocab$ix_to_char[indices], attention = result$attention,
       probabilities = row_softmax(result$logits))
}

# --test is deliberately an independent, explicit operation. It is never used
# for checkpoint selection, hyperparameter tuning, or routine live monitoring.
run_final_test <- function(directory, config, test_indices, test_plan, log_msg) {
  result_path <- file.path(directory, "test_results.rds")
  if (file.exists(result_path)) stop("Test has already been evaluated in this directory.")
  best <- load_model(file.path(directory, "best_model.rds"))
  measurement <- evaluate_passages(best, test_indices, test_plan, config$context_length)
  write_rds(list(loss_nats_per_char = measurement$loss,
                 passages = measurement$passages,
                 model_file = "best_model.rds", evaluated_at = as.character(Sys.time())),
            result_path)
  log_msg("TEST", "selected checkpoint | loss=%.6f nats/char | %d passages",
          measurement$loss, nrow(test_plan))
}

parsed <- parse_args(commandArgs(trailingOnly = TRUE), default_config())
config <- apply_smoke_defaults(parsed$config, parsed$supplied)
resume <- nzchar(config$resume)
checkpoint <- NULL
if (resume) {
  location <- if (dir.exists(config$resume)) file.path(config$resume, "latest_checkpoint.rds")
              else config$resume
  if (!file.exists(location)) stop("Checkpoint not found: ", location)
  checkpoint <- readRDS(location)
  if (!identical(checkpoint$version, 1L)) stop("Unsupported checkpoint format.")
  allowed <- c("iterations", "log_interval", "validation_interval", "checkpoint_interval",
               "sample_interval", "sample_length", "plot", "resume", "test")
  forbidden <- setdiff(parsed$supplied, allowed)
  if (length(forbidden)) stop("Cannot change these options on resume: ",
                              paste(forbidden, collapse = ", "))
  original <- checkpoint$config
  for (name in intersect(parsed$supplied, allowed)) original[[name]] <- config[[name]]
  config <- original
  config$resume <- location
  if (config$iterations < checkpoint$iteration) stop("Target updates precede the checkpoint.")
}
validate_config(config)
ensure_input_file(config, if (resume) "input" else parsed$supplied)
if (config$plot && !requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Install ggplot2 or rerun with --no-plot.")
}
if (config$plot) require_plots()

if (resume) {
  experiment_dir <- dirname(normalizePath(location, mustWork = TRUE))
  if (file.exists(file.path(experiment_dir, "STOP"))) stop("Remove STOP before resuming.")
  if (file.exists(file.path(experiment_dir, "FINISHED"))) unlink(file.path(experiment_dir, "FINISHED"))
} else {
  set.seed(config$seed)
  experiment_dir <- make_experiment_dir(config$output, config$seed)
}
path <- function(name) file.path(experiment_dir, name)
log_msg <- make_logger(path("experiment.log"))
log_msg("START", "min-char-transformer | base R | CPU-only | %s",
        if (resume) "resuming" else "new experiment")
log_msg("STAGE", "[1/5] Preparing text and fixed evaluation passages")
text <- read_text_file(config$input)
split <- split_characters(text, config$train_fraction, config$validation_fraction,
                          minimum_size = config$passage_chars + 1L)
vocab <- build_vocab(split$train)
train_ids <- encode_text(split$train, vocab)
validation_ids <- encode_text(split$validation, vocab)
test_ids <- encode_text(split$test, vocab)
validation_plan <- make_passage_plan(length(validation_ids), config$context_length,
                                     config$n_passages, config$passage_chars)
test_plan <- make_passage_plan(length(test_ids), config$context_length,
                               config$n_passages, config$passage_chars)
starts <- training_window_starts(length(train_ids), config$context_length)
sample_prompt <- decode_indices(head(train_ids, min(4L, config$context_length)), vocab)
attention_prompt <- decode_indices(head(train_ids, min(8L, config$context_length)), vocab)
checksum <- input_checksum(config$input)
log_msg("DATA", "characters=%d | vocabulary=%d | train=%d | validation=%d | test=%d",
        length(train_ids) + length(validation_ids) + length(test_ids), vocab$size,
        length(train_ids), length(validation_ids), length(test_ids))
log_msg("DATA", "validation=%d fixed passages x %d transitions | md5=%s",
        nrow(validation_plan), config$passage_chars, checksum)

if (resume) {
  if (!identical(checkpoint$input_md5, checksum) ||
      !identical(checkpoint$vocab_chars, vocab$chars) ||
      !identical(checkpoint$train_size, length(train_ids))) {
    stop("Corpus, training split, or vocabulary differs from saved checkpoint.")
  }
  model <- checkpoint$model; optimiser <- checkpoint$optimiser
  window_order <- checkpoint$window_order; next_window <- checkpoint$next_window
  smooth_loss <- checkpoint$smooth_loss
  metrics <- checkpoint$metrics; passage_metrics <- checkpoint$passage_metrics
  best_loss <- checkpoint$best_loss; best_iteration <- checkpoint$best_iteration
  iteration_start <- checkpoint$iteration; elapsed_before <- checkpoint$elapsed_seconds
  assign(".Random.seed", checkpoint$rng_state, envir = .GlobalEnv)
  log_msg("RESUME", "checkpoint update=%d | training elapsed=%s",
          iteration_start, format_duration(elapsed_before))
} else {
  log_msg("STAGE", "[2/5] Initialising one-block causal Transformer")
  model <- initialise_model(vocab$size, config$context_length,
                             config$embedding_size, config$feedforward_size)
  optimiser <- new_adam_state(model)
  window_order <- sample.int(length(starts)); next_window <- 1L
  smooth_loss <- log(vocab$size)
  measured <- evaluate_passages(model, validation_ids, validation_plan, config$context_length)
  best_loss <- measured$loss; best_iteration <- 0L
  metrics <- data.frame(iteration = 0L, train_loss_nats_per_char = NA_real_,
    validation_loss_nats_per_char = measured$loss, elapsed_seconds = 0,
    characters_processed = 0, approximate_corpus_passes = 0)
  passage_metrics <- cbind(iteration = 0L, measured$passages)
  save_model(model, path("best_model.rds")); save_vocab(vocab, path("vocab.rds"))
  cat("", file = path("samples.txt"))
  append_sample(path("samples.txt"), model, vocab, sample_prompt, 0L, best_loss, config)
  iteration_start <- 0L; elapsed_before <- 0
  log_msg("MODEL", "context=%d | embedding=%d | feed-forward=%d | heads=1 | blocks=1 | parameters=%d",
          config$context_length, config$embedding_size, config$feedforward_size,
          model_parameter_count(model))
  log_msg("MODEL", "optimiser=Adam | lr=%.6g | gradient norm clip=%.3g",
          config$lr, config$clip_norm)
  log_msg("BASE", "uniform=%.6f | bigram=%.6f | initial validation=%.6f nats/char",
          log(vocab$size), bigram_baseline(train_ids, validation_ids,
                                          validation_plan, vocab$size), best_loss)
}

metadata <- if (resume && file.exists(path("metadata.rds"))) readRDS(path("metadata.rds")) else list()
if (is.null(metadata$created_at)) metadata$created_at <- as.character(Sys.time())
metadata$config <- config
metadata$R_version <- R.version.string
metadata$RNG_kind <- RNGkind()
metadata$input_md5 <- checksum
metadata$input_file <- config$input
metadata$split_sizes <- split$sizes
metadata$vocabulary_size <- vocab$size
metadata$parameter_count <- model_parameter_count(model)
metadata$uniform_baseline_nats_per_char <- log(vocab$size)
metadata$bigram_baseline_nats_per_char <- bigram_baseline(train_ids, validation_ids,
    validation_plan, vocab$size)
metadata$validation_plan <- validation_plan
metadata$test_plan <- test_plan

# Plotting can fail independently of training, and is always reconstructed from
# saved metrics and model-derived attention snapshots.
render_plots <- function() {
  if (!config$plot) return(invisible(FALSE))
  tryCatch({ plot_experiment(experiment_dir); TRUE }, error = function(e) {
    log_msg("PLOT", "render skipped: %s", conditionMessage(e)); FALSE
  })
}

persist <- function(iteration, elapsed, checkpoint_now = FALSE, make_plots = FALSE) {
  write_csv(metrics, path("metrics.csv"))
  write_csv(passage_metrics, path("validation_passages.csv"))
  metadata$updated_at <<- as.character(Sys.time())
  metadata$completed_iterations <<- iteration
  metadata$best_iteration <<- best_iteration
  metadata$best_validation_nats_per_char <<- best_loss
  write_rds(metadata, path("metadata.rds"))
  if (checkpoint_now) {
    snapshot <- list(version = 1L, config = config, iteration = iteration,
      model = model, optimiser = optimiser, window_order = window_order,
      next_window = next_window, smooth_loss = smooth_loss, metrics = metrics,
      passage_metrics = passage_metrics, best_loss = best_loss,
      best_iteration = best_iteration, elapsed_seconds = elapsed,
      rng_state = .Random.seed, input_md5 = checksum,
      vocab_chars = vocab$chars, train_size = length(train_ids))
    write_rds(snapshot, path("latest_checkpoint.rds"))
  }
  if (make_plots) render_plots()
  invisible(NULL)
}

if (!resume) {
  write_rds(make_attention_snapshot(model, vocab, attention_prompt, 0L),
            path("attention_initial.rds"))
  write_rds(make_attention_snapshot(model, vocab, attention_prompt, 0L),
            path("attention_snapshot.rds"))
  persist(0L, 0, checkpoint_now = TRUE, make_plots = TRUE)
}
if (config$plot) log_msg("PLOT", "open %s in your browser",
                         normalizePath(path("training.html"), winslash = "/", mustWork = FALSE))
log_msg("STAGE", "[3/5] Training | progress every %d | validation every %d | checkpoint every %d",
        config$log_interval, config$validation_interval, config$checkpoint_interval)

started <- proc.time()[["elapsed"]]
last_completed <- iteration_start
stopped <- FALSE
if (iteration_start < config$iterations) {
  for (iteration in seq.int(iteration_start + 1L, config$iterations)) {
    if (next_window > length(window_order)) {
      window_order <- sample.int(length(starts))
      next_window <- 1L
    }
    start <- starts[window_order[next_window]]
    next_window <- next_window + 1L
    window <- sequence_window(train_ids, start, config$context_length)
    result <- loss_and_gradients(model, window$inputs, window$targets)
    clipped <- clip_gradients(result$grads, config$clip_norm)
    updated <- adam_update(model, clipped$grads, optimiser, learning_rate = config$lr)
    model <- updated$model; optimiser <- updated$state
    assert_model_finite(model)
    smooth_loss <- 0.99 * smooth_loss + 0.01 * result$loss
    last_completed <- iteration
    active_elapsed <- proc.time()[["elapsed"]] - started
    elapsed <- elapsed_before + active_elapsed
    if (iteration %% config$log_interval == 0L || iteration == config$iterations) {
      log_msg("TRAIN", "%s | loss=%.4f nats/char",
         progress_status(iteration, config$iterations, elapsed,
                         iteration - iteration_start, active_elapsed), smooth_loss)
    }
    check <- iteration %% config$validation_interval == 0L || iteration == config$iterations
    if (check) {
      measured <- evaluate_passages(model, validation_ids, validation_plan,
                                    config$context_length)
      metrics <- rbind(metrics, data.frame(iteration = iteration,
        train_loss_nats_per_char = smooth_loss,
        validation_loss_nats_per_char = measured$loss,
        elapsed_seconds = elapsed,
        characters_processed = as.double(iteration) * config$context_length,
        approximate_corpus_passes = (as.double(iteration) * config$context_length) /
                                     max(1, length(train_ids) - 1L)))
      passage_metrics <- rbind(passage_metrics, cbind(iteration = iteration,
                                                     measured$passages))
      if (measured$loss < best_loss) {
        best_loss <- measured$loss; best_iteration <- iteration
        save_model(model, path("best_model.rds"))
        log_msg("SAVE", "best validation model at update %d | %.6f nats/char",
                iteration, best_loss)
      }
      log_msg("VALID", "loss=%.6f nats/char | best=%.6f at update %d",
              measured$loss, best_loss, best_iteration)
      write_rds(make_attention_snapshot(model, vocab, attention_prompt, iteration),
                path("attention_snapshot.rds"))
    }
    if (iteration %% config$sample_interval == 0L || iteration == config$iterations) {
      append_sample(path("samples.txt"), model, vocab, sample_prompt, iteration,
                    if (check) measured$loss else best_loss, config)
    }
    checkpoint_now <- iteration %% config$checkpoint_interval == 0L ||
                      iteration == config$iterations
    stop_requested <- file.exists(path("STOP"))
    if (check || checkpoint_now || stop_requested) {
      persist(iteration, elapsed, checkpoint_now = checkpoint_now || stop_requested,
              make_plots = check)
    }
    if (stop_requested) {
      log_msg("STOP", "requested stop at update %d; resumable state saved", iteration)
      stopped <- TRUE
      break
    }
  }
}

log_msg("STAGE", "[4/5] Saving final state")
elapsed <- elapsed_before + (proc.time()[["elapsed"]] - started)
save_model(model, path("model.rds"))
persist(last_completed, elapsed, checkpoint_now = TRUE, make_plots = TRUE)
if (!stopped) {
  cat("\n", file = path("FINISHED"))
  log_msg("STAGE", "[5/5] Complete | updates=%d | time=%s | best validation=%.6f",
          last_completed, format_duration(elapsed), best_loss)
} else log_msg("STAGE", "[5/5] Paused; remove STOP and use --resume to continue")
log_msg("FILES", "%s | metrics.csv | samples.txt | best_model.rds | latest_checkpoint.rds",
        normalizePath(experiment_dir, winslash = "/", mustWork = TRUE))

if (config$test) {
  if (stopped || last_completed != config$iterations) {
    stop("Final test requires reaching the requested total update target.")
  }
  run_final_test(experiment_dir, config, test_ids, test_plan, log_msg)
}
