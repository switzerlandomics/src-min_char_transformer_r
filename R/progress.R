# Consistent, concise, human-readable experiment logs and progress reports.

format_duration <- function(seconds) {
  if (!is.finite(seconds) || seconds < 0) return("calculating")
  seconds <- as.integer(round(seconds))
  hours <- seconds %/% 3600L
  minutes <- (seconds %% 3600L) %/% 60L
  remaining <- seconds %% 60L
  if (hours > 0L) sprintf("%dh %02dm %02ds", hours, minutes, remaining)
  else if (minutes > 0L) sprintf("%dm %02ds", minutes, remaining)
  else sprintf("%ds", remaining)
}

make_logger <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  function(level, template, ...) {
    line <- sprintf("%s | %-6s | %s", base::format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                    level, sprintf(template, ...))
    cat(line, "\n", sep = "")
    cat(line, "\n", sep = "", file = path, append = TRUE)
    invisible(line)
  }
}

progress_status <- function(completed, total, elapsed, active_completed, active_elapsed,
                            bar_width = 24L) {
  fraction <- min(1, completed / total)
  filled <- floor(bar_width * fraction)
  bar <- paste0("[", paste0(rep("=", filled), collapse = ""),
                paste0(rep(".", bar_width - filled), collapse = ""), "]")
  rate <- if (active_elapsed > 0 && active_completed > 0) active_completed / active_elapsed else NA_real_
  eta <- if (is.finite(rate) && rate > 0) (total - completed) / rate else NA_real_
  sprintf("%s %5.1f%% | %d/%d | elapsed %s | ETA %s", bar, fraction * 100,
          completed, total, format_duration(elapsed), format_duration(eta))
}

# Replace a completed file only after its new contents have been written.
atomic_write <- function(path, writer) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".writing-", tmpdir = dirname(path))
  on.exit(if (file.exists(temporary)) unlink(temporary), add = TRUE)
  writer(temporary)
  if (!file.rename(temporary, path)) {
    # Windows does not always permit rename over an existing destination.
    if (file.exists(path)) unlink(path)
    if (!file.rename(temporary, path) && !file.copy(temporary, path, overwrite = TRUE)) {
      stop("Unable to save file: ", path)
    }
  }
  invisible(path)
}

write_rds <- function(object, path) atomic_write(path, function(tmp) saveRDS(object, tmp))
write_csv <- function(data, path) {
  atomic_write(path, function(tmp) utils::write.csv(data, tmp, row.names = FALSE))
}

with_sampling_seed <- function(seed, expression) {
  had_rng <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_rng) old_rng <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_rng) assign(".Random.seed", old_rng, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  force(expression)
}
