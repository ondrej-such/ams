# -----------------------------------------------------------------------------
# data_preparation.R
#
# recordings -> 500 ms speech cuts -> ECAPA-TDNN -> 192-dim embeddings,
# L2 normalised to unit length -> one CSV per language x story cell.
#
# Sourcing this file leaves `tr` (English) and `te` (Urdu) in memory.
# If the CSVs already exist they are simply read; the audio pipeline runs only
# when they are missing, so a collaborator can reproduce it from the recordings
# but does not have to.
#
# The audio pipeline needs Python: torch, torchaudio, speechbrain, soundfile,
# truststore, plus an ~80 MB pretrained model downloaded on first run.
# -----------------------------------------------------------------------------

rec_dir <- "recordings"
out_dir <- "data"

# only used when rebuilding from audio: full path to a python.exe that has
# torch, torchaudio, speechbrain, soundfile and truststore installed. Leave
# empty to let reticulate choose.
python_exe <- ""

stories <- list(
  EN = c(A = "AEN", M = "MEN", S = "SEN"),   # english, story N
  EB = c(A = "AEB", M = "MEB", S = "SEB"),   # english, story B
  UN = c(A = "AUN", M = "MUN", S = "SUN"),   # urdu,    story N
  UB = c(A = "AUB", M = "MUB", S = "SUB")    # urdu,    story B
)

seg_ms          <- 500L     # length of each cut
target_sr       <- 16000L   # rate the pretrained model expects
vad_floor_db    <- 40       # a frame is speech if within this many dB of peak
min_speech_frac <- 0.80     # fraction of a cut's frames that must be speech
cand_step_ms    <- 25L      # spacing of candidate cut positions

# noise augmentation, following Such et al. Subsection II-C
conditions <- rbind(
  data.frame(type = "white", snr = c(5, 10, 15, 20)),
  data.frame(type = "pink",  snr = c(5, 10, 15, 20)),
  data.frame(type = "brown", snr = 10)
)

build_embeddings <- function() {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  set.seed(1)                # noise realisations; everything else deterministic

  suppressPackageStartupMessages(library(reticulate))
  # Recent reticulate versions create their own empty ephemeral environment if
  # no interpreter is named, which will not have torch or speechbrain. Point
  # python_exe at the installation that does.
  if (nzchar(python_exe)) use_python(python_exe, required = TRUE)

  # behind a TLS-intercepting proxy the model download fails with
  # CERTIFICATE_VERIFY_FAILED; harmless to attempt otherwise
  try(import("truststore", convert = FALSE)$inject_into_ssl(), silent = TRUE)

  np <- import("numpy", convert = FALSE); sf <- import("soundfile", convert = FALSE)
  torch <- import("torch", convert = FALSE); ta <- import("torchaudio", convert = FALSE)
  torch$set_num_threads(4L)

  enc <- tryCatch(
    import("speechbrain.inference.speaker", convert = FALSE)$EncoderClassifier,
    error = function(e) import("speechbrain.pretrained", convert = FALSE)$EncoderClassifier)
  # SpeechBrain stages model files by symlink, which needs elevated privileges
  # on Windows; COPY duplicates them instead
  LocalStrategy <- import("speechbrain.utils.fetching", convert = FALSE)$LocalStrategy
  model <- enc$from_hparams(
    source = "speechbrain/spkrec-ecapa-voxceleb",
    savedir = file.path(out_dir, "model_ecapa"),
    run_opts = dict(device = "cpu"), local_strategy = LocalStrategy$COPY)

  seg_len <- as.integer(target_sr * seg_ms / 1000)
  fr_len  <- as.integer(target_sr * 0.025)   # 25 ms frames, 10 ms hop, matching
  fr_hop  <- as.integer(target_sr * 0.010)   # the network's own front end

  embed_l2 <- function(m) {
    e <- py_to_r(model$encode_batch(
      torch$from_numpy(np$asarray(m, dtype = np$float32)))$squeeze(1L)$detach()$numpy())
    e <- e / sqrt(rowSums(e^2))
    colnames(e) <- sprintf("dim_%03d", seq_len(ncol(e)))
    e
  }

  # coloured noise by shaping a white spectrum: power ~ 1/f for pink, 1/f^2 for
  # brown, so amplitude scales as 1/sqrt(f) and 1/f
  gen_noise <- function(n, type) {
    w <- rnorm(n)
    if (type == "white") return(w / sd(w))
    k <- 0:(n - 1); f <- pmin(k, n - k); f[1] <- 1
    amp <- switch(type, pink = 1 / sqrt(f), brown = 1 / f)
    x <- Re(fft(fft(w) * amp, inverse = TRUE)) / n
    x / sd(x)
  }
  mix_at_snr <- function(sig, noise, snr_db)
    sig + noise * sqrt(mean(sig^2) / (mean(noise^2) * 10^(snr_db / 10)))

  # greedy earliest-first selection gives the maximum number of NON-OVERLAPPING
  # cuts, so no two rows share audio
  cut_positions <- function(path) {
    res <- sf$read(path, dtype = "float32", always_2d = TRUE)
    wav <- np$mean(res[[0]], axis = 1L); sr <- py_to_r(res[[1]])
    wav_t <- torch$from_numpy(np$ascontiguousarray(wav))$unsqueeze(0L)
    if (sr != target_sr)
      wav_t <- ta$functional$resample(wav_t, as.integer(sr), target_sr)
    sig <- py_to_r(wav_t$squeeze(0L)$numpy()); n <- length(sig)

    fr_idx <- seq(0, n - fr_len, by = fr_hop)
    fr_rms <- sqrt(colMeans(matrix(sig[outer(seq_len(fr_len), fr_idx, "+")],
                                   nrow = fr_len)^2))
    fr_db  <- 20 * log10(fr_rms + 1e-12)
    is_speech <- fr_db > (max(fr_db) - vad_floor_db)

    cand <- seq(0, n - seg_len, by = as.integer(target_sr * cand_step_ms / 1000))
    csum <- c(0, cumsum(is_speech))
    f1 <- floor(cand / fr_hop) + 1
    f2 <- floor((cand + seg_len - fr_len) / fr_hop) + 1
    ok <- which((csum[f2 + 1] - csum[f1]) / (f2 - f1 + 1) >= min_speech_frac)

    starts <- integer(0); next_free <- 0
    for (j in ok) if (cand[j] >= next_free) {
      starts <- c(starts, cand[j]); next_free <- cand[j] + seg_len
    }
    cat(sprintf("    %-9s %6.2f s | speech %4.1f%% | %d cuts\n", basename(path),
                n / target_sr, 100 * mean(is_speech), length(starts)))
    list(sig = sig, starts = starts)
  }

  for (story in names(stories)) {
    cat(sprintf("\n=== %s ===\n", story))
    recs <- setNames(file.path(rec_dir, paste0(stories[[story]], ".mpeg")),
                     names(stories[[story]]))
    prep <- lapply(recs, cut_positions)

    # every speaker contributes the same number of cuts, so the three classes
    # are balanced by construction
    n_cuts <- min(sapply(prep, function(p) length(p$starts)))
    cat(sprintf("  balanced to %d cuts per speaker\n", n_cuts))

    clean_tabs <- list(); noisy_tabs <- list()
    for (spk in names(prep)) {
      s <- prep[[spk]]$starts
      starts <- s[round(seq(1, length(s), length.out = n_cuts))]
      sig <- prep[[spk]]$sig

      clean <- matrix(0, nrow = n_cuts, ncol = seg_len)
      for (i in seq_len(n_cuts))
        clean[i, ] <- sig[(starts[i] + 1):(starts[i] + seg_len)]

      E <- embed_l2(clean)
      clean_tabs[[spk]] <- data.frame(speaker = spk, E, stringsAsFactors = FALSE)
      noisy_tabs[[length(noisy_tabs) + 1]] <-
        data.frame(speaker = spk, E, stringsAsFactors = FALSE)

      for (k in seq_len(nrow(conditions))) {
        noisy <- clean
        for (i in seq_len(n_cuts))
          noisy[i, ] <- mix_at_snr(clean[i, ],
                                   gen_noise(seg_len, conditions$type[k]),
                                   conditions$snr[k])
        noisy_tabs[[length(noisy_tabs) + 1]] <-
          data.frame(speaker = spk, embed_l2(noisy), stringsAsFactors = FALSE)
      }
    }

    for (kind in c("clean", "noisy")) {
      d <- do.call(rbind, if (kind == "clean") clean_tabs else noisy_tabs)
      rownames(d) <- NULL
      f <- file.path(out_dir, sprintf("embeddings_%s_%s.csv", story, kind))
      write.csv(d, f, row.names = FALSE)
      cat(sprintf("  wrote %-28s %5d rows x %d cols\n", basename(f), nrow(d), ncol(d)))
    }
  }
}

## ---- load the CSVs; rebuilding from audio is opt-in -------------------------
# Rebuilding is never automatic: it needs Python and takes minutes, and a
# missing file is far more often a wrong working directory than a real request
# to regenerate.

rebuild <- TRUE      # set TRUE to regenerate the CSVs from recordings/

train_csv <- file.path(out_dir, "embeddings_EN_clean.csv")
test_csv  <- file.path(out_dir, "embeddings_UN_clean.csv")

if (rebuild) build_embeddings()

if (!file.exists(train_csv) || !file.exists(test_csv))
  stop("embedding CSVs not found in '", out_dir, "'\n",
       "  working directory is: ", getwd(), "\n",
       "  put embeddings_EN_clean.csv and embeddings_UN_clean.csv there,\n",
       "  or set  rebuild <- TRUE  at the bottom of data_preparation.R",
       call. = FALSE)

tr <- read.csv(train_csv)    # training: english narration
te <- read.csv(test_csv)     # test:     urdu narration
