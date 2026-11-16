#!/usr/bin/env Rscript
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD


suppressPackageStartupMessages({
  library(data.table)
  library(Rsamtools)
  library(GenomicAlignments)
  library(riboWaltz)
})

args <- commandArgs(trailingOnly = TRUE)
quiet <- any(args %in% c("--quiet", "--quite"))
args <- args[!(args %in% c("--quiet", "--quite"))]

if (length(args) != 5L) {
  stop("Usage: Rscript riboseq_qc_2.R <bam> <annotation_saf> <se|pe> <sample_out_tsv> <length_qc_out_tsv> [--quiet]")
}

bam_path <- args[[1]]
annotation_path <- args[[2]]
layout <- tolower(args[[3]])
sample_out_path <- args[[4]]
length_out_path <- args[[5]]

if (!(layout %in% c("se", "pe"))) {
  stop("Third argument must be 'se' or 'pe'")
}

QUALITY_N_VALID <- 1000L
QUALITY_ASSIGNABLE_CUTOFF <- 0.70
QUALITY_OFFSET_RANGE_CUTOFF <- 3
QUALITY_FRAME0_CUTOFF <- 0.55
QUALITY_FRAME_DELTA_CUTOFF <- 0.10

CDS_RATIO_CUTOFF <- 0.70
COVERAGE_CUTOFF <- 0.10

log_msg <- function(...) {
  if (!quiet) {
    cat(..., "\n", file = stderr(), sep = "")
    flush(stderr())
  }
}

quiet_ribowaltz <- function(expr) {
  if (quiet) {
    return(suppressWarnings(suppressMessages(force(expr))))
  }

  out_tc <- textConnection(".__rw_out__", "w", local = TRUE)
  msg_tc <- textConnection(".__rw_msg__", "w", local = TRUE)
  sink(out_tc)
  sink(msg_tc, type = "message")
  on.exit({
    sink(type = "message")
    sink()
    close(out_tc)
    close(msg_tc)
  }, add = TRUE)

  suppressWarnings(suppressMessages(force(expr)))
}

round_dt_numeric <- function(dt, digits = 4L) {
  dt <- as.data.table(copy(dt))
  num_cols <- names(dt)[vapply(dt, is.numeric, logical(1))]
  for (col in num_cols) {
    dt[, (col) := as.numeric(round(get(col), digits))]
  }
  dt
}

write_out_horizontal <- function(path, row) {
  fwrite(round_dt_numeric(as.data.table(row), digits = 4L), path, sep = "\t", quote = FALSE, na = "")
}

write_out_vertical <- function(path, row) {
  dt <- round_dt_numeric(as.data.table(row), digits = 4L)
  dt_vertical <- melt(dt,
                      measure.vars = names(dt),
                      variable.name = "metric",
                      value.name = "value")
  dt_vertical[, value := as.character(value)]
  fwrite(dt_vertical, path, sep = "\t", quote = FALSE, na = "", col.names = FALSE)
}

load_annotation <- function(path) {
  dt <- fread(path, sep = "\t")
  req <- c("tx_id", "tx_len", "cds_start", "cds_end", "valid_start", "valid_end")
  stopifnot(all(req %in% names(dt)))
  setDT(dt)
  dt[, `:=`(
    tx_id = as.character(tx_id),
    tx_len = as.integer(tx_len),
    cds_start = as.integer(cds_start),
    cds_end = as.integer(cds_end),
    valid_start = as.integer(valid_start),
    valid_end = as.integer(valid_end)
  )]
  dt
}

is_primary_flag <- function(flag) {
  (bitwAnd(flag, 0x4L) == 0L) &
    (bitwAnd(flag, 0x100L) == 0L) &
    (bitwAnd(flag, 0x800L) == 0L)
}

read_bam_table <- function(bam_path) {
  param <- ScanBamParam(what = c("qname", "rname", "pos", "cigar", "flag"))
  x <- scanBam(bam_path, param = param)[[1]]
  data.table(
    qname = as.character(x$qname),
    tx_id = as.character(x$rname),
    pos = as.integer(x$pos),
    cigar = as.character(x$cigar),
    flag = as.integer(x$flag)
  )
}

prepare_alignment_table <- function(bam_dt, annot) {
  if (nrow(bam_dt) == 0L) {
    return(data.table(
      qname = character(), tx_id = character(), pos = integer(), end = integer(), flag = integer(),
      tx_len = integer(), valid_start = integer(), valid_end = integer()
    ))
  }
  bam_dt <- bam_dt[is_primary_flag(flag)]
  if (nrow(bam_dt) == 0L) return(bam_dt[0])
  bam_dt[, end := as.integer(pos + cigarWidthAlongReferenceSpace(cigar) - 1L)]
  bam_dt <- annot[bam_dt, on = .(tx_id), nomatch = 0L]
  bam_dt <- bam_dt[!is.na(tx_len) & pos >= 1L & end >= pos & end <= tx_len]
  bam_dt
}

scan_counts_se <- function(aln_dt) {
  total_count_by_len <- integer()
  cds_count_by_len <- integer()
  observed_lengths <- integer()

  if (nrow(aln_dt) > 0L) {
    aln_dt[, len := as.integer(end - pos + 1L)]
    observed_lengths <- sort(unique(aln_dt$len))
    total_count_by_len <- tabulate(aln_dt$len, nbins = max(observed_lengths, 0L))
    cds_dt <- aln_dt[pos >= valid_start & end <= valid_end]
    if (nrow(cds_dt) > 0L) {
      cds_count_by_len <- tabulate(cds_dt$len, nbins = max(observed_lengths, 0L))
    }
  }

  list(
    total_count_by_len = total_count_by_len,
    cds_count_by_len = cds_count_by_len,
    observed_lengths = observed_lengths
  )
}

scan_counts_pe <- function(aln_dt) {
  total_count_by_len <- integer()
  cds_count_by_len <- integer()
  observed_lengths <- integer()

  if (nrow(aln_dt) > 0L) {
    aln_dt <- aln_dt[(bitwAnd(flag, 0x1L) != 0L) & (bitwAnd(flag, 0x8L) == 0L)]
  }
  if (nrow(aln_dt) == 0L) {
    return(list(
      total_count_by_len = total_count_by_len,
      cds_count_by_len = cds_count_by_len,
      observed_lengths = observed_lengths
    ))
  }

  pair_dt <- aln_dt[, if (.N == 2L) .(
    tx1 = tx_id[1L], tx2 = tx_id[2L], start1 = pos[1L], start2 = pos[2L], end1 = end[1L], end2 = end[2L],
    valid_start = valid_start[1L], valid_end = valid_end[1L]
  ), by = qname]
  if (nrow(pair_dt) == 0L) {
    return(list(
      total_count_by_len = total_count_by_len,
      cds_count_by_len = cds_count_by_len,
      observed_lengths = observed_lengths
    ))
  }

  pair_dt <- pair_dt[tx1 == tx2]
  if (nrow(pair_dt) == 0L) {
    return(list(
      total_count_by_len = total_count_by_len,
      cds_count_by_len = cds_count_by_len,
      observed_lengths = observed_lengths
    ))
  }

  pair_dt[, `:=`(frag_start = pmin(start1, start2), frag_end = pmax(end1, end2))]
  pair_dt[, frag_len := as.integer(frag_end - frag_start + 1L)]
  observed_lengths <- sort(unique(pair_dt$frag_len))
  total_count_by_len <- tabulate(pair_dt$frag_len, nbins = max(observed_lengths, 0L))

  cds_dt <- pair_dt[frag_start >= valid_start & frag_end <= valid_end]
  if (nrow(cds_dt) > 0L) {
    cds_count_by_len <- tabulate(cds_dt$frag_len, nbins = max(observed_lengths, 0L))
  }

  list(
    total_count_by_len = total_count_by_len,
    cds_count_by_len = cds_count_by_len,
    observed_lengths = observed_lengths
  )
}

make_rw_annotation <- function(annot) {
  data.table(
    transcript = annot$tx_id,
    l_tr = as.integer(annot$tx_len),
    l_utr5 = as.integer(pmax(annot$cds_start - 1L, 0L)),
    l_cds = as.integer(pmax(annot$cds_end - annot$cds_start + 1L, 0L)),
    l_utr3 = as.integer(pmax(annot$tx_len - annot$cds_end, 0L))
  )
}

make_valid_window <- function(annot) {
  data.table(
    transcript = annot$tx_id,
    valid_start = as.integer(annot$valid_start),
    valid_end = as.integer(annot$valid_end)
  )
}

infer_sample_name <- function(bam_path) {
  sub("\\.bam$", "", basename(bam_path))
}

make_bam_dir <- function(bam_path) {
  d <- tempfile("ribowaltz_bam_")
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  dest <- file.path(d, basename(bam_path))
  ok <- suppressWarnings(file.symlink(normalizePath(bam_path), dest))
  if (!isTRUE(ok)) file.copy(bam_path, dest, overwrite = TRUE)
  d
}

pick_offset_column <- function(dt, best_extremity) {
  nm <- names(dt)
  num_cols <- nm[vapply(dt, is.numeric, logical(1))]
  corrected <- num_cols[grepl("corr|correct", num_cols, ignore.case = TRUE)]
  hit <- if (best_extremity == "5end") corrected[grepl("5", corrected)] else corrected[grepl("3", corrected)]
  if (length(hit) > 0L) return(hit[[1]])
  fallback <- num_cols[grepl("offset|po", num_cols, ignore.case = TRUE)]
  hit2 <- if (best_extremity == "5end") fallback[grepl("5", fallback)] else fallback[grepl("3", fallback)]
  if (length(hit2) > 0L) return(hit2[[1]])
  if (length(fallback) > 0L) return(fallback[[1]])
  NA_character_
}

pick_psite_columns <- function(psite_sample) {
  nm <- names(psite_sample)
  low <- tolower(nm)
  transcript_col <- nm[grepl("^transcript$|transcript", low)][1]
  region_col <- nm[grepl("^region$|region", low)][1]
  cds_col <- nm[grepl("psite", low) & grepl("start", low)][1]
  tx_candidates <- nm[grepl("^psite$", low) | (grepl("psite", low) & !grepl("start|stop|5|3|offset|corr|length|reads?", low))]
  tx_col <- tx_candidates[1]
  list(transcript_col = transcript_col, region_col = region_col, tx_col = tx_col, cds_col = cds_col)
}

normalize_lengths <- function(lengths) {
  as.integer(sort(unique(lengths[is.finite(lengths) & lengths >= 1L])))
}

compute_empirical_peak_cluster <- function(total_count_by_len, observed_lengths) {
  observed_lengths <- normalize_lengths(observed_lengths)

  if (length(observed_lengths) == 0L || length(total_count_by_len) == 0L) {
    return(list(
      provisional_center = NA_integer_,
      peak_length = NA_integer_,
      peak_count = 0L,
      cluster_lengths = integer(),
      cluster_min = NA_integer_,
      cluster_max = NA_integer_
    ))
  }

  observed_lengths <- observed_lengths[observed_lengths <= length(total_count_by_len)]
  if (length(observed_lengths) == 0L) {
    return(list(
      provisional_center = NA_integer_,
      peak_length = NA_integer_,
      peak_count = 0L,
      cluster_lengths = integer(),
      cluster_min = NA_integer_,
      cluster_max = NA_integer_
    ))
  }

  count_dt <- data.table(
    length = observed_lengths,
    n_reads = as.integer(total_count_by_len[observed_lengths])
  )
  count_dt <- count_dt[order(length)]

  count_dt[, `:=`(
    n_reads_lag = c(NA_real_, head(as.numeric(n_reads), -1L)),
    n_reads_lead = c(tail(as.numeric(n_reads), -1L), NA_real_)
  )]
  count_dt[, ma3 := (n_reads_lag + as.numeric(n_reads) + n_reads_lead) / 3]

  ma3_dt <- count_dt[is.finite(ma3)]
  if (nrow(ma3_dt) == 0L) {
    peak_idx <- which.max(count_dt$n_reads)
    peak_length <- count_dt$length[[peak_idx]]
    peak_count <- count_dt$n_reads[[peak_idx]]
    return(list(
      provisional_center = peak_length,
      peak_length = peak_length,
      peak_count = peak_count,
      cluster_lengths = peak_length,
      cluster_min = peak_length,
      cluster_max = peak_length
    ))
  }

  provisional_center <- ma3_dt[which.max(ma3)]$length[[1]]

  local_lengths <- count_dt[length >= (provisional_center - 1L) & length <= (provisional_center + 1L)]
  if (nrow(local_lengths) == 0L) {
    local_lengths <- count_dt[length == provisional_center]
  }
  peak_row <- local_lengths[which.max(n_reads)]
  peak_length <- peak_row$length[[1]]
  peak_count <- as.integer(peak_row$n_reads[[1]])

  threshold <- 0.25 * peak_count
  count_dt[, above_threshold := n_reads >= threshold]

  peak_pos <- which(count_dt$length == peak_length)[1]
  if (!is.finite(peak_pos)) {
    return(list(
      provisional_center = provisional_center,
      peak_length = peak_length,
      peak_count = peak_count,
      cluster_lengths = integer(),
      cluster_min = NA_integer_,
      cluster_max = NA_integer_
    ))
  }

  left <- peak_pos
  while (left > 1L &&
         count_dt$above_threshold[[left - 1L]] &&
         (count_dt$length[[left]] - count_dt$length[[left - 1L]] == 1L)) {
    left <- left - 1L
  }

  right <- peak_pos
  while (right < nrow(count_dt) &&
         count_dt$above_threshold[[right + 1L]] &&
         (count_dt$length[[right + 1L]] - count_dt$length[[right]] == 1L)) {
    right <- right + 1L
  }

  cluster_lengths <- as.integer(count_dt$length[left:right])

  list(
    provisional_center = provisional_center,
    peak_length = peak_length,
    peak_count = peak_count,
    cluster_lengths = cluster_lengths,
    cluster_min = if (length(cluster_lengths) > 0L) min(cluster_lengths) else NA_integer_,
    cluster_max = if (length(cluster_lengths) > 0L) max(cluster_lengths) else NA_integer_
  )
}

tab_sum_by_lengths <- function(counts, lengths) {
  lengths <- normalize_lengths(lengths)
  lengths <- lengths[lengths <= length(counts)]
  if (length(lengths) == 0L) return(0L)
  sum(counts[lengths])
}

tab_nt_sum_by_lengths <- function(counts, lengths) {
  lengths <- normalize_lengths(lengths)
  lengths <- lengths[lengths <= length(counts)]
  if (length(lengths) == 0L) return(0)
  sum(lengths * counts[lengths])
}

evaluate_interval_qc <- function(reads_list, sample_id, valid_window_dt, lengths) {
  lengths <- normalize_lengths(lengths)

  row <- data.table(
    n_reads = 0L,
    n_valid_psites = 0L,
    assignable_fraction = NA_real_,
    dominant_offset = NA_real_,
    offset_range = NA_real_,
    frame0_pct = NA_real_,
    frame1_pct = NA_real_,
    frame2_pct = NA_real_,
    frame0_delta = NA_real_,
    quality_pass = 0L,
    interval_pass = 0L,
    quality_reason = "unknown"
  )

  if (length(lengths) == 0L) {
    row$quality_reason <- "no_selected_lengths"
    return(row)
  }

  reads_filtered <- quiet_ribowaltz(length_filter(
    data = reads_list,
    sample = sample_id,
    length_filter_mode = "custom",
    length_range = lengths,
    output_class = "datatable"
  ))

  n_before <- nrow(reads_filtered[[sample_id]])
  row$n_reads <- as.integer(n_before)
  if (n_before == 0L) {
    row$quality_reason <- "no_reads_after_length_filter"
    return(row)
  }

  offset_dt <- tryCatch(
    quiet_ribowaltz(as.data.table(psite(data = reads_filtered, extremity = "5end", plot = FALSE, txt = FALSE))),
    error = function(e) NULL
  )
  if (is.null(offset_dt) || nrow(offset_dt) == 0L) {
    row$quality_reason <- "psite_failed_or_empty"
    return(row)
  }

  sample_col <- names(offset_dt)[grepl("sample", names(offset_dt), ignore.case = TRUE)][1]
  if (is.na(sample_col) || !nzchar(sample_col)) {
    row$quality_reason <- "no_offset_sample_column"
    return(row)
  }

  offset_sample <- offset_dt[get(sample_col) == sample_id]
  if (nrow(offset_sample) == 0L) {
    row$quality_reason <- "empty_offset_table"
    return(row)
  }

  offset_col <- pick_offset_column(offset_sample, "5end")

  if (is.na(offset_col) || !nzchar(offset_col)) {
    row$assignable_fraction <- 0
    row$quality_reason <- "no_offset_signal"
    return(row)
  }

  offset_values <- as.numeric(offset_sample[[offset_col]])
  offset_weights <- if ("perc_reads" %in% names(offset_sample)) offset_sample$perc_reads else rep(1, nrow(offset_sample))
  keep_offset <- is.finite(offset_values)
  offset_values <- offset_values[keep_offset]
  offset_weights <- offset_weights[keep_offset]

  if (length(offset_values) == 0L) {
    row$assignable_fraction <- 0
    row$quality_reason <- "no_offset_signal"
    return(row)
  }

  reads_psite <- tryCatch(
    quiet_ribowaltz(psite_info(data = reads_filtered, offset = offset_dt, output_class = "datatable")),
    error = function(e) NULL
  )
  if (is.null(reads_psite) || is.null(reads_psite[[sample_id]])) {
    row$assignable_fraction <- 0
    row$dominant_offset <- as.numeric(stats::weighted.mean(offset_values, offset_weights))
    row$offset_range <- as.numeric(max(offset_values) - min(offset_values))
    row$quality_reason <- "psite_info_failed_or_empty"
    return(row)
  }

  n_after <- nrow(reads_psite[[sample_id]])
  assignable_fraction_val <- n_after / n_before

  psite_sample <- as.data.table(reads_psite[[sample_id]])
  cols <- pick_psite_columns(psite_sample)
  if (any(is.na(unlist(cols)))) {
    row$assignable_fraction <- assignable_fraction_val
    row$dominant_offset <- as.numeric(stats::weighted.mean(offset_values, offset_weights))
    row$offset_range <- max(offset_values) - min(offset_values)
    row$quality_reason <- "cannot_find_psite_columns"
    return(row)
  }

  setnames(psite_sample, cols$transcript_col, "transcript")
  setnames(psite_sample, cols$region_col, "region")
  setnames(psite_sample, cols$tx_col, "psite_tx")
  setnames(psite_sample, cols$cds_col, "psite_cds")

  psite_sample[, region := tolower(as.character(region))]
  cds_rows <- psite_sample[region == "cds"]
  cds_rows[, `:=`(
    psite_tx = as.numeric(psite_tx),
    psite_cds = as.numeric(psite_cds)
  )]
  cds_rows <- merge(cds_rows, valid_window_dt, by = "transcript", all.x = FALSE, all.y = FALSE)
  cds_rows <- cds_rows[is.finite(psite_tx) & psite_tx >= valid_start & psite_tx <= valid_end]

  row$n_valid_psites <- as.integer(nrow(cds_rows))
  if (nrow(cds_rows) == 0L) {
    row$assignable_fraction <- assignable_fraction_val
    row$dominant_offset <- as.numeric(stats::weighted.mean(offset_values, offset_weights))
    row$offset_range <- max(offset_values) - min(offset_values)
    row$quality_reason <- "no_valid_window_psites"
    return(row)
  }

  start_vals <- cds_rows$psite_cds
  start_vals <- start_vals[is.finite(start_vals)]
  if (length(start_vals) == 0L) {
    row$assignable_fraction <- assignable_fraction_val
    row$dominant_offset <- as.numeric(stats::weighted.mean(offset_values, offset_weights))
    row$offset_range <- max(offset_values) - min(offset_values)
    row$quality_reason <- "no_valid_window_psites"
    return(row)
  }

  frames <- start_vals %% 3L
  frame_tab <- tabulate(frames + 1L, nbins = 3L)
  frame_frac <- frame_tab / sum(frame_tab)

  dominant_offset_val <- as.numeric(stats::weighted.mean(offset_values, offset_weights))
  offset_range_val <- as.numeric(max(offset_values) - min(offset_values))
  frame0_pct_val <- as.numeric(frame_frac[[1]])
  frame1_pct_val <- as.numeric(frame_frac[[2]])
  frame2_pct_val <- as.numeric(frame_frac[[3]])
  frame0_delta_val <- as.numeric(frame_frac[[1]] - max(frame_frac[[2]], frame_frac[[3]]))
  row[, `:=`(
    assignable_fraction = assignable_fraction_val,
    dominant_offset = dominant_offset_val,
    offset_range = offset_range_val,
    frame0_pct = frame0_pct_val,
    frame1_pct = frame1_pct_val,
    frame2_pct = frame2_pct_val,
    frame0_delta = frame0_delta_val
  )]

  quality_reasons <- character()
  if (!is.finite(row$n_valid_psites) || row$n_valid_psites < QUALITY_N_VALID) quality_reasons <- c(quality_reasons, "low_valid_psites")
  if (!is.finite(row$assignable_fraction) || row$assignable_fraction < QUALITY_ASSIGNABLE_CUTOFF) quality_reasons <- c(quality_reasons, "low_assignable_fraction")
  if (!is.finite(row$offset_range) || row$offset_range > QUALITY_OFFSET_RANGE_CUTOFF) quality_reasons <- c(quality_reasons, "high_offset_range")
  if (!is.finite(row$frame0_pct) || row$frame0_pct < QUALITY_FRAME0_CUTOFF) quality_reasons <- c(quality_reasons, "low_frame0")
  if (!is.finite(row$frame0_delta) || row$frame0_delta < QUALITY_FRAME_DELTA_CUTOFF) quality_reasons <- c(quality_reasons, "low_frame_separation")

  row$quality_pass <- as.integer(length(quality_reasons) == 0L)
  row$interval_pass <- row$quality_pass
  row$quality_reason <- if (row$quality_pass == 1L) {
    "pass"
  } else {
    paste(quality_reasons, collapse = ";")
  }

  row
}

compute_selected_group_metrics <- function(total_count_by_len, cds_count_by_len, selected_lengths, total_tx_len) {
  selected_lengths <- normalize_lengths(selected_lengths)

  total_primary_tx_reads <- sum(total_count_by_len)
  cds_primary_reads <- sum(cds_count_by_len)
  filtered_primary_tx_reads <- tab_sum_by_lengths(total_count_by_len, selected_lengths)
  filtered_cds_reads <- tab_sum_by_lengths(cds_count_by_len, selected_lengths)
  filtered_nt <- tab_nt_sum_by_lengths(total_count_by_len, selected_lengths)

  cds_ratio <- if (filtered_primary_tx_reads > 0L) filtered_cds_reads / filtered_primary_tx_reads else 0
  coverage_tx <- if (total_tx_len > 0L) filtered_nt / total_tx_len else 0
  coverage_pass <- as.integer(cds_ratio >= CDS_RATIO_CUTOFF && coverage_tx >= COVERAGE_CUTOFF)

  list(
    total_primary_tx_reads = as.integer(total_primary_tx_reads),
    cds_primary_reads = as.integer(cds_primary_reads),
    filtered_primary_tx_reads = as.integer(filtered_primary_tx_reads),
    filtered_cds_reads = as.integer(filtered_cds_reads),
    cds_ratio = cds_ratio,
    coverage_tx = coverage_tx,
    coverage_pass = coverage_pass
  )
}

annot <- load_annotation(annotation_path)
sample_id <- infer_sample_name(bam_path)
annotation_dt <- make_rw_annotation(annot)
valid_window_dt <- make_valid_window(annot)
total_tx_len <- sum(annot$tx_len)

log_msg("[1/7] Reading BAM for raw length counts: ", bam_path)
bam_dt <- read_bam_table(bam_path)
aln_dt <- prepare_alignment_table(bam_dt, annot)
scan_res <- if (layout == "se") scan_counts_se(aln_dt) else scan_counts_pe(aln_dt)

observed_lengths <- normalize_lengths(scan_res$observed_lengths)
if (length(observed_lengths) == 0L) {
  observed_lengths <- integer()
}

log_msg("[2/7] Reading BAM with riboWaltz: ", bam_path)
bam_dir <- make_bam_dir(bam_path)
on.exit(unlink(bam_dir, recursive = TRUE), add = TRUE)
bam_base <- infer_sample_name(bam_path)
name_samples <- setNames(sample_id, bam_base)
reads_list <- quiet_ribowaltz(bamtolist(
  bamfolder = bam_dir,
  annotation = annotation_dt,
  transcript_align = TRUE,
  name_samples = name_samples,
  output_class = "datatable"
))

log_msg("[3/7] Evaluating all observed read lengths")
length_rows <- vector("list", length(observed_lengths))
for (i in seq_along(observed_lengths)) {
  L <- observed_lengths[[i]]
  qc <- evaluate_interval_qc(reads_list, sample_id, valid_window_dt, L)
  length_rows[[i]] <- data.table(length = L, qc)
}

if (length(length_rows) > 0L) {
  length_tbl <- rbindlist(length_rows, fill = TRUE)
} else {
  length_tbl <- data.table(
    length = integer(),
    n_reads = integer(),
    n_valid_psites = integer(),
    assignable_fraction = numeric(),
    dominant_offset = numeric(),
    offset_range = numeric(),
    frame0_pct = numeric(),
    frame1_pct = numeric(),
    frame2_pct = numeric(),
    frame0_delta = numeric(),
    quality_pass = integer(),
    interval_pass = integer(),
    quality_reason = character()
  )
}

length_tbl[, selection_class := fifelse(
  quality_pass == 1L, "pass", "reject"
)]

log_msg("[4/7] Selecting dominant passing length")
cand <- copy(length_tbl[quality_pass == 1L])
empirical_peak <- compute_empirical_peak_cluster(
  total_count_by_len = scan_res$total_count_by_len,
  observed_lengths = observed_lengths
)

sample_status <- "fail"
sample_reason <- "no_passing_lengths"
dominant_length <- NA_integer_
selected_min <- NA_integer_
selected_max <- NA_integer_
selected_lengths <- integer()
selected_lengths_chr <- ""

if (nrow(cand) > 0L) {
  center <- stats::median(observed_lengths)
  cand[, tie_center := abs(length - center)]
  setorder(cand, -n_valid_psites, -frame0_delta, -assignable_fraction, tie_center)
  dominant_length <- cand$length[[1]]
  dom_offset <- cand$dominant_offset[[1]]

  pass_tbl <- copy(length_tbl[quality_pass == 1L])
  block_tbl <- if (is.finite(dom_offset)) {
    pass_tbl[is.finite(dominant_offset) & abs(dominant_offset - dom_offset) <= 1]
  } else {
    pass_tbl[is.na(dominant_offset)]
  }
  block_tbl <- block_tbl[order(length)]

  if (nrow(block_tbl) > 0L) {
    block_tbl[, grp := cumsum(c(TRUE, diff(length) != 1L))]
    dom_grp <- block_tbl[length == dominant_length, grp][1]
    block_tbl <- block_tbl[grp == dom_grp]
    selected_lengths <- normalize_lengths(block_tbl$length)
    selected_lengths_chr <- paste(selected_lengths, collapse = ",")
    selected_min <- min(selected_lengths)
    selected_max <- max(selected_lengths)

    if (length(selected_lengths) >= 2L) {
      sample_status <- "pass"
      sample_reason <- "candidate_block_selected"
    } else if (length(selected_lengths) == 1L) {
      if (selected_lengths[[1]] %in% empirical_peak$cluster_lengths) {
        sample_status <- "pass"
        sample_reason <- "single_candidate_selected"
      } else {
        sample_status <- "fail"
        sample_reason <- "invalid_selection"
      }
    }
  } else {
    sample_reason <- "offset_inconsistent_lengths"
  }
}

log_msg("[5/7] Computing final block-level QC metrics")
sample_row <- data.table(
  sample_id = sample_id,
  bam = bam_path,
  dominant_length = dominant_length,
  selected_min = selected_min,
  selected_max = selected_max,
  selected_lengths = selected_lengths_chr,
  sample_status = sample_status,
  selection_reason = sample_reason,
  total_primary_tx_reads = 0L,
  cds_primary_reads = 0L,
  filtered_primary_tx_reads = 0L,
  filtered_cds_reads = 0L,
  cds_ratio = 0,
  coverage_tx = 0,
  coverage_pass = 0L,
  assignable_fraction = NA_real_,
  dominant_offset = NA_real_,
  offset_range = NA_real_,
  frame0_pct = NA_real_,
  frame1_pct = NA_real_,
  frame2_pct = NA_real_,
  frame0_delta = NA_real_,
  quality_pass = 0L,
  final_pass = 0L,
  final_reason = sample_reason
)

if (length(selected_lengths) > 0L) {
  metrics <- compute_selected_group_metrics(
    scan_res$total_count_by_len,
    scan_res$cds_count_by_len,
    selected_lengths,
    total_tx_len
  )
  sample_row[, `:=`(
    total_primary_tx_reads = metrics$total_primary_tx_reads,
    cds_primary_reads = metrics$cds_primary_reads,
    filtered_primary_tx_reads = metrics$filtered_primary_tx_reads,
    filtered_cds_reads = metrics$filtered_cds_reads,
    cds_ratio = metrics$cds_ratio,
    coverage_tx = metrics$coverage_tx,
    coverage_pass = metrics$coverage_pass
  )]

  block_qc <- evaluate_interval_qc(reads_list, sample_id, valid_window_dt, selected_lengths)
  sample_row[, `:=`(
    assignable_fraction = block_qc$assignable_fraction,
    dominant_offset = block_qc$dominant_offset,
    offset_range = block_qc$offset_range,
    frame0_pct = block_qc$frame0_pct,
    frame1_pct = block_qc$frame1_pct,
    frame2_pct = block_qc$frame2_pct,
    frame0_delta = block_qc$frame0_delta,
    quality_pass = block_qc$quality_pass
  )]

  final_pass_val <- as.integer(
    sample_status == "pass" &&
      block_qc$interval_pass == 1L &&
      metrics$coverage_pass == 1L
  )

  final_reason_val <- if (sample_status != "pass") {
    sample_reason
  } else if (block_qc$interval_pass != 1L) {
    block_qc$quality_reason
  } else if (metrics$coverage_pass != 1L) {
    if (metrics$cds_ratio < CDS_RATIO_CUTOFF && metrics$coverage_tx < COVERAGE_CUTOFF) {
      "low_cds_ratio;low_coverage"
    } else if (metrics$cds_ratio < CDS_RATIO_CUTOFF) {
      "low_cds_ratio"
    } else {
      "low_coverage"
    }
  } else {
    "pass"
  }

  sample_row[, `:=`(
    final_pass = final_pass_val,
    final_reason = final_reason_val
  )]
}

log_msg("[6/7] Writing outputs")
write_out_horizontal(length_out_path, length_tbl)
write_out_vertical(sample_out_path, sample_row)

log_msg("[7/7] Done")
log_msg(
  "Sample ", sample_id,
  ": dominant_length=", sample_row$dominant_length,
  " selected_block=", sample_row$selected_min, "-", sample_row$selected_max,
  " selected_lengths=", sample_row$selected_lengths,
  " status=", sample_row$sample_status,
  " quality_pass=", sample_row$quality_pass,
  " coverage_pass=", sample_row$coverage_pass,
  " final_pass=", sample_row$final_pass,
  " reason=", sample_row$final_reason
)


