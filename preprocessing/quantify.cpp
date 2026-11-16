// Program: DuET v1.1.0
// Author: Sungho Lee, Jae-Won Lee
// Affiliation: MOGAM Institute for Biomedical Research
// Contact: https://github.com/mogam-ai/DuET/issues
// Citation: TBD


#include <htslib/hts.h>
#include <htslib/sam.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits.h>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Window {
    int64_t tx_len = -1;
    int64_t cds_start = -1;
    int64_t cds_end = -1;
    int64_t valid_start = -1;
    int64_t valid_end = -1;
    bool has = false;
    std::string tx_id;
};

struct CountRow {
    uint64_t count = 0;
};

struct Args {
    std::string bam_path;
    std::string layout;
    std::string saf_path;
    std::string length_spec;
    int threads = 1;
    bool use_all_lengths = false;
    std::unordered_set<int64_t> allowed_lengths;
    bool quiet = false;
};

struct Stats {
    uint64_t total_records = 0;
    uint64_t total_primary = 0;
    uint64_t total_layout_kept = 0;
    uint64_t total_forward_kept = 0;
    uint64_t total_in_tx_bounds = 0;
    uint64_t total_qc_length = 0;
    uint64_t total_valid_5p = 0;
    uint64_t total_out_of_tx_bounds = 0;
    uint64_t total_tx_len_mismatch = 0;
};

static inline void log_msg(const Args& args, const std::string& msg) {
    if (!args.quiet) {
        std::cerr << msg << '\n';
    }
}

static inline bool is_primary_mapped(const bam1_t* b) {
    const uint16_t flag = b->core.flag;
    if (flag & BAM_FUNMAP) return false;
    if (flag & BAM_FSECONDARY) return false;
    if (flag & BAM_FSUPPLEMENTARY) return false;
    return true;
}

static inline int64_t ref_span_end_1based(const bam1_t* b) {
    const uint32_t* cigar = bam_get_cigar(b);
    const int64_t rlen = bam_cigar2rlen(b->core.n_cigar, cigar);
    return static_cast<int64_t>(b->core.pos) + rlen;
}

static inline int64_t query_length(const bam1_t* b) {
    const uint32_t* cigar = bam_get_cigar(b);
    return static_cast<int64_t>(bam_cigar2qlen(b->core.n_cigar, cigar));
}

static inline int64_t parse_i64(const std::string& s) {
    return std::strtoll(s.c_str(), nullptr, 10);
}

static inline std::string trim_copy(const std::string& s) {
    size_t beg = 0;
    while (beg < s.size() && std::isspace(static_cast<unsigned char>(s[beg]))) ++beg;
    size_t end = s.size();
    while (end > beg && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(beg, end - beg);
}

static std::string shell_quote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') {
            out += "'\"'\"'";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

static std::string absolute_path_or_die(const std::string& path) {
    char* resolved = realpath(path.c_str(), nullptr);
    if (!resolved) {
        std::cerr << "[error] failed to resolve absolute path: " << path << "\n";
        std::exit(1);
    }
    std::unique_ptr<char, decltype(&std::free)> holder(resolved, &std::free);
    return std::string(holder.get());
}

static std::string executable_path_or_fallback(const char* argv0) {
    char exe_buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    if (n > 0) {
        exe_buf[n] = '\0';
        return std::string(exe_buf);
    }

    char* resolved = realpath(argv0, nullptr);
    if (resolved) {
        std::unique_ptr<char, decltype(&std::free)> holder(resolved, &std::free);
        return std::string(holder.get());
    }

    return std::string(argv0);
}

static void parse_length_spec(Args& args) {
    const std::string spec = trim_copy(args.length_spec);
    if (spec == "-") {
        args.use_all_lengths = true;
        return;
    }

    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim_copy(tok);
        if (tok.empty()) {
            std::cerr << "[error] empty token found in length specification: " << args.length_spec << "\n";
            std::exit(1);
        }
        const int64_t len = parse_i64(tok);
        if (len < 1) {
            std::cerr << "[error] invalid read length in specification: " << tok << "\n";
            std::exit(1);
        }
        args.allowed_lengths.insert(len);
    }

    if (args.allowed_lengths.empty()) {
        std::cerr << "[error] no valid read lengths parsed from specification: " << args.length_spec << "\n";
        std::exit(1);
    }
}

static std::string normalized_length_spec(const Args& args) {
    if (args.use_all_lengths) return "-";
    std::vector<int64_t> lens(args.allowed_lengths.begin(), args.allowed_lengths.end());
    std::sort(lens.begin(), lens.end());
    std::ostringstream oss;
    for (size_t i = 0; i < lens.size(); ++i) {
        if (i) oss << ",";
        oss << lens[i];
    }
    return oss.str();
}

static Args parse_args(int argc, char** argv) {
    std::vector<std::string> pos;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quiet" || a == "--quite") {
            quiet = true;
        } else {
            pos.push_back(a);
        }
    }

    if (pos.size() != 5) {
        std::cerr
            << "Usage: " << argv[0]
            << " <bam_file> <pe|se> <saf_file> <length> <threads> [--quiet]\n";
        std::exit(1);
    }

    Args args;
    args.bam_path = pos[0];
    args.layout = pos[1];
    args.saf_path = pos[2];
    args.length_spec = pos[3];
    args.threads = std::atoi(pos[4].c_str());
    args.quiet = quiet;

    if (!(args.layout == "se" || args.layout == "pe")) {
        std::cerr << "[error] layout must be 'se' or 'pe'\n";
        std::exit(1);
    }
    if (args.threads < 1) {
        std::cerr << "[error] threads must be >= 1\n";
        std::exit(1);
    }

    parse_length_spec(args);
    return args;
}

static std::string reproduce_command(const char* argv0, const Args& args) {
    std::ostringstream oss;
    oss << shell_quote(executable_path_or_fallback(argv0))
        << ' ' << shell_quote(absolute_path_or_die(args.bam_path))
        << ' ' << shell_quote(args.layout)
        << ' ' << shell_quote(absolute_path_or_die(args.saf_path))
        << ' ' << shell_quote(normalized_length_spec(args))
        << ' ' << shell_quote(std::to_string(args.threads));
    if (args.quiet) {
        oss << ' ' << shell_quote("--quiet");
    }
    return oss.str();
}

static void load_saf(
    const Args& args,
    const bam_hdr_t* hdr,
    std::vector<Window>& windows,
    size_t& saf_rows,
    size_t& matched_rows,
    size_t& tx_len_mismatch_rows
) {
    std::unordered_map<std::string, int> ref_to_tid;
    ref_to_tid.reserve(static_cast<size_t>(hdr->n_targets) * 2U);
    for (int i = 0; i < hdr->n_targets; ++i) {
        ref_to_tid[hdr->target_name[i]] = i;
    }

    std::ifstream fin(args.saf_path);
    if (!fin) {
        std::cerr << "[error] failed to open SAF: " << args.saf_path << "\n";
        std::exit(1);
    }

    std::string line;
    if (!std::getline(fin, line)) {
        std::cerr << "[error] empty SAF: " << args.saf_path << "\n";
        std::exit(1);
    }

    std::vector<std::string> header;
    {
        std::stringstream hs(line);
        std::string field;
        while (std::getline(hs, field, '\t')) header.push_back(field);
    }

    auto find_col = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(header.size()); ++i) {
            if (header[i] == name) return i;
        }
        return -1;
    };

    const int idx_tx = find_col("tx_id");
    const int idx_tx_len = find_col("tx_len");
    const int idx_cds_start = find_col("cds_start");
    const int idx_cds_end = find_col("cds_end");
    const int idx_valid_start = find_col("valid_start");
    const int idx_valid_end = find_col("valid_end");

    if (idx_tx < 0 || idx_tx_len < 0 || idx_cds_start < 0 || idx_cds_end < 0 ||
        idx_valid_start < 0 || idx_valid_end < 0) {
        std::cerr << "[error] SAF must contain tx_id, tx_len, cds_start, cds_end, valid_start, valid_end\n";
        std::exit(1);
    }

    saf_rows = 0;
    matched_rows = 0;
    tx_len_mismatch_rows = 0;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::vector<std::string> fields;
        fields.reserve(header.size());
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, '\t')) fields.push_back(field);

        const int need = std::max(
            std::max(std::max(idx_tx, idx_tx_len), std::max(idx_cds_start, idx_cds_end)),
            std::max(idx_valid_start, idx_valid_end)
        );
        if (static_cast<int>(fields.size()) <= need) continue;

        ++saf_rows;

        const std::string& tx_id = fields[idx_tx];
        auto it = ref_to_tid.find(tx_id);
        if (it == ref_to_tid.end()) continue;

        const int tid = it->second;
        Window w;
        w.tx_id = tx_id;
        w.tx_len = parse_i64(fields[idx_tx_len]);
        w.cds_start = parse_i64(fields[idx_cds_start]);
        w.cds_end = parse_i64(fields[idx_cds_end]);
        w.valid_start = parse_i64(fields[idx_valid_start]);
        w.valid_end = parse_i64(fields[idx_valid_end]);

        if (w.tx_len < 1) {
            std::cerr << "[error] invalid tx_len for " << tx_id << "\n";
            std::exit(1);
        }
        if (w.cds_start < 1 || w.cds_end < w.cds_start || w.cds_end > w.tx_len) {
            std::cerr << "[error] invalid CDS interval for " << tx_id << "\n";
            std::exit(1);
        }
        if (w.valid_start < 1 || w.valid_end < w.valid_start || w.valid_end > w.tx_len) {
            std::cerr << "[error] invalid valid window for " << tx_id << "\n";
            std::exit(1);
        }

        const int64_t header_len = static_cast<int64_t>(hdr->target_len[tid]);
        if (header_len > 0 && header_len != w.tx_len) {
            ++tx_len_mismatch_rows;
        }

        w.has = true;
        windows[tid] = std::move(w);
        ++matched_rows;
    }
}

static inline bool in_tx_bounds(int64_t aln_start, int64_t aln_end, const Window& w) {
    return w.has && aln_start >= 1 && aln_end >= aln_start && aln_end <= w.tx_len;
}

static inline bool read_length_allowed(const Args& args, int64_t len) {
    if (len < 1) return false;
    if (args.use_all_lengths) return true;
    return args.allowed_lengths.find(len) != args.allowed_lengths.end();
}

static inline bool is_forward_on_transcript(const bam1_t* b) {
    return (b->core.flag & BAM_FREVERSE) == 0;
}

static inline int64_t five_prime_end_1based(const bam1_t* b) {
    const int64_t aln_start = static_cast<int64_t>(b->core.pos) + 1;
    const int64_t aln_end = ref_span_end_1based(b);
    return is_forward_on_transcript(b) ? aln_start : aln_end;
}

static inline bool fivep_in_valid_window(int64_t fivep, const Window& w) {
    return w.has && fivep >= w.valid_start && fivep <= w.valid_end;
}

static inline void log_progress_se(const Args& args, const Stats& stats) {
    if (!args.quiet) {
        std::cerr << "  processed_records=" << stats.total_records
                  << " primary=" << stats.total_primary
                  << " in_tx_bounds=" << stats.total_in_tx_bounds
                  << " qc_length=" << stats.total_qc_length
                  << " counted=" << stats.total_valid_5p << "\n";
    }
}

static inline void log_progress_pe(const Args& args, const Stats& stats) {
    if (!args.quiet) {
        std::cerr << "  processed_records=" << stats.total_records
                  << " primary=" << stats.total_primary
                  << " paired=" << stats.total_layout_kept
                  << " forward_only=" << stats.total_forward_kept
                  << " in_tx_bounds=" << stats.total_in_tx_bounds
                  << " qc_length=" << stats.total_qc_length
                  << " counted=" << stats.total_valid_5p << "\n";
    }
}

static void count_se(
    samFile* fp,
    bam_hdr_t* hdr,
    const Args& args,
    const std::vector<Window>& windows,
    std::vector<CountRow>& counts,
    Stats& stats
) {
    bam1_t* b = bam_init1();

    while (sam_read1(fp, hdr, b) >= 0) {
        ++stats.total_records;

        if (!is_primary_mapped(b)) continue;
        ++stats.total_primary;

        const int32_t tid = b->core.tid;
        if (tid < 0 || tid >= hdr->n_targets) continue;
        const Window& w = windows[tid];
        if (!w.has) continue;

        const int64_t aln_start = static_cast<int64_t>(b->core.pos) + 1;
        const int64_t aln_end = ref_span_end_1based(b);
        if (!in_tx_bounds(aln_start, aln_end, w)) {
            ++stats.total_out_of_tx_bounds;
            continue;
        }
        ++stats.total_in_tx_bounds;

        const int64_t read_len = query_length(b);
        if (!read_length_allowed(args, read_len)) continue;
        ++stats.total_qc_length;

        const int64_t fivep = five_prime_end_1based(b);
        if (!fivep_in_valid_window(fivep, w)) continue;

        ++counts[tid].count;
        ++stats.total_valid_5p;

        if (!args.quiet && stats.total_records % 5000000ULL == 0ULL) {
            log_progress_se(args, stats);
        }
    }

    if (!args.quiet && stats.total_records > 0 && stats.total_records % 5000000ULL != 0ULL) {
        log_progress_se(args, stats);
    }

    bam_destroy1(b);
}

static void count_pe(
    samFile* fp,
    bam_hdr_t* hdr,
    const Args& args,
    const std::vector<Window>& windows,
    std::vector<CountRow>& counts,
    Stats& stats
) {
    bam1_t* b = bam_init1();

    while (sam_read1(fp, hdr, b) >= 0) {
        ++stats.total_records;

        if (!is_primary_mapped(b)) continue;
        ++stats.total_primary;

        const uint16_t flag = b->core.flag;
        if (!(flag & BAM_FPAIRED)) continue;
        ++stats.total_layout_kept;

        // Use exactly one transcript-compatible direction per pair-like record stream:
        // keep only forward-on-transcript alignments and discard reverse-only observations.
        if (!is_forward_on_transcript(b)) continue;
        ++stats.total_forward_kept;

        const int32_t tid = b->core.tid;
        if (tid < 0 || tid >= hdr->n_targets) continue;
        const Window& w = windows[tid];
        if (!w.has) continue;

        const int64_t aln_start = static_cast<int64_t>(b->core.pos) + 1;
        const int64_t aln_end = ref_span_end_1based(b);
        if (!in_tx_bounds(aln_start, aln_end, w)) {
            ++stats.total_out_of_tx_bounds;
            continue;
        }
        ++stats.total_in_tx_bounds;

        const int64_t read_len = query_length(b);
        if (!read_length_allowed(args, read_len)) continue;
        ++stats.total_qc_length;

        const int64_t fivep = five_prime_end_1based(b);  // for retained forward records, this is aln_start
        if (!fivep_in_valid_window(fivep, w)) continue;

        ++counts[tid].count;
        ++stats.total_valid_5p;

        if (!args.quiet && stats.total_records % 5000000ULL == 0ULL) {
            log_progress_pe(args, stats);
        }
    }

    if (!args.quiet && stats.total_records > 0 && stats.total_records % 5000000ULL != 0ULL) {
        log_progress_pe(args, stats);
    }

    bam_destroy1(b);
}

static void write_output_stdout(
    const bam_hdr_t* hdr,
    const std::vector<Window>& windows,
    const std::vector<CountRow>& counts
) {
    long double total_count = 0.0L;
    for (int tid = 0; tid < hdr->n_targets; ++tid) {
        if (!windows[tid].has) continue;
        total_count += static_cast<long double>(counts[tid].count);
    }

    std::cout << "txID\tcount\tCPM\n";
    std::cout << std::fixed << std::setprecision(6);

    for (int tid = 0; tid < hdr->n_targets; ++tid) {
        if (!windows[tid].has) continue;
        const long double cpm = (total_count > 0.0L)
            ? (static_cast<long double>(counts[tid].count) * 1000000.0L / total_count)
            : 0.0L;

        std::cout << windows[tid].tx_id << '\t'
                  << counts[tid].count << '\t'
                  << static_cast<double>(cpm) << '\n';
    }
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    if (!args.quiet) {
        std::cerr << "$ " << reproduce_command(argv[0], args) << '\n';
    }

    log_msg(args, "[1/5] Opening BAM: " + args.bam_path);
    log_msg(args, "  layout=" + args.layout +
                  " threads=" + std::to_string(args.threads) +
                  " lengths=" + normalized_length_spec(args));

    samFile* fp = sam_open(args.bam_path.c_str(), "r");
    if (!fp) {
        std::cerr << "[error] failed to open BAM: " << args.bam_path << "\n";
        return 1;
    }
    if (args.threads > 1) {
        hts_set_threads(fp, args.threads);
    }

    bam_hdr_t* hdr = sam_hdr_read(fp);
    if (!hdr) {
        std::cerr << "[error] failed to read BAM header\n";
        sam_close(fp);
        return 1;
    }

    std::vector<Window> windows(hdr->n_targets);
    size_t saf_rows = 0;
    size_t matched_rows = 0;
    size_t tx_len_mismatch_rows = 0;

    log_msg(args, "[2/5] Reading SAF: " + args.saf_path);
    load_saf(args, hdr, windows, saf_rows, matched_rows, tx_len_mismatch_rows);
    if (matched_rows != saf_rows || tx_len_mismatch_rows > 0) {
        std::string msg = "  saf_rows=" + std::to_string(saf_rows) +
                          " matched_rows=" + std::to_string(matched_rows);
        if (tx_len_mismatch_rows > 0) {
            msg += " tx_len_mismatch_rows=" + std::to_string(tx_len_mismatch_rows);
        }
        log_msg(args, msg);
    }

    if (tx_len_mismatch_rows > 0) {
        log_msg(args, "  [warn] BAM header target_len and SAF tx_len mismatch detected."
                      " Coordinate compatibility may be wrong for these references.");
    }

    std::vector<CountRow> counts(hdr->n_targets);
    Stats stats;
    stats.total_tx_len_mismatch = tx_len_mismatch_rows;

    log_msg(args, "[3/5] Streaming BAM and counting only QC-passed reads with 5' end inside valid window");
    if (args.layout == "se") {
        count_se(fp, hdr, args, windows, counts, stats);
    } else {
        count_pe(fp, hdr, args, windows, counts, stats);
    }

    log_msg(args, "[4/5] Writing output to stdout");
    write_output_stdout(hdr, windows, counts);

    log_msg(args, "[5/5] Done");
    log_msg(args, "  total_records=" + std::to_string(stats.total_records));
    log_msg(args, "  total_primary=" + std::to_string(stats.total_primary));
    if (args.layout == "pe") {
        log_msg(args, "  total_paired_records=" + std::to_string(stats.total_layout_kept));
        log_msg(args, "  total_forward_records_used_for_pe_logic=" + std::to_string(stats.total_forward_kept));
    }
    log_msg(args, "  total_in_tx_bounds=" + std::to_string(stats.total_in_tx_bounds));
    log_msg(args, "  total_out_of_tx_bounds=" + std::to_string(stats.total_out_of_tx_bounds));
    log_msg(args, "  total_qc_length=" + std::to_string(stats.total_qc_length));
    log_msg(args, "  total_counted_valid_5p=" + std::to_string(stats.total_valid_5p));
    if (stats.total_out_of_tx_bounds > 0) {
        log_msg(args, "  [warn] Some alignments extend beyond SAF tx_len. This can indicate incompatible coordinates or tx_len metadata.");
    }

    bam_hdr_destroy(hdr);
    sam_close(fp);
    return 0;
}

