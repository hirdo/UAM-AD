"""
Per-scenario evaluation for RCAEval OnlineBoutique RE3-OB dataset.

RE3-OB uses code-level fault injection with 5 fault types (f1–f5).
Runs UAM-AD on each fault-type test file and reports F1/P/R per scenario.

Run twice to compare baseline vs trace-enhanced:

  # Baseline: log + metric only
  python codes/common/eval_per_scenario_rcaeval_re3_ob.py \\
      --data data/rcaeval_re3_ob --dataset rcaeval_re3_ob --data_type fuse \\
      --open_trace False --batch_size 128 --window_size 30 \\
      --epoches 5 5 --patience 3 \\
      --result_dir data/rcaeval_re3_ob/result_phase3_fuse_baseline

  # Trace: log + metric + trace (GAT)
  python codes/common/eval_per_scenario_rcaeval_re3_ob.py \\
      --data data/rcaeval_re3_ob --dataset rcaeval_re3_ob --data_type fuse \\
      --open_trace True --batch_size 128 --window_size 30 \\
      --epoches 5 5 --patience 3 \\
      --result_dir data/rcaeval_re3_ob/result_phase3_fuse_trace

Seeds are separate run.py processes (--run_start inclusive, --run_end exclusive) and seeds already on disk are
skipped, so --run_start 0 --run_end 1 now and --run_start 1 --run_end 5 later give the same 5 seeds. Defaults follow
frozen_config.json (norm_sum); add --enable_rca True for UAM-AD. Results: mean±std over seeds, not the best seed.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

import eval_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

FAULT_TYPES = ["f1", "f2", "f3", "f4", "f5"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_scenario_files(data_dir: str):
    """Return sorted list of (fault_type, test_pkl_path) for all fault types."""
    entries = []
    for fault in FAULT_TYPES:
        path = os.path.join(data_dir, f"test_{fault}.pkl")
        if os.path.exists(path):
            entries.append((fault, path))
        else:
            logging.warning(f"Missing test file: {path}")
    return entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Per-scenario evaluation for RCAEval OnlineBoutique RE3-OB dataset"
    )
    # ── data / model args (passed through to run.py) ──
    p.add_argument("--data",           required=True)
    p.add_argument("--dataset",        default="rcaeval_re3_ob")
    p.add_argument("--data_type",      default="fuse", choices=["fuse", "kpi", "log"])
    p.add_argument("--open_trace",     default="False")
    p.add_argument("--num_services",   default=11,  type=int)
    p.add_argument("--trace_c",        default=6,   type=int)
    p.add_argument("--epoches",        default=[5, 5], nargs="+", type=int)
    p.add_argument("--batch_size",     default=128, type=int)
    p.add_argument("--patience",       default=3,   type=int)
    p.add_argument("--window_size",    default=30,  type=int,
                   help="Sliding window size in timesteps (1 timestep = 1 second).")
    p.add_argument("--val_percentile", default=95,  type=float,
                   help="Percentile of normal training losses as anomaly threshold.")
    p.add_argument("--alpha",          default=0.5, type=float)
    p.add_argument("--open_gan_sep",   default="True")
    p.add_argument("--run_start",      default=0,   type=int)
    p.add_argument("--run_end",        default=1,   type=int)
    p.add_argument("--result_dir",     default=None,
                   help="Base result dir. Each fault type gets its own subdir. "
                        "Defaults to {data}/result_phase3_{data_type}_"
                        "{'trace' if open_trace else 'baseline'}")
    p.add_argument("--gate_lambda",    default=0.01, type=float,
                   help="L1 regularizer on residual-gated trace gate g (auto-applied when open_trace=True).")
    p.add_argument("--score_rule",     default="norm_sum", choices=["raw_sum", "norm_sum"],
                   help="Frozen config: norm_sum (run.py's own default is raw_sum).")
    p.add_argument("--enable_rca",     default="False",
                   help="True for UAM-AD (open_trace True): also write the service ranking used for HR@k.")
    p.add_argument("--rca_top_k", default=5, type=int,
                   help="Services kept per sample in rca_results.json; 5 is needed for HR@5.")
    p.add_argument("--run_py",         default=None,
                   help="Path to run.py (auto-detected if not set)")
    args = p.parse_args()

    # Convert data path to absolute so run.py subprocess (cwd=codes/) resolves it
    args.data = os.path.abspath(args.data)
    if args.result_dir:
        args.result_dir = os.path.abspath(args.result_dir)

    # Auto-detect run.py
    if args.run_py:
        run_py = args.run_py
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_py = os.path.normpath(os.path.join(this_dir, "..", "run.py"))
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py not found at {run_py}. Use --run_py.")

    open_trace_str = str(args.open_trace).lower()
    suffix = "trace" if open_trace_str in ("true", "1", "yes") else "baseline"
    result_base = args.result_dir or os.path.join(
        args.data, f"result_phase3_{args.data_type}_{suffix}"
    )

    scenario_files = _find_scenario_files(args.data)
    if not scenario_files:
        logging.error("No scenario test files found. Run preprocess_rcaeval_re3_ob.py first.")
        sys.exit(1)

    logging.info(f"Found {len(scenario_files)} scenarios.")
    logging.info(f"Results will be saved under: {result_base}/\n")

    elapsed, per_scenario = {}, {}
    total_start = time.perf_counter()

    for fault, test_pkl in scenario_files:
        sc_result_dir = os.path.join(result_base, fault)
        os.makedirs(sc_result_dir, exist_ok=True)

        logging.info(f"{'='*60}")
        logging.info(f"Scenario: {fault}")
        logging.info(f"  test_pkl : {test_pkl}")
        logging.info(f"  result   : {sc_result_dir}")

        # One run.py process per seed: seeds already on disk are skipped, so seeds can be added in later invocations
        # (e.g. --run_start 0 --run_end 1 now, --run_start 1 --run_end 5 later).
        elapsed[fault] = 0.0
        for seed in range(args.run_start, args.run_end):
            if seed in eval_seeds.read_seeds(sc_result_dir):
                logging.info(f"  seed {seed}: already done, skipped")
                continue
            cmd = [
                sys.executable, run_py,
                "--data",           args.data,
                "--dataset",        args.dataset,
                "--data_type",      args.data_type,
                "--open_trace",     args.open_trace,
                "--num_services",   str(args.num_services),
                "--trace_c",        str(args.trace_c),
                "--epoches",        *[str(e) for e in args.epoches],
                "--batch_size",     str(args.batch_size),
                "--patience",       str(args.patience),
                "--window_size",    str(args.window_size),
                "--val_percentile", str(args.val_percentile),
                "--alpha",          str(args.alpha),
                "--open_gan_sep",   args.open_gan_sep,
                "--run_start",      str(seed),
                "--run_end",        str(seed + 1),
                "--test_pkl",       test_pkl,
                "--result_dir",     sc_result_dir,
                "--gate_lambda",    str(args.gate_lambda),
                "--score_rule",     args.score_rule,
                "--enable_rca",     args.enable_rca,
                "--rca_top_k",         str(args.rca_top_k),
            ]
            sc_start = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=os.path.dirname(run_py))
                if proc.returncode != 0:
                    logging.warning(f"  seed {seed}: run.py exited with code {proc.returncode}")
            except Exception as e:
                logging.error(f"  seed {seed}: failed to run: {e}")
            elapsed[fault] += time.perf_counter() - sc_start
        per_scenario[fault] = eval_seeds.read_seeds(sc_result_dir)

    per_scenario = {k: v for k, v in per_scenario.items() if v}
    if not per_scenario:
        logging.error("No results collected.")
        return

    eval_seeds.print_table(
        f"{args.dataset} per-scenario results ({args.data_type}, open_trace={args.open_trace}, {args.score_rule}); "
        f"mean±std over seeds", per_scenario, elapsed)
    logging.info(f"Total wall time of this invocation: {(time.perf_counter() - total_start) / 60:.1f} min")

    summary = {
        "config": {
            "data_type":      args.data_type,
            "open_trace":     args.open_trace,
            "score_rule":     args.score_rule,
            "enable_rca":     args.enable_rca,
            "val_percentile": args.val_percentile,
            "window_size":    args.window_size,
            "epoches":        args.epoches,
        },
        "per_scenario_per_seed": {fault: {str(s): m for s, m in by_seed.items()} for fault, by_seed in per_scenario.items()},
        "elapsed_sec_this_invocation": elapsed,
    }
    summary_path = os.path.join(result_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
