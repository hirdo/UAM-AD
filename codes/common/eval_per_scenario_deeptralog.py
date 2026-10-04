"""
Per-F-case evaluation for the DeepTraLog dataset (sync trace branch, Step 3
of the async-trace plan). Mirrors eval_per_scenario_sn.py exactly; see that
file for the general pattern this follows.

For each F-case test file under {data}/scenarios/, runs the model
(unmodified architecture — reused as-is per the plan's Câu 2 decision) on:
    train.pkl  = a sample of normal traces
    test_pkl   = scenarios/test_{fcase}.pkl
Aggregates F1 / Precision / Recall / AUROC / AUPRC across F-cases.

Usage:
    python codes/common/eval_per_scenario_deeptralog.py \\
        --data data/deeptralog \\
        --open_trace True \\
        --epoches 10 10 --batch_size 128 --patience 5 \\
        --window_size 5 --val_percentile 95 \\
        --run_start 0 --run_end 1
"""

import argparse
import json
import logging
import os
import re as _re
import subprocess
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")


def _find_scenario_files(data_dir: str):
    scenarios_dir = os.path.join(data_dir, "scenarios")
    if not os.path.isdir(scenarios_dir):
        raise FileNotFoundError(f"scenarios/ not found under {data_dir}. Re-run preprocess_deeptralog.py first.")
    entries = []
    for fname in sorted(os.listdir(scenarios_dir)):
        if fname.startswith("test_") and fname.endswith(".pkl"):
            entries.append((fname[len("test_"):-len(".pkl")], os.path.join(scenarios_dir, fname)))
    return entries


def _scan_latest_results(result_dir: str):
    if not os.path.isdir(result_dir):
        return None
    best = None
    for root, dirs, files in os.walk(result_dir):
        for fname in files:
            if fname != "info_score.txt":
                continue
            try:
                with open(os.path.join(root, fname)) as f:
                    for line in f:
                        if not line.startswith("* Test --"):
                            continue
                        vals = {}
                        for kv in line[len("* Test --"):].split():
                            k, _, v = kv.partition(":")
                            try:
                                vals[k] = float(v)
                            except ValueError:
                                pass
                        if "f1" in vals and (best is None or vals["f1"] > best["f1"]):
                            best = vals
            except Exception:
                continue
    return best


def main():
    p = argparse.ArgumentParser(description="Per-F-case evaluation for DeepTraLog dataset")
    p.add_argument("--data", required=True)
    p.add_argument("--dataset", default="deeptralog")
    p.add_argument("--data_type", default="fuse", choices=["fuse", "kpi", "log"])
    p.add_argument("--open_trace", default="True")
    p.add_argument("--open_async_trace", default="False",
                   help="Step 4: enable the additive async trace branch (async_trace_model_v3.py).")
    p.add_argument("--score_rule", default="norm_sum", choices=["raw_sum", "norm_sum"],
                   help="Score rule (see run.py). norm_sum for BOTH the sync-only and the "
                        "sync+async arm so the comparison is like-for-like.")
    p.add_argument("--epoches", default=[10, 10], nargs="+", type=int)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--patience", default=5, type=int)
    p.add_argument("--window_size", default=5, type=int)
    p.add_argument("--val_percentile", default=95, type=float)
    p.add_argument("--alpha", default=0.16, type=float)
    p.add_argument("--open_gan_sep", default="True")
    p.add_argument("--open_unmatch_zoomout", default="False",
                   help="No metric in this dataset (kpi_c=1, all zero) -> the unmatched-KPI "
                        "contrastive hinge is degenerate (see plan Câu 2 'Chạy log+trace khi "
                        "không có metric'); off by default.")
    p.add_argument("--run_start", default=0, type=int)
    p.add_argument("--run_end", default=1, type=int)
    p.add_argument("--gate_lambda", default=0.01, type=float)
    p.add_argument("--fcases", nargs="*", default=None, help="Subset of F-cases to evaluate (default: all found).")
    p.add_argument("--result_dir", default=None)
    p.add_argument("--run_py", default=None)
    args = p.parse_args()

    run_py = args.run_py or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run.py"))
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py not found at {run_py}")

    args.data = os.path.abspath(args.data)
    if args.result_dir:
        args.result_dir = os.path.abspath(args.result_dir)

    has_trace = str(args.open_trace).lower() in ("true", "1", "yes")
    has_async = str(args.open_async_trace).lower() in ("true", "1", "yes")
    if has_trace:
        suffix = "trace_both_sync_and_async" if has_async else "trace_only_sync"
    else:
        suffix = "baseline_async" if has_async else "baseline"
    result_base = args.result_dir or os.path.join(args.data, f"result_per_scenario_{args.data_type}_{suffix}")

    scenario_files = _find_scenario_files(args.data)
    if args.fcases:
        wanted = set(args.fcases)
        scenario_files = [(n, p) for n, p in scenario_files if n in wanted]
    if not scenario_files:
        logging.error("No matching F-case test files found.")
        sys.exit(1)

    logging.info(f"Found {len(scenario_files)} F-case(s) to evaluate.")
    logging.info(f"Results -> {result_base}/\n")

    results = []
    total_start = time.perf_counter()
    for fcode, test_pkl in scenario_files:
        fc_result_dir = os.path.join(result_base, fcode)
        os.makedirs(fc_result_dir, exist_ok=True)
        logging.info(f"{'='*60}\nF-case: {fcode}\n  test_pkl: {test_pkl}\n  result  : {fc_result_dir}")

        cmd = [
            sys.executable, run_py,
            "--data", args.data, "--dataset", args.dataset, "--data_type", args.data_type,
            "--open_trace", args.open_trace,
            "--open_async_trace", args.open_async_trace,
            "--score_rule", args.score_rule,
            "--epoches", *[str(e) for e in args.epoches],
            "--batch_size", str(args.batch_size), "--patience", str(args.patience),
            "--window_size", str(args.window_size), "--val_percentile", str(args.val_percentile),
            "--alpha", str(args.alpha), "--open_gan_sep", args.open_gan_sep,
            "--open_unmatch_zoomout", args.open_unmatch_zoomout,
            "--run_start", str(args.run_start), "--run_end", str(args.run_end),
            "--test_pkl", test_pkl, "--result_dir", fc_result_dir,
            "--gate_lambda", str(args.gate_lambda),
        ]
        sc_start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=False, text=True, cwd=os.path.dirname(run_py))
            if proc.returncode != 0:
                logging.warning(f"  run.py exited with code {proc.returncode}")
        except Exception as e:
            logging.error(f"  Failed on {fcode}: {e}")
            results.append((fcode, 0.0, 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan")))
            continue
        elapsed = time.perf_counter() - sc_start

        res = _scan_latest_results(fc_result_dir)
        if res is None:
            logging.warning(f"  No result files found in {fc_result_dir}")
            results.append((fcode, 0.0, 0.0, 0.0, elapsed, float("nan"), float("nan"), float("nan")))
        else:
            f1, pc, rc = res["f1"], res.get("pc", 0.0), res.get("rc", 0.0)
            auroc, auprc, oracle_f1 = res.get("auroc", float("nan")), res.get("auprc", float("nan")), res.get("oracle_f1", float("nan"))
            results.append((fcode, f1, pc, rc, elapsed, auroc, auprc, oracle_f1))
            logging.info(f"  -> F1={f1:.4f} P={pc:.4f} R={rc:.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f} oracleF1={oracle_f1:.4f} time={elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    f1s = np.array([r[1] for r in results]); pcs = np.array([r[2] for r in results]); rcs = np.array([r[3] for r in results])
    times = np.array([r[4] for r in results]); aurocs = np.array([r[5] for r in results]); auprcs = np.array([r[6] for r in results]); oracles = np.array([r[7] for r in results])

    col = 12
    sep = "-" * (col + 66)
    print(f"\n{'='*104}\n  Per-F-case Evaluation ({args.data_type}, open_trace={args.open_trace}); F1/P/R at val threshold (p{args.val_percentile:g})\n{'='*104}")
    print(f"  {'F-case':<{col}}  {'F1':>6}  {'Precision':>9}  {'Recall':>6}  {'AUROC':>6}  {'AUPRC':>6}  {'OracleF1':>8}  {'Time(s)':>8}")
    print(f"  {sep}")
    for fcode, f1, pc, rc, t, au, ap, of1 in results:
        print(f"  {fcode:<{col}}  {f1:>6.4f}  {pc:>9.4f}  {rc:>6.4f}  {au:>6.4f}  {ap:>6.4f}  {of1:>8.4f}  {t:>8.1f}")
    print(f"  {sep}")
    print(f"  {'Mean':<{col}}  {f1s.mean():>6.4f}  {pcs.mean():>9.4f}  {rcs.mean():>6.4f}  {np.nanmean(aurocs):>6.4f}  {np.nanmean(auprcs):>6.4f}  {np.nanmean(oracles):>8.4f}  {times.mean():>8.1f}")
    print(f"  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)\n{'='*104}\n")

    summary = {
        "config": {"data_type": args.data_type, "open_trace": args.open_trace, "val_percentile": args.val_percentile, "window_size": args.window_size},
        "per_fcase": [{"fcase": fc, "f1": f1, "precision": pc, "recall": rc, "auroc": au, "auprc": ap, "oracle_f1": of1, "elapsed_sec": t}
                      for fc, f1, pc, rc, t, au, ap, of1 in results],
        "aggregate": {"f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
                      "precision_mean": float(pcs.mean()), "recall_mean": float(rcs.mean()),
                      "auroc_mean": float(np.nanmean(aurocs)), "auprc_mean": float(np.nanmean(auprcs)),
                      "oracle_f1_mean": float(np.nanmean(oracles)), "total_elapsed_sec": float(total_elapsed)},
    }
    with open(os.path.join(result_base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved -> {os.path.join(result_base, 'summary.json')}")


if __name__ == "__main__":
    main()
