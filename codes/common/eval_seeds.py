"""
Shared by the eval_per_scenario_* scripts: per-seed results and resumable seed runs.

run.py names each result folder by a hash of the parameters (which include the run index) and writes info_score.txt and
params.json into it, so the seed of a folder is "run_times" in its params.json. Reading results per seed (instead of
taking the best F1 over all folders) lets seeds be run in separate invocations: run seed 0 now, seeds 1-4 later.
"""
import glob
import json
import os
import re

import numpy as np

SHOWN = ["f1", "pc", "rc", "auroc", "auprc"]       # oracle_f1 of run.py is point-adjusted, so it is not shown


def read_seeds(result_dir):
    """{seed: {metric: value}} from every <hash>/info_score.txt under result_dir."""
    out = {}
    for f in glob.glob(os.path.join(result_dir, "*", "info_score.txt")):
        d = os.path.dirname(f)
        try:
            seed = json.load(open(os.path.join(d, "params.json")))["run_times"]
            line = next(l for l in open(f) if l.startswith("* Test --"))
        except (OSError, KeyError, ValueError, StopIteration):
            continue
        out[seed] = {k: float(v) for k, v in re.findall(r"(\w+):([-\d.eE]+|nan)", line[len("* Test --"):])}
    return out


def mean_std(values):
    return f"{np.mean(values):.4f}±{np.std(values):.4f}" if len(values) else "-"


def print_table(title, per_scenario, elapsed):
    """per_scenario: {scenario: {seed: metrics}}. Prints mean±std over seeds per scenario, then the mean over scenarios
    of each seed (seeds present in every scenario) with its mean±std over seeds."""
    print(f"\n{'=' * 100}\n  {title}\n{'=' * 100}")
    print(f"  {'scenario':<22} {'n':>2}  " + "  ".join(f"{m:>13}" for m in SHOWN) + f"  {'time(s)':>8}")
    for sc, by_seed in per_scenario.items():
        cells = [mean_std([r[m] for r in by_seed.values() if m in r]) for m in SHOWN]
        print(f"  {sc:<22} {len(by_seed):>2}  " + "  ".join(f"{c:>13}" for c in cells) + f"  {elapsed.get(sc, 0):>8.0f}")
    common = set.intersection(*[set(v) for v in per_scenario.values()]) if per_scenario else set()
    if common:
        cells = [mean_std([np.mean([per_scenario[sc][s][m] for sc in per_scenario]) for s in sorted(common)])
                 for m in SHOWN]
        print(f"  {'mean over scenarios':<22} {len(common):>2}  " + "  ".join(f"{c:>13}" for c in cells)
              + f"  {sum(elapsed.values()):>8.0f}")
    print("=" * 100 + "\n")
