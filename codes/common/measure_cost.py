"""
Phase 3, Table E (cost): parameters, CPU inference time and peak RAM of UAM-AD, UAC-AD, ART, AnoFusion, Eadro, OrEdge on one test file.

One fresh process per (model, dataset). CPU only (CUDA hidden), the torch default number of threads, the same batch size (128) for every model, the whole test file.
The data are loaded and the network built BEFORE the clock starts; RSS (Windows working set, read through ctypes) is taken at that moment (base), then sampled every 5 ms during the inference pass.
  unit      = what the model scores: one sample (1 s of telemetry) for ART, AnoFusion, Eadro, OrEdge; one 30-sample window for UAM-AD / UAC-AD. Units differ, so the
              comparable figure is the inference time of the whole test file (test_samples is stored for ms per 1000 samples).
  delta MB  = peak RSS during inference - RSS before it (what the inference itself adds); the absolute peak is reported too.
  params    = parameters of the network used at inference (UAM-AD/UAC-AD: the generator only, no GAN discriminator; OrEdge: its whole module, whose forward always
              runs the reconstruction branch; AnoFusion: the network, not the mutual-information matrices, which are data).
UAM-AD and UAC-AD load their stored item-8 checkpoint (seed 0) through run.py --pre_model, with BaseModel.evaluate replaced by the timed forward loop.
ART, AnoFusion, Eadro and OrEdge stored no checkpoint in item 9, so their networks are built with the runner's own code and left at their initial weights:
inference time and memory do not depend on the weight values (they depend on the shapes), but this is NOT a trained model's output.

Usage:
  python measure_cost.py --model art --dataset rcaeval_re2_ob --test test_cpu          (one worker)
  python measure_cost.py --all --datasets rcaeval_re2_ob:test_cpu [...]                (orchestrator: one process per model)
Results: <out>/<dataset>/<model>.json
"""
import argparse
import gc
import json
import os
import runpy
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(HERE)
DATA = os.path.join(os.path.dirname(CODES), "data")
COMP = os.path.join(CODES, "competitors")
OUT = r"D:\ClaudeWork\phase3\cost"
ITEM8 = r"D:\ClaudeWork\phase3\item8"
MODELS = ["uacad", "uamad", "anofusion", "art", "eadro", "oredge"]
BS = 128


def rss_bytes():
    """Working set of this process (Windows psapi through ctypes, so no extra package is needed)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize


class Peak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.max, self.halt = 0, threading.Event()

    def run(self):
        while not self.halt.is_set():
            self.max = max(self.max, rss_bytes())
            time.sleep(0.005)


def timed(fn):
    gc.collect()
    base = rss_bytes()
    pk = Peak()
    pk.max = base
    pk.start()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    pk.halt.set()
    pk.join()
    mb = 2 ** 20
    return dict(infer_s=dt, rss_base_mb=base / mb, rss_peak_mb=pk.max / mb, delta_mb=(pk.max - base) / mb)


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


def finish(a, fn, n_units, params, extra=None):
    import torch
    r = dict(model=a.model, dataset=a.dataset, test=a.test, device="cpu", torch_threads=torch.get_num_threads(), batch_size=BS, n_units=int(n_units),
             params=params, trained_weights=a.model in ("uamad", "uacad"), repeat=a.repeat) | timed(lambda: [fn() for _ in range(a.repeat)]) | (extra or {})
    r["ms_per_unit"] = 1000 * r["infer_s"] / max(n_units * a.repeat, 1)
    r["ms_per_1000_samples"] = 1e6 * r["infer_s"] / max(r["samples_done"] * a.repeat, 1)
    r["limit"] = a.limit
    os.makedirs(os.path.join(a.out, a.dataset, a.test), exist_ok=True)
    json.dump(r, open(os.path.join(a.out, a.dataset, a.test, f"{a.model}.json"), "w"), indent=1)
    print(json.dumps(r), flush=True)


def common(a):
    sys.path[:0] = [os.path.join(COMP, d) for d in ("art", "eadro", "oredge", "anofusion")] + [COMP, os.path.join(CODES, "models")]
    import run_art as R
    d = os.path.join(DATA, a.dataset)
    meta = R.load(os.path.join(d, "meta.pkl"))
    kpi_idx = R.metric_index(meta)
    return R, d, meta, kpi_idx, max(len(i) for i in kpi_idx), list(meta["services"])


def train_scale(R, d, kpi_idx, c_metric, services):
    train = R.load(os.path.join(d, "train.pkl"))
    f, _, _, _, _, k = R.build(train, kpi_idx, c_metric)
    return train, f, k


def sn_test_stem(sc):
    import glob
    return os.path.basename(glob.glob(os.path.join(DATA, "sn", "scenarios", f"test_{sc}_*.pkl"))[0])[:-4]


def worker_art_sn(a):
    """ART on SN: the data, the scale and the scored windows of run_art_sn.py (a.test = test_<scenario>)."""
    import numpy as np, torch
    sys.path[:0] = [os.path.join(COMP, "art")]
    import run_art as R
    from art_model import AutoRegressor
    from run_art_sn import world
    from sn_data import SNData
    d = os.path.join(DATA, "sn")
    sn = SNData(d, pad=0)
    feats, adj, c_metric = world(sn, no_logs=False)
    S = len(sn.services)
    m = np.abs(feats[sn.positions("train")]).max(0, keepdims=True)
    dev = torch.device("cpu")
    D = R.Data(np.clip(feats / np.where(m > 0, m, 1.0), -100.0, 100.0), adj, dev, S % 2 == 1)
    pos = np.intersect1d(R.valid_positions(sn.run, sn.sec), sn.positions(sn_test_stem(a.test[len("test_"):])))
    torch.manual_seed(0)
    net = AutoRegressor(S + int(S % 2 == 1), feats.shape[2]).to(dev)
    finish(a, lambda: R.score(net, D, pos, BS, S), len(pos), n_params(net), dict(test_samples=int(len(pos)), samples_done=int(len(pos))))


def worker_eadro_sn(a):
    """Eadro on SN: the arrays and the held-out windows of run_eadro_sn.py (history off)."""
    import numpy as np, torch
    sys.path[:0] = [os.path.join(COMP, "eadro")]
    import run_eadro as E
    from eadro_model import Eadro
    from sn_labeled import SNLabeled
    L = SNLabeled(os.path.join(DATA, "sn"), E.T)
    D = L.arrays(torch.device("cpu"), False)
    held = L.fold(sn_test_stem(a.test[len("test_"):]))[1]
    torch.manual_seed(0)
    net = Eadro(len(L.sn.services), L.c_metric, E.N_TRACE, E.T, D["X"].shape[-1] - L.c_metric - E.N_TRACE)
    finish(a, lambda: E.predict(net, D, held, L.c_metric, BS), len(held), n_params(net), dict(test_samples=int(len(held)), samples_done=int(len(held))))


def worker_oredge_sn(a):
    """OrEdge on SN: the arrays, log counts and held-out windows of run_oredge_sn.py (history off)."""
    import collections
    import numpy as np, torch
    sys.path[:0] = [os.path.join(COMP, "oredge"), os.path.join(COMP, "eadro")]
    import run_oredge as O
    from oredge_model import OrEdge
    from run_eadro import T
    from sn_labeled import SNLabeled
    L = SNLabeled(os.path.join(DATA, "sn"), T)
    sn = L.sn
    dev = torch.device("cpu")
    D = L.arrays(dev, False)
    ptr = sn.positions("train")
    vocab = {t: i for i, t in enumerate(sorted(collections.Counter(t for p in ptr for t in sn.samples[p]["logs"])))}
    counts = np.zeros((len(sn.samples), len(vocab)), dtype=np.float32)
    for p, w in enumerate(sn.samples):
        for t in w["logs"]:
            j = vocab.get(t)
            if j is not None:
                counts[p, j] += 1
    counts = counts / np.maximum(counts[ptr].max(0), 1.0)
    D["L"] = torch.tensor(L.tile(counts), device=dev).clamp_(-O.CLIP, O.CLIP)
    D["M"] = torch.tensor(O.owner_mask(vocab, sn.services), device=dev)
    held = L.fold(sn_test_stem(a.test[len("test_"):]))[1]
    torch.manual_seed(0)
    net = OrEdge(L.c_metric, D["L"].shape[1], O.N_TRACE, O.T).to(dev)
    finish(a, lambda: O.predict(net, D, held, L.c_metric, BS), len(held), n_params(net), dict(test_samples=int(len(held)), samples_done=int(len(held))))


def worker_art(a):
    import numpy as np, torch
    if a.dataset == "sn":
        return worker_art_sn(a)
    from art_model import AutoRegressor
    R, d, meta, kpi_idx, c_metric, services = common(a)
    S = len(kpi_idx)
    pad = S % 2 == 1
    train, f, k = train_scale(R, d, kpi_idx, c_metric, services)
    scale = np.maximum(np.abs(R.add_logs(f, k, train, services)).max(0, keepdims=True), 1e-8)
    del train, f, k
    split = R.load(os.path.join(d, a.test + ".pkl"))
    f, ad, r, sec, y, keys = R.build(split, kpi_idx, c_metric)
    f = R.add_logs(f, keys, split, services)
    del split
    dev = torch.device("cpu")
    D = R.Data(f / scale, ad, dev, pad)
    ite = R.valid_positions(r, sec)
    ite = ite[:a.limit] if a.limit else ite
    torch.manual_seed(0)
    net = AutoRegressor(S + int(pad), f.shape[2]).to(dev)
    del f, ad
    finish(a, lambda: R.score(net, D, ite, BS, S), len(ite), n_params(net), dict(test_samples=int(len(r)), samples_done=int(len(ite))))


def worker_eadro(a):
    import numpy as np, torch
    if a.dataset == "sn":
        return worker_eadro_sn(a)
    import run_eadro as E
    from eadro_model import Eadro
    R, d, meta, kpi_idx, c_metric, services = common(a)
    train, f, k = train_scale(R, d, kpi_idx, c_metric, services)
    scale = np.maximum(np.abs(R.add_logs(f, k, train, services)).max(0, keepdims=True), 1e-8)
    del train, f, k
    dev = torch.device("cpu")
    D = E.make(os.path.join(d, a.test + ".pkl"), kpi_idx, c_metric, scale, services, a.dataset, dev)
    torch.manual_seed(0)
    net = Eadro(len(services), c_metric, E.N_TRACE, E.T, D["X"].shape[-1] - c_metric - E.N_TRACE).to(dev)
    idx = np.arange(len(D["pos"]))
    idx = idx[:a.limit] if a.limit else idx
    finish(a, lambda: E.predict(net, D, idx, c_metric, BS), len(idx), n_params(net), dict(test_samples=int(D["n_total"]), samples_done=int(len(idx))))


def worker_oredge(a):
    import collections
    import numpy as np, torch
    if a.dataset == "sn":
        return worker_oredge_sn(a)
    import run_oredge as O
    from oredge_model import OrEdge
    R, d, meta, kpi_idx, c_metric, services = common(a)
    train, f, k = train_scale(R, d, kpi_idx, c_metric, services)
    scale = np.maximum(np.abs(R.add_logs(f, k, train, services)).max(0, keepdims=True), 1e-8)
    vocab = {t: i for i, t in enumerate(sorted(collections.Counter(t for s in train.values() for t in s["logs"])))}
    lscale = np.maximum(O.log_counts(train, vocab).max(0), 1.0)
    mask = O.owner_mask(vocab, services)
    del train, f, k
    dev = torch.device("cpu")
    D = O.make_all(os.path.join(d, a.test + ".pkl"), kpi_idx, c_metric, scale, lscale, vocab, mask, services, a.dataset, dev)
    torch.manual_seed(0)
    net = OrEdge(c_metric, D["L"].shape[1], O.N_TRACE, O.T).to(dev)
    idx = np.arange(len(D["pos"]))
    idx = idx[:a.limit] if a.limit else idx
    finish(a, lambda: O.predict(net, D, idx, c_metric, BS), len(idx), n_params(net), dict(test_samples=int(D["n_total"]), samples_done=int(len(idx))))


def worker_anofusion_sn(a):
    """AnoFusion on SN: the channels, the scale and the scored windows of run_anofusion_sn.py (history off)."""
    import collections
    import numpy as np, torch
    sys.path[:0] = [os.path.join(COMP, "anofusion"), COMP]
    import run_anofusion as F
    import run_anofusion_sn as FS
    from anofusion_model import AnoFusionNet
    from sn_data import SNData
    sn = SNData(os.path.join(DATA, "sn"), pad=0)
    ptr = sn.positions("train")
    vocab = {t: i for i, t in enumerate(sorted(collections.Counter(t for p in ptr for t in sn.samples[p]["logs"])))}
    X = FS.channels(sn, vocab)
    m = np.abs(X[ptr]).max(0)
    Xt = torch.tensor(np.clip(X / np.where(m > 0, m, 1.0), -100.0, 100.0))
    pos = np.intersect1d(F.valid_positions(sn.run), sn.positions(sn_test_stem(a.test[len("test_"):])))
    N = X.shape[1]
    torch.manual_seed(0)
    A = torch.rand(N, N, 6)                  # the mutual-information matrices of the authors: same shape, values irrelevant for cost
    net = AnoFusionNet(N, F.WINDOW)
    finish(a, lambda: F.predict_err(net, A, Xt, pos, BS), len(pos), n_params(net),
           dict(channels=int(N), metric_channels=int(len(sn.meta["metric_names"])), test_samples=int(len(pos)), samples_done=int(len(pos))))


def worker_anofusion(a):
    import numpy as np, torch
    if a.dataset == "sn":
        return worker_anofusion_sn(a)
    sys.path[:0] = [os.path.join(COMP, "anofusion"), COMP]
    import run_anofusion as F
    from anofusion_model import AnoFusionNet
    d = os.path.join(DATA, a.dataset)
    train = F.load(os.path.join(d, "train.pkl"))
    vocab = F.build_vocab(train)
    Xtr, _, _, _ = F.build_channels(train, vocab)
    n_metric = len(next(iter(train.values()))["kpis"])
    del train
    m = np.abs(Xtr).max(0)
    scale = np.maximum(m, 1e-8)
    N = Xtr.shape[1]
    del Xtr
    te = F.load(os.path.join(d, a.test + ".pkl"))
    Xte, rte, _, _ = F.build_channels(te, vocab)
    del te
    dev = torch.device("cpu")
    ite = F.valid_positions(rte)
    ite = ite[:a.limit] if a.limit else ite
    Xt = torch.tensor(Xte / scale, device=dev)
    del Xte
    torch.manual_seed(0)
    A = torch.rand(N, N, 6, device=dev)       # the mutual-information matrices of the authors: same shape, values irrelevant for cost
    net = AnoFusionNet(N, F.WINDOW).to(dev)
    finish(a, lambda: F.predict_err(net, A, Xt, ite, BS), len(ite), n_params(net), dict(channels=int(N), metric_channels=int(n_metric), test_samples=int(len(rte)), samples_done=int(len(ite))))


def worker_ours(a):
    """UAM-AD / UAC-AD: run.py --pre_model with the stored seed-0 checkpoint of the scenario; BaseModel.evaluate is replaced by a timed forward loop."""
    sys.path[:0] = [CODES, HERE]
    os.chdir(CODES)
    import numpy as np, torch
    import models.basev3 as b
    import rescore_item10 as RS
    ns, _, _ = RS.DATASETS[a.dataset]
    scen = a.test[len("test_"):]
    ck = RS.checkpoint_dir(ITEM8, a.dataset, a.model, scen, 0)
    test_pkl = os.path.join(DATA, "sn", "scenarios", sn_test_stem(scen) + ".pkl") if a.dataset == "sn" else os.path.join(DATA, a.dataset, a.test + ".pkl")
    tmp = os.path.join(a.out, "_tmp", a.dataset, a.model)
    args = RS.run_py_args(a.dataset, ns, test_pkl, tmp, os.path.join(ck, "model.ckpt")) + ["--gpu", "false", "--gpu_device", "-1", "--batch_size", str(BS)]   # run.py sets CUDA_VISIBLE_DEVICES from --gpu_device itself
    if a.model == "uacad":
        args += ["--open_trace", "False"]

    def timed_forward(self, test_loader, datatype="Test", threshold=None):
        self.model.eval()
        self.loss_fusion.eval()
        n = len(test_loader.dataset)

        def run():
            dist, emb = [], []
            with torch.no_grad():
                for bi in test_loader:
                    res = self.model.forward(self._BaseModel__input2device(bi))
                    dist.append(res["fusion_loss"].cpu().numpy())
                    emb.append(res["features"][2].cpu().numpy())
        finish(a, run, n, n_params(self.model), dict(checkpoint=ck, windows=int(n), test_samples=int(n * test_loader.dataset.window_size), samples_done=int(n * test_loader.dataset.window_size)))
        raise SystemExit(0)

    # models/utils.py:170 builds `nn.LayerNorm(d).cuda()` on every attention call (a fresh, default-initialised LayerNorm); on a CPU run that call is made a no-op in this
    # process only, which computes exactly the same thing on the CPU. The repository code is not changed.
    torch.nn.Module.cuda = lambda self, device=None: self
    b.BaseModel.evaluate = timed_forward
    sys.argv = ["run.py"] + args
    runpy.run_path(os.path.join(CODES, "run.py"), run_name="__main__")


def orchestrate(a):
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "status.txt"), "a", buffering=1)
    for spec in a.datasets:
        ds, test = spec.split(":")
        for m in a.models:
            if os.path.isfile(os.path.join(a.out, ds, test, f"{m}.json")) and not a.redo:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
            lim = dict(x.split("=") for x in a.limits).get(m, "0")
            cmd = [sys.executable, os.path.abspath(__file__), "--model", m, "--dataset", ds, "--test", test, "--out", a.out, "--limit", lim, "--repeat", str(a.repeat)]
            t0 = time.time()
            log.write(f"{time.strftime('%m-%d %H:%M:%S')} START {ds} {test} {m}\n")
            with open(os.path.join(a.out, f"{ds}_{test}_{m}.log"), "w", encoding="utf-8") as lg:
                rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=CODES).returncode
            log.write(f"{time.strftime('%m-%d %H:%M:%S')} {'OK' if rc == 0 else 'FAIL rc=%s' % rc} {ds} {test} {m} [{time.time() - t0:.0f}s]\n")
    log.write("ALL DONE\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--dataset")
    ap.add_argument("--test")
    ap.add_argument("--datasets", nargs="+", default=["rcaeval_re2_ob:test_cpu"], help="dataset:test_file_stem")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="scoring passes inside the clock (SN test files hold 32-263 windows, too short for one pass)")
    ap.add_argument("--limit", type=int, default=0, help="worker: score only the first N units (0 = the whole test file)")
    ap.add_argument("--limits", nargs="*", default=[], help="orchestrator: model=N, e.g. anofusion=2560")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.all:
        orchestrate(a)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path[:0] = [os.path.join(COMP, d) for d in ("art", "eadro", "oredge", "anofusion")] + [COMP, os.path.join(CODES, "models")]
        {"art": worker_art, "eadro": worker_eadro, "oredge": worker_oredge, "anofusion": worker_anofusion, "uamad": worker_ours, "uacad": worker_ours}[a.model](a)
