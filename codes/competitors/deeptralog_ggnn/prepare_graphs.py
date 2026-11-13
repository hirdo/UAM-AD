"""
Convert DeepTraLog's GraphData (graph_data.z01..z07 + graph_data.zip, FudanSELab/DeepTraLog) into a compact
npz that the GGNN baseline (run_ggnn.py) reads. The zip's central directory is broken, so entries are found by
scanning local headers and inflated as raw deflate (same trick as docs/preprocess_deeptralog_*.md).

The GGNN only uses the node event id (last field of node_info) and edge_index, so only those are kept.
Event embeddings (embedding_id_url+type.csv in the authors' code) are NOT published; they are rebuilt from the
event text (TF-IDF + SVD), see event_embedding().

Usage:
  cat graph_data.z0? graph_data.zip > combined.zip      # in the GraphData folder
  python prepare_graphs.py --zip combined.zip --out_dir <dir>
"""
import argparse
import json
import os
import re
import struct
import zlib

import numpy as np

NAMES = ["id_url+type.csv"] + [f"process{i}.jsons" for i in range(8)]


def find_entries(data):
    found = {}
    for m in re.finditer(b"PK\x03\x04", data):
        i = m.start()
        if i + 30 > len(data):
            continue
        fn = struct.unpack("<H", data[i + 26:i + 28])[0]
        ex = struct.unpack("<H", data[i + 28:i + 30])[0]
        name = data[i + 30:i + 30 + fn].decode("latin1")
        if name in NAMES and name not in found:
            cs = struct.unpack("<I", data[i + 18:i + 22])[0]
            found[name] = (i + 30 + fn + ex, cs)
    missing = [n for n in NAMES if n not in found]
    assert not missing, missing
    return found


def iter_lines(data, start, csize, chunk=1 << 22):
    d = zlib.decompressobj(-15)
    buf = b""
    for p in range(start, start + csize, chunk):
        buf += d.decompress(data[p:min(p + chunk, start + csize)])
        *lines, buf = buf.split(b"\n")
        for ln in lines:
            if ln.strip():
                yield ln
    buf += d.flush()
    if buf.strip():
        yield buf


def event_embedding(csv_bytes, dim):
    import csv, io
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8"))))[1:]
    text = [r[3] for r in rows]
    ids = [int(r[0]) for r in rows]
    assert ids == list(range(len(ids))), "event ids are not 0..n-1"
    X = TfidfVectorizer(token_pattern=r"[A-Za-z]{2,}", lowercase=True).fit_transform(text)
    k = min(dim, X.shape[1] - 1, X.shape[0] - 1)
    Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    out = np.zeros((len(ids), dim), dtype=np.float32)
    out[:, :k] = Z
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--emb_dim", default=300, type=int)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = open(a.zip, "rb").read()
    ent = find_entries(data)

    s, cs = ent["id_url+type.csv"]
    emb = event_embedding(zlib.decompressobj(-15).decompress(data[s:s + cs]), a.emb_dim)

    nodes, edges, n_nodes, n_edges = [], [], [], []
    tid, is_normal, etype = [], [], []
    for name in NAMES[1:]:
        s, cs = ent[name]
        c = 0
        for ln in iter_lines(data, s, cs):
            g = json.loads(ln)
            ni = g["node_info"]
            nodes.append(np.fromiter((int(n[-1]) for n in ni), dtype=np.int16, count=len(ni)))
            ei = np.asarray(g["edge_index"], dtype=np.int32).reshape(-1, 2)
            edges.append(ei)
            n_nodes.append(len(ni))
            n_edges.append(len(ei))
            tid.append(g["trace_id"])
            is_normal.append(bool(g["trace_bool"]))
            etype.append(g["error_trace_type"])
            c += 1
        print(name, c, flush=True)
    np.savez(os.path.join(a.out_dir, "compact.npz"),
             nodes=np.concatenate(nodes), edges=np.concatenate(edges),
             n_nodes=np.array(n_nodes, dtype=np.int32), n_edges=np.array(n_edges, dtype=np.int32),
             trace_id=np.array(tid), is_normal=np.array(is_normal), error_type=np.array(etype),
             event_emb=emb)
    print("traces", len(tid), "normal", int(np.sum(is_normal)), "mean nodes", float(np.mean(n_nodes)),
          "max nodes", int(np.max(n_nodes)), "events", emb.shape)


if __name__ == "__main__":
    main()
