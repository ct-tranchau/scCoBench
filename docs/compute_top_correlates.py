"""
Compute full all-vs-all Pearson and Spearman correlations among all expressed
genes in each Arabidopsis sample, then save the TOP 50 most correlated
partners per gene to compressed JSON for the website lookup tab.

Full matrices are NOT saved to disk (~4 GB each) — only the top-50 lookups.
"""
import os, gc, gzip, json, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

COUNT_DIR = "/Volumes/T7 Shield/GFP_2024_V1/GEO_submission_scRNAseq"
OUT_DIR   = "/Volumes/T7 Shield/GFP_2024_V1/Binomial_thining/website/gene_corr_full"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES   = ["WER", "CORTEX", "PET111", "SCR"]
TOP_N     = 50
MIN_CELLS = 10


def normalize_rows(X):
    X = X.astype(np.float32, copy=True)
    X -= X.mean(axis=1, keepdims=True)
    n = np.sqrt((X * X).sum(axis=1, keepdims=True))
    n[n == 0] = 1.0
    X /= n
    return X


def top_n_per_row(R, n=50):
    """For each row of R, return [[partner_idx, r*100_int], ...] sorted desc by r."""
    N = R.shape[0]
    np.fill_diagonal(R, -2.0)  # exclude self from "top"
    out = [None] * N
    for i in range(N):
        # argpartition gives unsorted top-n; then sort just those n
        idx = np.argpartition(-R[i], n)[:n]
        ordering = np.argsort(-R[i, idx])
        sorted_idx = idx[ordering]
        out[i] = [[int(j), int(round(float(R[i, j]) * 100))] for j in sorted_idx]
    return out


for sample in SAMPLES:
    print(f"\n=== {sample} ===")
    t_sample = time.time()

    t0 = time.time()
    df = pd.read_csv(os.path.join(COUNT_DIR, f"count_matrix_{sample}.csv"),
                     index_col=0)
    M = df.values.astype(np.float32)
    genes_all = df.index.to_numpy()
    print(f"  loaded: {M.shape} in {time.time()-t0:.1f}s")

    n_express = (M > 0).sum(axis=1)
    keep = n_express >= MIN_CELLS
    M = M[keep]
    genes = genes_all[keep].tolist()
    N = M.shape[0]
    print(f"  expressed (>= {MIN_CELLS} cells): {N}/{len(genes_all)}")

    mean_expr = M.mean(axis=1)
    n_express_kept = n_express[keep].astype(int).tolist()

    # ---- Pearson ----
    print("  Pearson...")
    t0 = time.time()
    Mn = normalize_rows(M)
    print(f"    normalize: {time.time()-t0:.1f}s")
    t0 = time.time()
    R = (Mn @ Mn.T).astype(np.float32)
    print(f"    GEMM: {time.time()-t0:.1f}s, R shape {R.shape}")
    del Mn; gc.collect()
    t0 = time.time()
    pearson_top = top_n_per_row(R, TOP_N)
    print(f"    top-{TOP_N} extraction: {time.time()-t0:.1f}s")
    del R; gc.collect()

    # ---- Spearman ----
    print("  Spearman...")
    t0 = time.time()
    Mr = np.empty_like(M)
    for i in range(N):
        Mr[i] = rankdata(M[i])
    print(f"    rank rows: {time.time()-t0:.1f}s")
    t0 = time.time()
    Mrn = normalize_rows(Mr)
    print(f"    normalize: {time.time()-t0:.1f}s")
    del Mr; gc.collect()
    t0 = time.time()
    R = (Mrn @ Mrn.T).astype(np.float32)
    print(f"    GEMM: {time.time()-t0:.1f}s")
    del Mrn; gc.collect()
    t0 = time.time()
    spearman_top = top_n_per_row(R, TOP_N)
    print(f"    top-{TOP_N} extraction: {time.time()-t0:.1f}s")
    del R; gc.collect()

    # ---- Save ----
    out = {
        "sample":       sample,
        "n_genes":      N,
        "top_n":        TOP_N,
        "genes":        genes,
        "mean_expr":    [round(float(x), 3) for x in mean_expr],
        "n_express":    n_express_kept,
        "pearson_top":  pearson_top,
        "spearman_top": spearman_top,
    }
    out_path = os.path.join(OUT_DIR, f"{sample}.json.gz")
    t0 = time.time()
    with gzip.open(out_path, "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB) in {time.time()-t0:.1f}s")

    del M, out, pearson_top, spearman_top
    gc.collect()

    print(f"  total: {time.time()-t_sample:.1f}s")

print("\nDone.")
