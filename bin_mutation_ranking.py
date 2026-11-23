"""Rank mutations by the impact bin they fall into.

Run: python bin_mutation_ranking.py
Install once: pip install numpy pandas scipy
Put variants_qbinned.tsv in the same folder as this script.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact

# ---- Settings you may need to change ---------------------------------------
BIN_COL = "bin"        # name of the column holding the 12 bins
BIN_ORDER = None        # e.g. list(range(1, 13)); None = sorted unique values
WT_SEQ = None           # paste the wild-type sequence, or None to infer it
                        # (inferred = most common base at each position)
TOP_N = 5               # mutations to report per bin
MIN_COUNT = 5           # ignore mutations seen fewer times than this in a bin
# -----------------------------------------------------------------------------

# 1. Load sequences, scores and bins.
data = pd.read_csv(Path(__file__).with_name("variants_qbinned.tsv"), sep="\t",
                   usecols=["sequence", "score", BIN_COL])
letters = np.array([list(s.upper()) for s in data["sequence"]])
n_rna, n_pos = letters.shape
bins = BIN_ORDER if BIN_ORDER is not None else sorted(data[BIN_COL].unique())
print(f"{n_rna:,} RNAs, {n_pos} positions, {len(bins)} bins: {bins}")

# 2. Wild-type sequence (provided or consensus; mutations are rare so the
# consensus equals the wild type).
if WT_SEQ:
    wt = np.array(list(WT_SEQ.upper()))
    assert len(wt) == n_pos, "WT_SEQ length does not match the sequences"
else:
    wt = np.array([pd.Series(letters[:, p]).mode().iloc[0] for p in range(n_pos)])

# 3. Mutation indicator matrix: one column per (position, mutant base).
base_list = ["A", "C", "G", "T"]
mut_names, cols = [], []
for p in range(n_pos):
    for b in base_list:
        if b == wt[p]:
            continue
        mut_names.append((p + 1, wt[p], b))           # (position, wt, mutant)
        cols.append(sparse.csr_matrix((letters[:, p] == b).astype(np.float32)).T)
M = sparse.hstack(cols).tocsr()                        # RNAs x mutations

# 4. Bin membership matrix (RNAs x bins) and counts of each mutation per bin.
bin_idx = data[BIN_COL].map({b: i for i, b in enumerate(bins)}).to_numpy()
B = sparse.csr_matrix((np.ones(n_rna), (np.arange(n_rna), bin_idx)),
                      shape=(n_rna, len(bins)))
counts = (M.T @ B).toarray()                           # mutations x bins
bin_sizes = np.asarray(B.sum(axis=0)).ravel()
mut_totals = counts.sum(axis=1)

# 5. Per-bin summary: how mutated is each bin, and what are its scores?
muts_per_rna = np.asarray(M.sum(axis=1)).ravel()
print("\nBIN SUMMARY")
print(f"{'bin':>5} {'RNAs':>9} {'mean score':>11} {'mean #muts':>11}")
for i, b in enumerate(bins):
    sel = bin_idx == i
    print(f"{b!s:>5} {sel.sum():>9,} {data['score'][sel].mean():>11.4f} "
          f"{muts_per_rna[sel].mean():>11.3f}")

# 6. For every bin, rank mutations by enrichment vs. the rest of the dataset.
# Enrichment = log2(frequency of mutation in this bin / frequency elsewhere),
# with a pseudocount; Fisher's exact test gives a p-value for each.
rows = []
for i, b in enumerate(bins):
    n_in, n_out = bin_sizes[i], n_rna - bin_sizes[i]
    for m, (pos, w, mt) in enumerate(mut_names):
        a = counts[m, i]                               # mutation in this bin
        if a < MIN_COUNT:
            continue
        c = mut_totals[m] - a                          # mutation in other bins
        log2_enrich = np.log2(((a + 0.5) / (n_in + 1)) / ((c + 0.5) / (n_out + 1)))
        p = fisher_exact([[a, n_in - a], [c, n_out - c]])[1]
        rows.append((b, pos, f"{w}{pos}{mt}", int(a), a / n_in, log2_enrich, p))

res = pd.DataFrame(rows, columns=["bin", "position", "mutation", "count_in_bin",
                                  "freq_in_bin", "log2_enrichment", "p_value"])
res["q_value"] = np.minimum(1, res["p_value"] * len(res) /
                            res["p_value"].rank(method="first"))  # approx. BH
res.to_csv(Path(__file__).with_name("bin_mutation_enrichment.tsv"),
           sep="\t", index=False)

print(f"\nTOP {TOP_N} ENRICHED MUTATIONS IN EACH BIN "
      "(full table saved to bin_mutation_enrichment.tsv)")
for b in bins:
    sub = res[(res["bin"] == b) & (res["log2_enrichment"] > 0)]
    sub = sub.sort_values("log2_enrichment", ascending=False).head(TOP_N)
    print(f"\nBin {b}  (n = {int(bin_sizes[bins.index(b)]):,})")
    if sub.empty:
        print("  no enriched mutations meeting the minimum count")
    for _, r in sub.iterrows():
        print(f"  {r.mutation:<10} count={r.count_in_bin:<5} "
              f"freq={r.freq_in_bin:.4f}  log2 enrich={r.log2_enrichment:+.2f}  "
              f"q={r.q_value:.2g}")

# 7. Rank bins themselves by how strongly mutations separate them from the rest
# (number of significantly enriched mutations, q < 0.05).
sig = res[(res["q_value"] < 0.05) & (res["log2_enrichment"] > 0)]
print("\nBINS RANKED BY NUMBER OF SIGNIFICANTLY ENRICHED MUTATIONS")
for b, n in sig.groupby("bin").size().reindex(bins, fill_value=0) \
               .sort_values(ascending=False).items():
    print(f"  Bin {b}: {n}")


