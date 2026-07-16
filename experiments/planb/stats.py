"""
Uncertainty for the Plan B analyses.

Every correlation in this repo used to be a bare point estimate from one decode of
one seed — `spearmanr(x, y)[0]`, p-value discarded. On the controlled sweeps that is
n=6 clips, so "reverb -0.27 -> -0.80" was reported with no way to tell it from noise.
This module supplies the missing intervals.

Two things here are easy to get wrong and are handled explicitly:

1. CLUSTERING. The sweeps have ~108 ROWS but only ~6 CLIPS: each clip appears once
   per (family, severity) cell. Rows from one clip are not independent — a talker the
   model happens to rate low is low in every cell. Resampling rows would treat n=108
   as the sample size and report intervals ~4x too narrow. `cluster` (default: the
   clip id) makes the bootstrap resample CLIPS with replacement, which is the honest
   unit of independent evidence. This is a big deal here, not a technicality: with 6
   clusters the CIs come out wide, and that is the finding.

2. PAIRED COMPARISON. "Is model B better calibrated than model A" is a question about
   the DIFFERENCE of two correlations measured on the same clips, so the models'
   errors are correlated and two independent CIs cannot answer it (overlapping CIs do
   not imply a non-significant delta). `paired_delta_ci` bootstraps the difference
   directly, on the common support only.

With ~6 clusters, a percentile bootstrap is approximate — treat these as descriptive
uncertainty, not exact coverage. The honest fix for that is more speakers, not a
cleverer interval.
"""

import numpy as np
from scipy.stats import spearmanr

N_BOOT = 10_000
SEED = 0


def _clean(x, y, cluster=None):
    """Drop pairs where either value is missing/non-finite."""
    x = np.asarray([np.nan if v is None else float(v) for v in x], dtype=float)
    y = np.asarray([np.nan if v is None else float(v) for v in y], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    c = np.asarray(cluster if cluster is not None else np.arange(len(x)), dtype=object)
    return x[ok], y[ok], c[ok]


def _boot_indices(cluster, rng):
    """One cluster-bootstrap resample: draw clusters with replacement."""
    groups = {}
    for i, c in enumerate(cluster):
        groups.setdefault(c, []).append(i)
    keys = list(groups)
    drawn = rng.integers(len(keys), size=len(keys))
    idx = []
    for d in drawn:
        idx.extend(groups[keys[d]])
    return np.asarray(idx, dtype=int)


def spearman_ci(x, y, cluster=None, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Spearman rho with a cluster-bootstrap percentile CI.

    Returns dict(rho, lo, hi, p_naive, n, n_clusters). rho is NaN if n < 3.

    `p_naive` is scipy's p-value, which assumes every ROW is independent. When
    `cluster` is passed that assumption is false by construction, so p_naive is
    reported only for continuity with the old numbers and must not be used to call
    significance — prefer the CI. The two can disagree (they answer different
    questions and rest on different independence assumptions); when they do, the CI
    is the one that knows about the clustering.
    """
    x, y, c = _clean(x, y, cluster)
    n, k = len(x), len({*c.tolist()})
    if n < 3:
        return {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_naive": float("nan"), "n": n, "n_clusters": k}
    res = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = _boot_indices(c, rng)
        if len(idx) < 3:
            continue
        xb, yb = x[idx], y[idx]
        # a resample can be constant (all one clip, one severity) -> rho undefined
        if np.all(xb == xb[0]) or np.all(yb == yb[0]):
            continue
        boots.append(spearmanr(xb, yb)[0])
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    lo, hi = (np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if len(boots) > 50 else (float("nan"), float("nan")))
    return {"rho": float(res[0]), "lo": float(lo), "hi": float(hi),
            "p_naive": float(res[1]), "n": n, "n_clusters": k}


def paired_delta_ci(x, y_a, y_b, cluster=None, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """CI for rho(x, y_b) - rho(x, y_a) on the clips BOTH models scored.

    Restricting to common support matters: the original model emits unparseable MOS on
    some clips, and scoring the two models over different row sets is exactly the
    confound that makes a raw before/after delta unreadable. n below is the common
    support, which may be smaller than either model's own n.

    `p` is the fraction of bootstrap deltas on the far side of 0 (two-sided), i.e. an
    inversion of the percentile CI, NOT an exact test.
    """
    xa = np.asarray([np.nan if v is None else float(v) for v in x], dtype=float)
    ya = np.asarray([np.nan if v is None else float(v) for v in y_a], dtype=float)
    yb = np.asarray([np.nan if v is None else float(v) for v in y_b], dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(yb)
    c = np.asarray(cluster if cluster is not None else np.arange(len(xa)), dtype=object)[ok]
    xa, ya, yb = xa[ok], ya[ok], yb[ok]
    n, k = len(xa), len({*c.tolist()})
    nan = float("nan")
    if n < 3:
        return {"delta": nan, "lo": nan, "hi": nan, "p": nan, "n": n, "n_clusters": k,
                "rho_a": nan, "rho_b": nan}
    rho_a, rho_b = spearmanr(xa, ya)[0], spearmanr(xa, yb)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = _boot_indices(c, rng)
        if len(idx) < 3:
            continue
        xs, as_, bs = xa[idx], ya[idx], yb[idx]
        if np.all(xs == xs[0]) or np.all(as_ == as_[0]) or np.all(bs == bs[0]):
            continue
        d = spearmanr(xs, bs)[0] - spearmanr(xs, as_)[0]
        if np.isfinite(d):
            boots.append(d)
    boots = np.asarray(boots)
    if len(boots) <= 50:
        return {"delta": float(rho_b - rho_a), "lo": nan, "hi": nan, "p": nan,
                "n": n, "n_clusters": k, "rho_a": float(rho_a), "rho_b": float(rho_b)}
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"delta": float(rho_b - rho_a), "lo": float(lo), "hi": float(hi),
            "p": float(min(p, 1.0)), "n": n, "n_clusters": k,
            "rho_a": float(rho_a), "rho_b": float(rho_b)}


def fmt(d):
    """'+0.83 [+0.41, +0.96] (n=108, 6 clips)'

    Deliberately prints the CI, not p_naive: with clustered rows the CI is the
    defensible summary. paired_delta_ci's `p` IS bootstrap-derived, so it is shown.
    """
    v = d.get("rho", d.get("delta", float("nan")))
    if not np.isfinite(v):
        return f"n/a (n={d['n']})"
    s = f"{v:+.3f}"
    if np.isfinite(d["lo"]):
        s += f" [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    if "p" in d and np.isfinite(d["p"]):
        s += f" p={d['p']:.3f}"
    return s + f" (n={d['n']}, {d['n_clusters']} clips)"


def n_clusters_warning(k, floor=12):
    """Percentile bootstrap needs a decent number of independent units.

    With k clusters the resample can only ever draw from k distinct values, so with
    k~6 the interval is approximate and biased NARROW — it will overstate confidence.
    Returns a warning string, or None.
    """
    if k < floor:
        return (f"WARNING: only {k} independent clusters — percentile-bootstrap CIs are "
                f"approximate and likely too narrow. Treat as descriptive, not inferential; "
                f"the real fix is more speakers/clips, not a different interval.")
    return None
