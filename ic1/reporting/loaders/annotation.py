"""Load the *human* in/out annotations into tidy frames for the docs.

Unlike the other in/out loaders (which score the LLM/ML systems), this one
describes the human screening effort itself: how many annotations there are,
who did them, how well the coders agree, and how each coder compares to the
adjudicated ``RESOLVED`` value.

Source is the flat export ``data/exports/inout.csv``: one row per (item, coder)
with a one-hot include/exclude pair. Decision encoding — a document is *included*
whenever ``incl|1 == 1`` (this also covers the handful of rows where both flags
are set), *excluded* when only ``incl|0 == 1``, and *missing* when neither flag
is set (an abstention). The special coder ``RESOLVED`` carries the adjudicated
gold value, not a coder decision, and is split out.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from statsmodels.stats.inter_rater import fleiss_kappa

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INOUT_CSV = _REPO_ROOT / "data" / "exports" / "inout.csv"

GOLD_USER = "RESOLVED"


def _decision(incl1: float, incl0: float) -> float:
    """One-hot include/exclude pair -> 1 (include) / 0 (exclude) / NaN (missing).

    Include wins whenever ``incl|1`` is set (covers the few contradictory rows
    where both flags are 1); exclude needs ``incl|0`` alone; anything else (both
    blank) is an abstention.
    """
    if incl1 == 1:
        return 1.0
    if incl0 == 1:
        return 0.0
    return np.nan


def load_inout_annotations(csv: Path | str | None = None) -> pd.DataFrame:
    """Long frame of every human annotation, one row per (item, coder).

    Columns: ``item_id``, ``username``, ``decision`` (1/0/NaN), ``is_gold``
    (the adjudicated ``RESOLVED`` row). Missing decisions are kept as NaN so
    counts can report abstentions; drop them before scoring.
    """
    path = Path(csv) if csv is not None else DEFAULT_INOUT_CSV
    df = pd.read_csv(path)
    df["decision"] = [
        _decision(a, b) for a, b in zip(df["incl|1"], df["incl|0"], strict=True)
    ]
    df["is_gold"] = df.username == GOLD_USER
    return df[["item_id", "username", "decision", "is_gold"]].reset_index(drop=True)


def _coders(ann: pd.DataFrame) -> pd.DataFrame:
    """Coder rows only (drops the adjudicated gold rows)."""
    return ann[~ann.is_gold]


def _gold(ann: pd.DataFrame) -> pd.Series:
    """item_id -> adjudicated decision (the ``RESOLVED`` value)."""
    return ann[ann.is_gold].set_index("item_id").decision


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def inout_coder_counts(ann: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-coder annotation counts, split by decision.

    One row per coder: ``n`` (total annotations), ``n_include``, ``n_exclude``,
    ``n_missing``, sorted by ``n`` descending. Powers the "how many per coder"
    bar.
    """
    ann = load_inout_annotations() if ann is None else ann
    cod = _coders(ann)
    g = cod.groupby("username")
    out = pd.DataFrame({
        "n": g.size(),
        "n_include": g.decision.apply(lambda s: int((s == 1).sum())),
        "n_exclude": g.decision.apply(lambda s: int((s == 0).sum())),
        "n_missing": g.decision.apply(lambda s: int(s.isna().sum())),
    })
    return out.sort_values("n", ascending=False).reset_index()


def _nominal_sets(cod: pd.DataFrame) -> pd.Series:
    """item_id -> tuple of the coders assigned to it (the nominal team).

    Keyed off *membership*, not who happened to answer, so an abstention does
    not fragment a team into a new set.
    """
    return cod.groupby("item_id").username.apply(lambda s: tuple(sorted(s.unique())))


def inout_coderset_composition(ann: pd.DataFrame | None = None) -> pd.DataFrame:
    """What the 5,000 items are made of, one row per nominal coder-set.

    Columns: ``coders`` (tuple), ``label`` (compact ``c00+c03+c04`` string),
    ``size`` (number of coders in the set), ``n_items``. Sorted by ``n_items``
    descending.
    """
    ann = load_inout_annotations() if ann is None else ann
    sets = _nominal_sets(_coders(ann))
    counts = sets.value_counts()
    comp = pd.DataFrame({"coders": counts.index.tolist(), "n_items": counts.to_numpy()})
    comp["size"] = comp.coders.apply(len)
    comp["label"] = comp.coders.apply(
        lambda cs: "+".join(c.replace("coder_0", "c").replace("coder_", "c") for c in cs)
    )
    return comp[["coders", "label", "size", "n_items"]].reset_index(drop=True)


def inout_screening_raster(ann: pd.DataFrame | None = None) -> dict:
    """Build the document x coder screening raster and its ordering.

    Every one of the 5,000 documents becomes a column and every coder a row.
    Columns are grouped by nominal coder-set, and *within* each set sorted by
    include-fraction so each block reads left->right as unanimous-exclude ->
    split votes -> unanimous-include (disagreement forms a fringe at the block's
    right edge). Returns a dict of aligned arrays:

    - ``matrix``  : (n_coders, n_items) float, 1=include / 0=exclude / NaN=not
                    assigned or abstained.
    - ``coders``  : row labels (sorted coder usernames).
    - ``gold``    : (n_items,) the adjudicated decision per column (NaN if none).
    - ``split``   : (n_items,) vote split, 0 (unanimous) .. 1 (evenly split),
                    NaN where fewer than two decisions.
    - ``n_items`` : column count.
    """
    ann = load_inout_annotations() if ann is None else ann
    cod = _coders(ann)
    gold = _gold(ann)
    coders = sorted(cod.username.unique())
    row = {c: i for i, c in enumerate(coders)}

    sets = _nominal_sets(cod)
    dec = cod.dropna(subset=["decision"])
    n_inc = dec[dec.decision == 1].groupby("item_id").size().reindex(sets.index, fill_value=0)
    n_dec = dec.groupby("item_id").size().reindex(sets.index, fill_value=0)
    frac = n_inc / n_dec.replace(0, np.nan)
    split = 1.0 - 2.0 * (frac - 0.5).abs()  # 0 unanimous .. 1 evenly split

    order = pd.DataFrame({
        "set": sets.astype(str), "frac": frac.fillna(-1.0), "n_dec": n_dec,
    }).sort_values(
        ["set", "frac", "n_dec"], ascending=[True, True, False], kind="stable"
    )
    items = order.index.tolist()
    col = {it: i for i, it in enumerate(items)}

    matrix = np.full((len(coders), len(items)), np.nan)
    for it, u, d in zip(cod.item_id, cod.username, cod.decision, strict=True):
        if pd.notna(d):
            matrix[row[u], col[it]] = d

    return {
        "matrix": matrix,
        "coders": coders,
        "gold": gold.reindex(items).to_numpy(dtype=float),
        "split": split.reindex(items).to_numpy(dtype=float),
        "n_items": len(items),
    }


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def _fleiss(counts: np.ndarray) -> float:
    """Fleiss' kappa on a subjects x categories count matrix (silences the
    divide-by-zero warning statsmodels emits on a degenerate single-category
    set)."""
    if len(counts) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fleiss_kappa(counts))


def _count_matrix(cod_nonmissing: pd.DataFrame, size: int | None = None) -> np.ndarray:
    """[[n_exclude, n_include], ...] per item; optionally keep only items with
    exactly ``size`` ratings (complete cases, so every row has equal raters)."""
    g = cod_nonmissing.groupby("item_id").decision
    inc = g.apply(lambda s: int((s == 1).sum()))
    exc = g.apply(lambda s: int((s == 0).sum()))
    m = pd.DataFrame({"exc": exc, "inc": inc})
    if size is not None:
        m = m[m.sum(axis=1) == size]
    return m[["exc", "inc"]].to_numpy()


def _pooled_three_rater(cod: pd.DataFrame) -> pd.DataFrame:
    """Down-sample to a uniform 3 raters per item for a single global Fleiss.

    Items with >=3 non-missing decisions are kept; on 4-decision items the
    alphabetically-last coder's rating is dropped (deterministic, so the number
    is reproducible). Items with <3 decisions are excluded from the pooled
    estimate (they still count toward per-set / pairwise / percent agreement).
    """
    nm = cod.dropna(subset=["decision"])
    per = nm.groupby("item_id").username.nunique()
    keep = nm[nm.item_id.isin(per[per >= 3].index)].copy()
    last = keep.groupby("item_id").username.transform(
        lambda s: sorted(s.unique())[-1] if s.nunique() >= 4 else ""
    )
    return keep[keep.username != last]


def inout_overall_agreement(ann: pd.DataFrame | None = None) -> dict:
    """Corpus-wide agreement: pooled 3-rater Fleiss' kappa and % unanimity.

    Returns ``{"fleiss", "n_items", "pct_agreement", "n_items_agreement"}``.
    ``fleiss`` pools to a uniform three raters (see ``_pooled_three_rater``);
    ``pct_agreement`` is the share of items (>=2 raters) whose coders were
    unanimous.
    """
    ann = load_inout_annotations() if ann is None else ann
    cod = _coders(ann)
    pooled = _count_matrix(_pooled_three_rater(cod), size=3)

    nm = cod.dropna(subset=["decision"])
    full = pd.DataFrame(
        _count_matrix(nm), columns=["exc", "inc"]
    ) if len(nm) else pd.DataFrame(columns=["exc", "inc"])
    n = full.sum(axis=1)
    multi = full[n >= 2]
    unanimous = (multi.max(axis=1) == multi.sum(axis=1))
    return {
        "fleiss": _fleiss(pooled),
        "n_items": int(len(pooled)),
        "pct_agreement": float(unanimous.mean()) if len(multi) else float("nan"),
        "n_items_agreement": int(len(multi)),
    }


def inout_agreement_by_set(ann: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per nominal coder-set agreement: Fleiss' kappa on complete-case items.

    One row per set (>=2 coders) with ``label``, ``size``, ``n_items``
    (nominal), ``n_complete`` (items where every set member answered), ``fleiss``,
    ``pct_agreement`` and ``resolved_incl_rate`` (share of the set's documents
    the adjudicated value included). Sets whose complete-case items are
    single-category yield NaN kappa (kept, flagged by the NaN). Sorted by
    ``n_items`` desc.
    """
    ann = load_inout_annotations() if ann is None else ann
    cod = _coders(ann)
    gold = _gold(ann)
    comp = inout_coderset_composition(ann)
    sets = _nominal_sets(cod)
    nm = cod.dropna(subset=["decision"])

    rows = []
    for _, r in comp.iterrows():
        cs = r.coders
        if len(cs) < 2:
            continue
        items = sets[sets == cs].index
        sub = nm[nm.item_id.isin(items)]
        counts = _count_matrix(sub, size=len(cs))
        n_complete = len(counts)
        pct = float((counts.max(1) == counts.sum(1)).mean()) if n_complete else float("nan")
        g = gold.reindex(items).dropna()
        rows.append({
            "label": r.label, "size": len(cs), "n_items": r.n_items,
            "n_complete": n_complete,
            "fleiss": _fleiss(counts) if n_complete else float("nan"),
            "pct_agreement": pct,
            "resolved_incl_rate": float(g.mean()) if len(g) else float("nan"),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def inout_unanimous_dispersion(
    ann: pd.DataFrame | None = None, min_unan: int = 30
) -> dict:
    """Test whether coder-sets over-/under-include *as a block* on the documents
    they screened unanimously.

    On unanimous documents the adjudicated value just equals the set's consensus
    (never independently checked — see the annotation docs), so a set that shares
    a directional bias bakes it into the gold undetected. Under the null that
    every set drew from a common pool and screened without bias, each set's
    unanimous *inclusion* rate (unanimous-include ÷ unanimous docs) should scatter
    around a pooled rate with only binomial noise. Between-set spread beyond that
    is over-dispersion — the fingerprint of block-level bias (or, if documents
    were not randomly allocated, of genuinely different batch prevalence).

    Each set also carries ``resolved_rate`` — the adjudicated inclusion rate over
    *all* the set's documents (the best truth proxy; independent of the set on
    its contested documents). Comparing ``rate`` (unanimous consensus) to
    ``resolved_rate`` separates *bias* from *agreement-on-positives*: a set whose
    unanimous rate is high but whose resolved rate is ordinary is well-aligned,
    not over-including.

    Returns ``{"sets", "pooled", "pooled_resolved", "chi2", "df", "pval", "phi",
    "i2"}`` where ``sets`` is a DataFrame (``label``, ``size``, ``n_unan``,
    ``n_all``, ``k_incl``, ``rate``, ``resolved_rate``, ``z``) sorted by ``rate``
    desc, ``phi`` = χ²/df (1 = chance), and ``i2`` the heterogeneity fraction.
    Only sets with >= ``min_unan`` unanimous documents are included.
    """
    from scipy import stats

    ann = load_inout_annotations() if ann is None else ann
    cod = _coders(ann).dropna(subset=["decision"])
    gold = _gold(ann)
    sets = _nominal_sets(cod)
    labels = {r.coders: r.label for _, r in inout_coderset_composition(ann).iterrows()}

    g = cod.groupby("item_id").decision
    n = g.size()
    ninc = g.apply(lambda s: int((s == 1).sum()))
    item = pd.DataFrame({"n": n, "ninc": ninc})
    item["unan_inc"] = item.ninc == item.n
    item["unanimous"] = (item.ninc == 0) | (item.ninc == item.n)
    item["set"] = sets.reindex(item.index)
    item["gold"] = gold.reindex(item.index)

    rows = []
    for cs, grp in item.groupby(item["set"]):
        if not isinstance(cs, tuple) or len(cs) < 3:
            continue
        u = grp[grp.unanimous]
        if len(u) < min_unan:
            continue
        g_all = grp.gold.dropna()
        rows.append({
            "label": labels.get(cs, ",".join(cs)), "size": len(cs),
            "n_unan": int(len(u)), "n_all": int(len(grp)),
            "k_incl": int(u.unan_inc.sum()),
            "resolved_rate": float(g_all.mean()) if len(g_all) else float("nan"),
        })
    df = pd.DataFrame(rows)
    df["rate"] = df.k_incl / df.n_unan

    dof = len(df) - 1

    def _dispersion(k: pd.Series, nrows: pd.Series) -> dict:
        """χ² heterogeneity of a set of proportions k/nrows around their pool."""
        pool = float(k.sum() / nrows.sum())
        chi2 = float((((k - nrows * pool) ** 2) / (nrows * pool * (1 - pool))).sum())
        return {
            "pooled": pool, "chi2": chi2,
            "pval": float(1 - stats.chi2.cdf(chi2, dof)),
            "phi": chi2 / dof if dof else float("nan"),
            "i2": max(0.0, (chi2 - dof) / chi2) if chi2 > 0 else 0.0,
        }

    un = _dispersion(df.k_incl, df.n_unan)
    df["z"] = (df.rate - un["pooled"]) / np.sqrt(
        un["pooled"] * (1 - un["pooled"]) / df.n_unan)
    res = _dispersion((df.resolved_rate * df.n_all).round(), df.n_all)
    return {
        "sets": df.sort_values("rate", ascending=False).reset_index(drop=True),
        "pooled": un["pooled"], "pooled_resolved": res["pooled"], "df": dof,
        "chi2": un["chi2"], "pval": un["pval"], "phi": un["phi"], "i2": un["i2"],
        "resolved_chi2": res["chi2"], "resolved_pval": res["pval"],
        "resolved_phi": res["phi"], "resolved_i2": res["i2"],
    }


def inout_pairwise_kappa(
    ann: pd.DataFrame | None = None, min_shared: int = 30
) -> pd.DataFrame:
    """Cohen's kappa between every pair of coders on their co-annotated items.

    Square matrix indexed/columned by coder; the diagonal is 1.0. Pairs sharing
    fewer than ``min_shared`` items are left NaN (too sparse to read). Built from
    complete pairwise cases (both coders gave a non-missing decision).
    """
    ann = load_inout_annotations() if ann is None else ann
    nm = _coders(ann).dropna(subset=["decision"])
    wide = nm.pivot_table(index="item_id", columns="username", values="decision", aggfunc="first")
    coders = sorted(wide.columns)
    mat = pd.DataFrame(np.nan, index=coders, columns=coders, dtype=float)
    for c in coders:
        mat.loc[c, c] = 1.0
    for a, b in combinations(coders, 2):
        pair = wide[[a, b]].dropna()
        if len(pair) < min_shared or pair[a].nunique() < 2 or pair[b].nunique() < 2:
            continue
        k = cohen_kappa_score(pair[a], pair[b])
        mat.loc[a, b] = mat.loc[b, a] = k
    return mat


# ---------------------------------------------------------------------------
# F1 vs the adjudicated (RESOLVED) value
# ---------------------------------------------------------------------------


def inout_coder_f1(ann: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-coder precision/recall/F1 against the adjudicated gold value.

    One row per coder over the items that coder decided (gold exists for all),
    plus two summary rows — ``average coder`` (macro mean of the per-coder
    scores) and ``pooled`` (every coder decision scored in one confusion matrix,
    i.e. the micro / typical-annotation score). ``n`` is the scored item count.
    Sorted coders by F1 desc, summaries last.
    """
    ann = load_inout_annotations() if ann is None else ann
    gold = _gold(ann)
    nm = _coders(ann).dropna(subset=["decision"])

    rows = []
    for u, g in nm.groupby("username"):
        yt = gold.reindex(g.item_id).to_numpy()
        yp = g.decision.to_numpy()
        keep = ~np.isnan(yt)
        yt, yp = yt[keep].astype(int), yp[keep].astype(int)
        if len(yt) == 0:
            continue
        rows.append({
            "coder": u, "n": int(len(yt)),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
        })
    per = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    # Pooled (micro): all coder decisions vs their item's gold, one matrix.
    yt = gold.reindex(nm.item_id).to_numpy()
    yp = nm.decision.to_numpy()
    keep = ~np.isnan(yt)
    yt, yp = yt[keep].astype(int), yp[keep].astype(int)
    summaries = pd.DataFrame([
        {
            "coder": "average coder", "n": int(per.n.mean()),
            "precision": per.precision.mean(), "recall": per.recall.mean(),
            "f1": per.f1.mean(),
        },
        {
            "coder": "pooled", "n": int(len(yt)),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
        },
    ])
    return pd.concat([per, summaries], ignore_index=True)
