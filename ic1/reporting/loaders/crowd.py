"""Score the crowd screening against the adjudicated gold, for the docs.

The crowd received the in/out screening documents as a RIS file split into two
buckets, so *the file a record came back in is the crowd's decision*:

- ``DESTINY_Crowd_possibly_relevant.txt`` -> the crowd *included* the document
- ``DESTINY_Crowd_exclude.txt``           -> the crowd *excluded* it

Each record carries the nacsos ``item_id`` in the ``MISC3`` tag. Gold is the
adjudicated ``RESOLVED`` value (``incl|1``) from the shareable resolved export,
so this loader never depends on the (private) gold baked into the RIS ``MISC4``.

The crowd's 1,000 documents are NOT the held-out test set -- they cut across
every split -- so we score two scopes: every crowd-annotated document (the
crowd's delivered performance) and the test-split subset (a like-for-like row
against the automated systems, which are scored on that same test set).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from ic1.evaluation_splits.splits_model import EvaluationSplits

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CROWD_DIR = _REPO_ROOT / "data" / "private" / "crowd"
DEFAULT_RESOLVED_CSV = _REPO_ROOT / "data" / "exports" / "inout_resolved.csv"
DEFAULT_SPLITS_PATH = _REPO_ROOT / "data" / "exports" / "inout_splits.json"

_INCLUDE_FILE = "DESTINY_Crowd_possibly_relevant.txt"
_EXCLUDE_FILE = "DESTINY_Crowd_exclude.txt"

GOLD_USER = "RESOLVED"


def _parse_ris(path: Path) -> list[dict[str, str]]:
    """Minimal RIS reader for the crowd export.

    The crowd files use non-standard multi-char tags (``MISC3``/``MISC4``) that
    rispy's tag regex rejects, and the format is trivial -- ``TAG  - value``
    lines, records terminated by an ``ER  -`` line -- so we parse it directly.
    """
    records: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("ER  -"):
            if cur:
                records.append(cur)
                cur = {}
            continue
        if "  - " in line:
            tag, val = line.split("  - ", 1)
            cur[tag.strip()] = val.strip()
    if cur:
        records.append(cur)
    return records


def load_crowd_decisions(crowd_dir: Path | str | None = None) -> pd.DataFrame:
    """One row per crowd-annotated item: ``item_id`` + crowd ``decision`` (1/0).

    The bucket file is the crowd's decision. On the handful of items returned in
    *both* files, include wins (matches the coder ``_decision`` convention).
    """
    d = Path(crowd_dir) if crowd_dir is not None else DEFAULT_CROWD_DIR
    rows = [{"item_id": r["MISC3"], "decision": 1} for r in _parse_ris(d / _INCLUDE_FILE)]
    rows += [{"item_id": r["MISC3"], "decision": 0} for r in _parse_ris(d / _EXCLUDE_FILE)]
    df = pd.DataFrame(rows)
    return (
        df.sort_values("decision", ascending=False)
        .drop_duplicates("item_id", keep="first")
        .reset_index(drop=True)
    )


def _score(dec: pd.DataFrame, label: str, *, seed: int) -> dict:
    """Point precision/recall/F1/F2 + 95% HDIs for one scored subset.

    Uses the same Dirichlet-multinomial confusion-matrix posterior as the ML/LLM
    head-to-head (``posterior_metric_summaries``): two calls (beta=1, beta=2)
    cover F1 and F2, Precision/Recall are shared.
    """
    from ic1.classify.inout.utils.metrics import posterior_metric_summaries

    yt = dec.gold.to_numpy().astype(int)
    yp = dec.decision.to_numpy().astype(int)
    row = {
        "label": label,
        "n": int(len(yt)),
        "n_pos": int(yt.sum()),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "f2": fbeta_score(yt, yp, beta=2, zero_division=0),
    }
    s1 = posterior_metric_summaries(yt, yp, beta=1, seed=seed)
    s2 = posterior_metric_summaries(yt, yp, beta=2, seed=seed)
    for key, m in (("precision", s1["Precision"]), ("recall", s1["Recall"]),
                   ("f1", s1["Fbeta"]), ("f2", s2["Fbeta"])):
        row[f"{key}_lo"], row[f"{key}_hi"] = m["hdi_lo"], m["hdi_hi"]
    return row


def load_inout_crowd(
    crowd_dir: Path | str | None = None,
    resolved_csv: Path | str | None = None,
    splits_path: Path | str | None = None,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Crowd precision/recall/F1/F2 (point + 95% HDI) against the adjudicated gold.

    Two rows: ``all crowd items`` (the crowd's delivered performance over every
    document it screened) and ``test subset`` (restricted to the held-out test
    split, for a like-for-like read against the automated systems). Raises
    ``FileNotFoundError`` when the private crowd export is absent, so the build
    can skip it on a clean checkout.
    """
    dec = load_crowd_decisions(crowd_dir)

    res = pd.read_csv(Path(resolved_csv) if resolved_csv is not None else DEFAULT_RESOLVED_CSV)
    gold = res[res.username == GOLD_USER].set_index("item_id")["incl|1"]
    dec["gold"] = dec.item_id.map(gold)
    dec = dec.dropna(subset=["gold"]).reset_index(drop=True)

    splits = EvaluationSplits.load(
        Path(splits_path) if splits_path is not None else DEFAULT_SPLITS_PATH
    )
    dec["in_test"] = dec.item_id.isin(set(splits.test))

    rows = [_score(dec, "all crowd items", seed=seed)]
    test = dec[dec.in_test]
    if len(test):
        rows.append(_score(test, "test subset", seed=seed))
    return pd.DataFrame(rows)
