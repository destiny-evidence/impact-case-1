"""Analyse the blind re-screen of the label-noise experiment (#28).

The disagreement cells were routed in full, so their label changes are counted directly. In the sampled control
cells, records are grouped by original label (and, when scoring a system, by its prediction). Each group's size is
known for the whole test set, and its overturn rate gets a Jeffreys Beta(1/2, 1/2) posterior from the re-screened
records in the group. A group without re-screened records takes the rate of its cell and original label. Point
estimates use the observed rates, and intervals are 95% highest-density intervals over posterior draws. The blind label is one
resolver's decision, so it is a cleaner reference and not a true label.
"""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from ic1.annotation.reannotation.export_rescreen import OUT_CSV as RESCREEN_CSV
from ic1.annotation.reannotation.routing import GOLD_CSV, OUT_CSV as ROUTING_CSV, SEED
from ic1.reporting.loaders.inout import _ML_MODEL_DIR, MODE_LABELS

CELLS = ['unanimous-disagree', 'unanimous-agree', 'resolved-disagree', 'resolved-agree']
DISAGREEMENT_CELLS = ['unanimous-disagree', 'resolved-disagree']
N_DRAWS = 4000

# Change in (TP, FP, FN) when a record with this (original label, prediction) has its label overturned.
CHANGE = {(1, 1): (-1, 1, 0), (1, 0): (0, 0, -1), (0, 1): (1, -1, 0), (0, 0): (0, 0, 1)}


def load_cell_sizes(routing_csv: Path = ROUTING_CSV, item_ids: pd.Index | None = None) -> pd.Series:
    """Number of labelled test records in each cell, counting only item_ids if given."""
    routing = pd.read_csv(routing_csv, dtype={'item_id': str})
    if item_ids is not None:
        routing = routing[routing.item_id.isin(item_ids)]
    return routing.cell.value_counts().reindex(CELLS)


def load_rescreen(rescreen_csv: Path = RESCREEN_CSV, cell_sizes: pd.Series | None = None) -> pd.DataFrame:
    """Answered re-screen records with the original label, the blind label and the design weight.

    Records in cells routed in full have weight 1. Elsewhere the weight is cell size / records answered.
    """
    cell_sizes = load_cell_sizes() if cell_sizes is None else cell_sizes
    rs = pd.read_csv(rescreen_csv, dtype={'item_id': str})
    rs = rs[rs.blind_label.notna()].copy()
    rs['label'] = rs.label.astype(int)
    rs['blind_label'] = rs.blind_label.astype(int)
    rs['overturned'] = rs.label != rs.blind_label
    rs['census'] = rs.sampling_fraction == 1
    rs['weight'] = rs.cell.map(cell_sizes / rs.cell.value_counts()).where(~rs.census, 1.0)
    return rs


def load_system_predictions(gold_csv: Path = GOLD_CSV, ml_model_dir: Path = _ML_MODEL_DIR) -> pd.DataFrame:
    """One 0/1 column per system on the test documents that have both LLM and ML predictions, indexed by
    item_id, built as in load_inout_comparison.
    """
    gold = pd.read_csv(gold_csv, dtype={'external_id': str})
    gold = gold[gold.attribute_label.isin(MODE_LABELS)]
    llm = gold.pivot_table(index='external_id', columns='attribute_label', values='llm_extraction', aggfunc='first')
    llm = (llm == True).astype(int).rename(columns=MODE_LABELS)  # noqa: E712 (values are True/False/NaN)

    def thresholded(model: str) -> pd.Series:
        preds = pd.read_csv(ml_model_dir / model / 'test_predictions.csv', dtype={'item_id': str}).set_index('item_id').y_prob
        threshold = json.loads((ml_model_dir / model / 'train_info.json').read_text())['threshold']
        return (preds > threshold).astype(int)

    ml = pd.DataFrame({'ml_only': thresholded('ml_only'), 'filtering': thresholded('filtering')})
    llm = llm.loc[llm.index.intersection(ml.index)]
    ml = ml.loc[llm.index]

    out = pd.DataFrame(index=llm.index)
    for mode in MODE_LABELS.values():
        out[f'LLM only ({mode})'] = llm[mode]
    out['ML only'] = ml.ml_only
    for mode in MODE_LABELS.values():
        out[f'ML → LLM ({mode})'] = ml.filtering & llm[mode]
    return out


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, method='wilson')
    return float(lo), float(hi)


def _hdi(values: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    """Shortest interval containing mass of the draws."""
    ordered = np.sort(values[~np.isnan(values)])
    width = int(np.ceil(mass * len(ordered)))
    start = int(np.argmin(ordered[width - 1 :] - ordered[: len(ordered) - width + 1]))
    return float(ordered[start]), float(ordered[start + width - 1])


def expected_overturns(population: pd.DataFrame, rs: pd.DataFrame, group: str | None = None, seed: int = SEED) -> pd.DataFrame:
    """Expected label overturns per (cell, original label, group) stratum of the population, with posterior draws.

    population has one row per scored test record with cell, label and the group column. Columns of the result are
    cell, label, group, size, overturns (from the observed rate), draws (array of N_DRAWS) and estimated (False
    where neither the stratum nor its cell and label had a re-screened record, so overturns are taken as zero).
    """
    rng = np.random.default_rng(seed)
    keys = ['cell', 'label'] if group is None else ['cell', 'label', group]
    sampled = rs.join(population[[group]], on='item_id') if group is not None and group not in rs else rs
    rows = []
    for group_key, stratum in population.groupby(keys):
        key = cast(tuple, group_key)
        cell, label = key[0], key[1]
        size = len(stratum)
        in_stratum = sampled
        for column, value in zip(keys, key, strict=True):
            in_stratum = in_stratum[in_stratum[column] == value]
        in_cell_label = sampled[(sampled.cell == cell) & (sampled.label == label)]
        census = bool(in_cell_label.census.all()) if len(in_cell_label) else False
        if census:
            k = int(in_stratum.overturned.sum())
            point, draws, estimated = float(k), np.full(N_DRAWS, float(k)), True
        else:
            basis = in_stratum if len(in_stratum) else in_cell_label
            n, k = len(basis), int(basis.overturned.sum())
            estimated = n > 0
            point = size * k / n if n else 0.0
            draws = size * rng.beta(0.5 + k, 0.5 + n - k, N_DRAWS) if n else np.zeros(N_DRAWS)
        rows.append(
            {'cell': cell, 'label': label, 'group': key[2] if group else None, 'size': size, 'overturns': point, 'draws': draws, 'estimated': estimated}
        )
    return pd.DataFrame(rows)


def overturn_table(rs: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Per cell, how often the blind label differs from the original, and the estimated label errors in the cell.

    Rates carry Wilson intervals over the re-screened records. In cells routed in full, unanswered records count as
    unchanged and error counts are exact.
    """
    strata = expected_overturns(population, rs)
    rows = []
    for cell in CELLS:
        sub, cell_strata = rs[rs.cell == cell], strata[strata.cell == cell]
        n, k = len(sub), int(sub.overturned.sum())
        lo, hi = _wilson(k, n)
        est_lo, est_hi = _hdi(np.sum(list(cell_strata.draws), axis=0))
        rows.append(
            {
                'cell': cell,
                'cell_size': int(cell_strata['size'].sum()),
                'answered': n,
                'include→exclude': int((sub.overturned & (sub.label == 1)).sum()),
                'exclude→include': int((sub.overturned & (sub.label == 0)).sum()),
                'overturn_rate': k / n if n else np.nan,
                'rate_lo': lo,
                'rate_hi': hi,
                'est_label_errors': float(cell_strata.overturns.sum()),
                'est_lo': est_lo,
                'est_hi': est_hi,
            },
        )
    table = pd.DataFrame(rows)
    est_lo, est_hi = _hdi(np.sum(list(strata.draws), axis=0))
    total = {
        'cell': 'total',
        'cell_size': table.cell_size.sum(),
        'answered': table.answered.sum(),
        'include→exclude': table['include→exclude'].sum(),
        'exclude→include': table['exclude→include'].sum(),
        'est_label_errors': table.est_label_errors.sum(),
        'est_lo': est_lo,
        'est_hi': est_hi,
    }
    total |= {'overturn_rate': total['est_label_errors'] / total['cell_size'], 'rate_lo': est_lo / total['cell_size'], 'rate_hi': est_hi / total['cell_size']}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def unestimated(population: pd.DataFrame, rs: pd.DataFrame) -> pd.DataFrame:
    """Cell and original label strata whose label errors could not be estimated, with their sizes."""
    strata = expected_overturns(population, rs)
    return strata.loc[~strata.estimated, ['cell', 'label', 'size']]


def _weighted_kappa(original: pd.Series, blind: pd.Series, weight: pd.Series) -> float:
    total = weight.sum()
    observed = (weight * (original == blind)).sum() / total
    p_original, p_blind = (weight * original).sum() / total, (weight * blind).sum() / total
    expected = p_original * p_blind + (1 - p_original) * (1 - p_blind)
    return float((observed - expected) / (1 - expected)) if expected < 1 else np.nan


def consistency_table(rs: pd.DataFrame, by: str) -> pd.DataFrame:
    """Agreement between blind and original labels per group, per cell and weighted over cells.

    Weighted agreement and kappa use the design weights, so they describe the group's records in the test set.
    """
    rows = []
    for group, sub in rs.groupby(by):
        row: dict[str, object] = {by: group, 'answered': len(sub)}
        for cell in CELLS:
            in_cell = sub[sub.cell == cell]
            row[cell] = f'{int((~in_cell.overturned).sum())}/{len(in_cell)}' if len(in_cell) else ''
        row['weighted_agreement'] = float((sub.weight * ~sub.overturned).sum() / sub.weight.sum())
        row['weighted_kappa'] = _weighted_kappa(sub.label, sub.blind_label, sub.weight)
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics(tp, fp, fn) -> dict:
    """Precision, recall, F1 and F2 from (weighted) counts, element-wise over arrays; 0/0 gives NaN."""
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    with np.errstate(invalid='ignore', divide='ignore'):
        return {
            'precision': tp / (tp + fp),
            'recall': tp / (tp + fn),
            'f1': 2 * tp / (2 * tp + fp + fn),
            'f2': 5 * tp / (5 * tp + 4 * fn + fp),
        }


def _confusion(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """TP, FP and FN."""
    return np.array([((truth == 1) & (pred == 1)).sum(), ((truth == 0) & (pred == 1)).sum(), ((truth == 1) & (pred == 0)).sum()], dtype=float)


def corrected_metrics(rs: pd.DataFrame, population: pd.DataFrame, systems: list[str]) -> pd.DataFrame:
    """Each system scored against the original labels, against the original labels with only the
    disagreement cells replaced by the blind label (naive), and against the blind label (corrected).

    The corrected confusion matrix is the exact one against the original labels plus the change from the
    overturns in each (cell, original label, prediction) stratum.
    """
    label = population.label.to_numpy()
    naive_truth = population.label.copy()
    disagreement = rs[rs.cell.isin(DISAGREEMENT_CELLS)].set_index('item_id').blind_label
    naive_truth.loc[disagreement.index] = disagreement

    rows = []
    for system in systems:
        pred = population[system].to_numpy()
        exact = _confusion(label, pred)
        rows.append({'system': system, 'reference': 'original', **_metrics(*exact)})
        rows.append({'system': system, 'reference': 'naive', **_metrics(*_confusion(naive_truth.to_numpy(), pred))})

        strata = expected_overturns(population, rs, group=system)
        change = np.array([CHANGE[(label, pred)] for label, pred in zip(strata.label, strata.group, strict=True)], dtype=float)
        point = exact + change.T @ strata.overturns.to_numpy()
        draws = exact[:, None] + change.T @ np.stack(list(strata.draws))
        row: dict[str, object] = {'system': system, 'reference': 'corrected', **_metrics(*point)}
        for metric, values in _metrics(*draws).items():
            row[f'{metric}_lo'], row[f'{metric}_hi'] = _hdi(values)
        rows.append(row)
    return pd.DataFrame(rows).astype(dict.fromkeys(('precision', 'recall', 'f1', 'f2'), float))


def error_composition(rs: pd.DataFrame, population: pd.DataFrame, systems: list[str]) -> pd.DataFrame:
    """Per system, the false positives and false negatives against the original labels, and the estimated
    number and share of each that the blind label reverses (label noise) as opposed to confirms (model error).
    """
    rows = []
    for system in systems:
        strata = expected_overturns(population, rs, group=system)
        row: dict[str, object] = {'system': system}
        for kind, value in (('FP', 1), ('FN', 0)):
            count = int(((population[system] == value) & (population.label != value)).sum())
            errors = strata[(strata.group == value) & (strata.label != value)]
            noise = float(errors.overturns.sum())
            lo, hi = _hdi(np.sum(list(errors.draws), axis=0)) if len(errors) else (0.0, 0.0)
            row |= {
                kind: count,
                f'{kind}_label_noise': noise,
                f'{kind}_noise_lo': lo,
                f'{kind}_noise_hi': hi,
                f'{kind}_noise_share': noise / count if count else np.nan,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def reason_table(rs: pd.DataFrame) -> pd.DataFrame:
    """Blind reasons given on overturned records, by cell."""
    overturned = rs[rs.overturned]
    return pd.crosstab(overturned.blind_reason.fillna('none'), overturned.cell, margins=True)


def load_population(preds: pd.DataFrame, routing_csv: Path = ROUTING_CSV) -> pd.DataFrame:
    """Labelled test records that every system predicts, with cell, original label and each system's prediction."""
    routing = pd.read_csv(routing_csv, dtype={'item_id': str}).set_index('item_id')
    population = routing[['cell', 'label']].join(preds, how='inner')
    population = population[population.label.notna()]
    return population.astype({'label': int})


def main() -> None:
    preds = load_system_predictions()
    population = load_population(preds)
    cell_sizes = population.cell.value_counts().reindex(CELLS)
    rs = load_rescreen(cell_sizes=cell_sizes)
    systems = list(preds.columns)
    print(f'Scoring the {len(population)} labelled test records that every system predicts.\n')

    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.3f}'.format)
    print('Overturn rates by cell\n', overturn_table(rs, population).to_string(index=False), '\n')
    print('Consistency by arm\n', consistency_table(rs, 'arm').to_string(index=False), '\n')
    print('Consistency by reviewer\n', consistency_table(rs, 'reviewer').to_string(index=False), '\n')
    print('Operating points\n', corrected_metrics(rs, population, systems).to_string(index=False), '\n')
    print('Error composition\n', error_composition(rs, population, systems).to_string(index=False), '\n')
    print('Flags on overturned records\n', reason_table(rs).to_string())
    for cell, label, size in unestimated(population, rs).itertuples(index=False, name=None):
        print(f'\nNo {cell} record with original label {label} was re-screened, so errors among its {size} such records are not estimated.')


if __name__ == '__main__':
    main()
