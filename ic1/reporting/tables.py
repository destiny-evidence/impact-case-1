"""Render metric DataFrames to committed Markdown fragments for the docs site.

Same principle as the figures: generate the artifact locally (where the data
lives) and commit it, so the mystmd build needs no data or Python — it just
``{include}``s the fragment.
"""

from __future__ import annotations

import html
from typing import cast

import matplotlib as mpl
import pandas as pd
from matplotlib.colors import to_hex

# Red (low) -> yellow -> blue (high): ColorBrewer's colour-blind-safe stand-in
# for the red-green scale, so the F1 badge reads as bad -> good without relying
# on the red/green channel.
_F1_CMAP = mpl.colormaps['RdYlBu']


def _score_span(v: float | None, *, prefix: str = 'F1 ') -> str:
    """An F1 value as a coloured pill; background maps 0 (red) -> 1 (blue)."""
    if v is None:
        return '—'
    r, g, b, _ = _F1_CMAP(max(0.0, min(1.0, v)))
    fg = '#000' if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else '#fff'
    return (
        f"<span class='f1-badge' style='background-color:{to_hex((r, g, b))};"
        f"color:{fg};padding:0 .35em;border-radius:.35em;font-weight:600'>"
        f'{prefix}{v:.2f}</span>'
    )


def _check(v: bool) -> str:
    return '✓' if v else '✗'


def _disagreement_table(disagreements: list[dict], max_reasoning: int) -> str:
    """The FN/FP documents for one concept: what the model got wrong and why."""
    rows = ['<tr><th>document</th><th>human</th><th>LLM</th><th>LLM quotation (if provided)</th><th>LLM reasoning</th></tr>']
    for d in disagreements:
        doc = html.escape(d['document'][:70])
        verbatim = html.escape((d.get('verbatim') or '')[:max_reasoning])
        reason = html.escape((d['reasoning'] or '')[:max_reasoning])
        rows.append(f'<tr><td>{doc}</td><td>{_check(d["human"])}</td><td>{_check(d["llm"])}</td><td>{verbatim}</td><td>{reason}</td></tr>')
    return "<table class='drilldown-fails'>\n" + '\n'.join(rows) + '\n</table>'


def _concept_html(node: dict, *, fail_threshold: float, max_reasoning: int) -> str:
    """One concept: summary line (own F1 + confusion) over its children/failures."""
    own = node['own']
    has_children = bool(node['children'])
    label = html.escape(node['label'])
    parts = [f'<b>{label}</b>' if has_children else label]
    warn = False
    if own:
        if own['f1'] is not None:
            parts.append(_score_span(own['f1']))
            warn = own['f1'] < fail_threshold
        else:
            parts.append('no gold')
            warn = own['fp'] > 0
        parts.append(f'TP{own["tp"]} FP{own["fp"]} FN{own["fn"]}')
    if has_children:
        parts.append(f'children {_score_span(node["children_f1"])} (n={node["n_children_scoreable"]})')
    summary = ' · '.join(parts) + (' ⚠' if warn else '')

    inner = []
    if own and own['disagreements']:
        inner.append(_disagreement_table(own['disagreements'], max_reasoning))
    inner += [_concept_html(c, fail_threshold=fail_threshold, max_reasoning=max_reasoning) for c in node['children']]
    if not inner:  # leaf with nothing to expand — plain line, no disclosure widget
        return f"<div class='drilldown-leaf'>{summary}</div>"
    return f'<details><summary>{summary}</summary>\n' + '\n'.join(inner) + '\n</details>'


def taxonomy_drilldown_html(
    tree: list[dict],
    *,
    fail_threshold: float = 0.5,
    max_reasoning: int = 240,
) -> str:
    """Collapsible scheme -> concept -> failing-document tree (raw HTML fragment).

    Consumes ``load_taxonomy_concept_tree``. Each ``<summary>`` carries the
    node's own document-level F1 and TP/FP/FN, plus (for parents) the pooled
    direct-children F1; a ⚠ flags concepts scoring below ``fail_threshold``.
    Expanding a concept reveals the specific human/LLM disagreements and the
    model's reasoning — the failure evidence — then its child concepts.
    """
    blocks = []
    for s in tree:
        summary = f'<b>{html.escape(s["scheme_title"])}</b> · direct-children {_score_span(s["children_f1"])} (n={s["n_children_scoreable"]})'
        body = '\n'.join(_concept_html(c, fail_threshold=fail_threshold, max_reasoning=max_reasoning) for c in s['children'])
        blocks.append(f'<details><summary>{summary}</summary>\n{body}\n</details>')
    return '\n\n'.join(blocks) + '\n'


def metrics_markdown(
    df: pd.DataFrame,
    *,
    label_col: str = 'Operating point',
    decimals: int = 2,
    bold_max: bool = True,
) -> str:
    """GFM table with the maximum in each numeric column bolded.

    ``bold_max`` bolds the best cell per metric column (e.g. Precision in the
    high-precision row, Recall in the high-recall row).
    """
    numeric = [c for c in df.columns if c != label_col]
    maxima = {c: df[c].max() for c in numeric}

    def cell(col: str, val: float) -> str:
        s = f'{val:.{decimals}f}'
        return f'**{s}**' if bold_max and abs(val - maxima[col]) < 1e-9 else s

    lines = [
        '| ' + ' | '.join([label_col, *numeric]) + ' |',
        '|' + '|'.join(['---'] * (len(numeric) + 1)) + '|',
    ]
    for _, r in df.iterrows():
        cells = [str(r[label_col])] + [cell(c, r[c]) for c in numeric]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def comparison_markdown(df: pd.DataFrame, *, decimals: int = 2) -> str:
    """System comparison: point estimate + 95% HDI per metric, plus cost.

    Consumes ``load_inout_comparison``. Metric cells read ``0.72 [0.66–0.78]``;
    the cost column extrapolates the LLM run's per-doc cost to the full corpus
    (the cascade pays it only on the fraction the filter forwards).
    """

    def ci(r: pd.Series, m: str) -> str:
        return f'{r[m]:.{decimals}f} [{r[f"{m}_lo"]:.{decimals}f}–{r[f"{m}_hi"]:.{decimals}f}]'

    def money(v: float) -> str:
        if v >= 1e6:
            return f'${v / 1e6:.1f}M'
        if v >= 1e3:
            return f'${v / 1e3:.0f}k'
        return f'${v:.0f}'

    cols = ['System', 'Precision', 'Recall', 'F1', 'F2', 'Sent to LLM', 'Corpus cost']
    lines = [
        '| ' + ' | '.join(cols) + ' |',
        '|' + '|'.join(['---'] * len(cols)) + '|',
    ]
    for _, r in df.iterrows():
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(r.system),
                    ci(r, 'precision'),
                    ci(r, 'recall'),
                    ci(r, 'f1'),
                    ci(r, 'f2'),
                    f'{r.prop_to_llm * 100:.0f}%',
                    money(r.corpus_cost),
                ]
            )
            + ' |'
        )
    return '\n'.join(lines) + '\n'


def crowd_markdown(crowd: pd.DataFrame, *, decimals: int = 2) -> str:
    """Crowd precision/recall/F1/F2 (point + 95% HDI) per scoring scope.

    Consumes ``load_inout_crowd``. One row per scope (all crowd items / test
    subset); metric cells read ``0.64 [0.56–0.72]``.
    """

    def ci(r: pd.Series, m: str) -> str:
        return f'{r[m]:.{decimals}f} [{r[f"{m}_lo"]:.{decimals}f}–{r[f"{m}_hi"]:.{decimals}f}]'

    cols = ['Scope', 'n', 'Precision', 'Recall', 'F1', 'F2']
    lines = [
        '| ' + ' | '.join(cols) + ' |',
        '|' + '|'.join(['---'] * len(cols)) + '|',
    ]
    for _, r in crowd.iterrows():
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(r.label),
                    f'{int(r.n):,}',
                    ci(r, 'precision'),
                    ci(r, 'recall'),
                    ci(r, 'f1'),
                    ci(r, 'f2'),
                ]
            )
            + ' |'
        )
    return '\n'.join(lines) + '\n'


def annotation_agreement_markdown(by_set: pd.DataFrame, overall: dict, *, decimals: int = 2) -> str:
    """Per coder-set agreement table (Fleiss' κ, % agreement), overall row last.

    Consumes ``inout_agreement_by_set`` and ``inout_overall_agreement``. Columns:
    Coder-set, Size, Documents (complete cases), Fleiss' κ, % agreement.
    """
    cols = ['Coder-set', 'Size', 'Documents', "Fleiss' κ", '% agreement']
    lines = [
        '| ' + ' | '.join(cols) + ' |',
        '|' + '|'.join(['---'] * len(cols)) + '|',
    ]

    def krow(label: str, size: str, n: int, k: float, pct: float) -> str:
        kc = '—' if pd.isna(k) else f'{k:.{decimals}f}'
        return f'| {label} | {size} | {n:,} | {kc} | {pct * 100:.0f}% |'

    for _, r in by_set.sort_values('fleiss', ascending=False).iterrows():
        lines.append(krow(r.label, str(int(r['size'])), int(r.n_complete), r.fleiss, r.pct_agreement))
    lines.append('|' + '|'.join([' '] * len(cols)) + '|')
    lines.append(krow('**Overall** (pooled 3-rater)', '3', overall['n_items'], overall['fleiss'], overall['pct_agreement']))
    return '\n'.join(lines) + '\n'


def annotation_f1_markdown(f1df: pd.DataFrame, *, decimals: int = 2) -> str:
    """Per-coder precision/recall/F1 vs the adjudicated value, summaries last.

    Consumes ``inout_coder_f1``. Coders sorted by F1 descending; the ``average
    coder`` and ``pooled`` summary rows are set off at the bottom.
    """
    cols = ['Coder', 'Documents', 'Precision', 'Recall', 'F1']
    lines = [
        '| ' + ' | '.join(cols) + ' |',
        '|' + '|'.join(['---'] * len(cols)) + '|',
    ]

    def row(label: str, r: pd.Series, *, bold: bool = False) -> str:
        def c(v: float) -> str:
            s = f'{v:.{decimals}f}'
            return f'**{s}**' if bold else s

        name = f'**{label}**' if bold else label
        return f'| {name} | {int(r.n):,} | {c(r.precision)} | {c(r.recall)} | {c(r.f1)} |'

    summ = f1df[f1df.coder.isin(['average coder', 'pooled'])].set_index('coder')
    per = f1df[~f1df.coder.isin(['average coder', 'pooled'])]
    for _, r in per.sort_values('f1', ascending=False).iterrows():
        lines.append(row(r.coder.replace('coder_', 'coder '), r))
    lines.append('|' + '|'.join([' '] * len(cols)) + '|')
    for name in ('average coder', 'pooled'):
        if name in summ.index:
            lines.append(row(name, cast('pd.Series', summ.loc[name]), bold=True))
    return '\n'.join(lines) + '\n'


def prompts_markdown(prompts: dict, *, full: bool = False) -> str:
    """Render the final screening prompts as collapsible dropdowns for the docs.

    Consumes ``load_inout_prompts``. The three scope prompts are near identical,
    so by default it shows the system prompt, the ``headline`` operating point in
    full, and the other two as a **unified diff against the headline** (fenced
    ``diff``, git-style +/- lines). ``full=True`` instead emits every scope
    prompt in its entirety.
    """
    import difflib

    def block(title: str, body: str, lang: str = 'text', *, open_: bool = False) -> str:
        opt = '\n:open:' if open_ else ''
        return f':::{{dropdown}} {title}{opt}\n```{lang}\n{body}\n```\n:::\n'

    headline = prompts['headline']
    modes = prompts['modes']
    out = [block('System prompt', prompts['system'])]

    if full:
        for mode, text in modes.items():
            out.append(block(f'Scope prompt — {mode}', text))
        return '\n'.join(out)

    out.append(block(f'Scope prompt — {headline} (headline)', modes[headline], open_=True))
    for mode, text in modes.items():
        if mode == headline:
            continue
        diff = '\n'.join(
            difflib.unified_diff(
                modes[headline].splitlines(),
                text.splitlines(),
                fromfile=headline,
                tofile=mode,
                lineterm='',
            )
        )
        out.append(block(f'Scope prompt — {mode} (diff vs {headline})', diff, 'diff'))
    return '\n'.join(out)


_CELL_LABELS = {
    'unanimous-disagree': 'Unanimous, LLM disagrees',
    'unanimous-agree': 'Unanimous, LLM agrees',
    'resolved-disagree': 'Resolved, LLM disagrees',
    'resolved-agree': 'Resolved, LLM agrees',
    'total': '**All test records**',
}
_ARM_LABELS = {
    'unanimous': 'Unanimous coders',
    'self': 'Resolved, same resolver (test–retest)',
    'cross': 'Resolved, other resolver',
    'both': 'Resolved by both, re-screened by one',
}


def _md_table(cols: list[str], rows: list[list[str]]) -> str:
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(lines) + '\n'


def label_noise_overturn_markdown(overturns: pd.DataFrame, *, decimals: int = 1) -> str:
    """Overturned labels per design cell and the estimated label errors in the test set.

    Consumes ``load_label_noise()['overturns']``. Rates carry Wilson intervals over the re-screened records;
    estimated errors carry 95% HDIs and are exact in cells re-screened in full.
    """
    rows = []
    for _, r in overturns.iterrows():
        exact = r.est_lo == r.est_hi
        est = f'{r.est_label_errors:.{decimals}f}' + ('' if exact else f' [{r.est_lo:.{decimals}f}–{r.est_hi:.{decimals}f}]')
        rows.append(
            [
                _CELL_LABELS[r.cell],
                f'{int(r.cell_size):,}',
                f'{int(r.answered):,}',
                f'{int(r["include→exclude"])} / {int(r["exclude→include"])}',
                f'{r.overturn_rate * 100:.{decimals}f}% [{r.rate_lo * 100:.{decimals}f}–{r.rate_hi * 100:.{decimals}f}%]',
                est,
            ],
        )
    cols = ['Cell', 'Test records', 'Re-screened', 'Overturned (include→exclude / exclude→include)', 'Overturn rate', 'Estimated label errors']
    return _md_table(cols, rows)


def label_noise_arms_markdown(arms: pd.DataFrame, *, decimals: int = 2) -> str:
    """Agreement of the blind decision with the original label per routing arm.

    Consumes ``load_label_noise()['arms']``. Cell columns read agreed / re-screened; the weighted columns weight
    each record by the number of test records it stands for.
    """
    cells = [c for c in _CELL_LABELS if c != 'total']
    rows = []
    for _, r in arms.iterrows():
        kappa = '—' if pd.isna(r.weighted_kappa) else f'{r.weighted_kappa:.{decimals}f}'
        rows.append(
            [_ARM_LABELS.get(r.arm, r.arm), str(int(r.answered)), *[r[c] or '—' for c in cells], f'{r.weighted_agreement * 100:.1f}%', kappa],
        )
    cols = ['Arm', 'Re-screened', *[_CELL_LABELS[c] for c in cells], 'Weighted agreement', "Weighted Cohen's κ"]
    return _md_table(cols, rows)


def label_noise_metrics_markdown(metrics: pd.DataFrame, *, decimals: int = 2) -> str:
    """Each system against the original and the corrected labels, with the naive F1 alongside.

    Consumes ``load_label_noise()['metrics']``. Metric cells read ``original → corrected [95% HDI]``.
    """
    wide = metrics.set_index(['system', 'reference'])
    rows = []
    for system in dict.fromkeys(metrics.system):
        o, c, n = wide.loc[(system, 'original')], wide.loc[(system, 'corrected')], wide.loc[(system, 'naive')]
        cells = [
            f'{o[m]:.{decimals}f} → **{c[m]:.{decimals}f}** [{c[f"{m}_lo"]:.{decimals}f}–{c[f"{m}_hi"]:.{decimals}f}]'
            for m in ('precision', 'recall', 'f1', 'f2')
        ]
        rows.append([system, *cells, f'{n.f1:.{decimals}f}'])
    cols = ['System', 'Precision', 'Recall', 'F1', 'F2', 'F1, disagreements only (naive)']
    return _md_table(cols, rows)


def label_noise_composition_markdown(composition: pd.DataFrame, *, decimals: int = 1) -> str:
    """Each system's errors against the original labels, and how many of them are label noise.

    Consumes ``load_label_noise()['composition']``. Label-noise cells read ``estimate [95% HDI] (share)``.
    """

    def noise(r: pd.Series, kind: str) -> str:
        est, lo, hi, share = r[f'{kind}_label_noise'], r[f'{kind}_noise_lo'], r[f'{kind}_noise_hi'], r[f'{kind}_noise_share']
        interval = '' if lo == hi else f' [{lo:.{decimals}f}–{hi:.{decimals}f}]'
        return f'{est:.{decimals}f}{interval} ({share * 100:.0f}%)'

    rows = [[str(r.system), str(int(r.FP)), noise(r, 'FP'), str(int(r.FN)), noise(r, 'FN')] for _, r in composition.iterrows()]
    cols = ['System', 'False positives', 'of which label noise', 'False negatives', 'of which label noise']
    return _md_table(cols, rows)


# Compact tables for the label-noise slides, which show the high-recall LLM and two-by-two design cells.
HIGH_RECALL_SYSTEM = 'LLM only (high recall)'


def _cell_grid(values: dict[str, str]) -> str:
    """A two-by-two grid with label type as rows and LLM agreement as columns, from a cell → text mapping."""
    rows = [[f'**{kind.capitalize()}**', values[f'{kind}-disagree'], values[f'{kind}-agree']] for kind in ('unanimous', 'resolved')]
    return _md_table(['', 'LLM disagrees', 'LLM agrees'], rows)


def label_noise_slide_errors_markdown(composition: pd.DataFrame, system: str = HIGH_RECALL_SYSTEM) -> str:
    """False positive and false negative counts of one system against the original labels."""
    r = composition.set_index('system').loc[system]
    rows = [
        ['Included by the LLM, excluded by the humans (false positives)', str(int(r.FP))],
        ['Excluded by the LLM, included by the humans (false negatives)', str(int(r.FN))],
    ]
    return _md_table(['', 'Count'], rows)


def label_noise_slide_cell_sizes_markdown(overturns: pd.DataFrame) -> str:
    """Test records per design cell."""
    sizes = overturns.set_index('cell').cell_size
    return _cell_grid({c: f'{int(sizes[c])} records' for c in sizes.index if c != 'total'})


def label_noise_slide_rescreened_markdown(overturns: pd.DataFrame) -> str:
    """Records that received a blind decision out of the test records per design cell."""
    o = overturns.set_index('cell')
    return _cell_grid({c: f'{int(o.answered[c])} of {int(o.cell_size[c])}' for c in o.index if c != 'total'})


def label_noise_slide_overturned_markdown(overturns: pd.DataFrame, kind: str) -> str:
    """Overturned labels out of the re-screened records for one label type (``unanimous`` or ``resolved``)."""
    o = overturns.set_index('cell')
    cells = []
    for agreement in ('disagree', 'agree'):
        r = o.loc[f'{kind}-{agreement}']
        cells.append(f'{int(r["include→exclude"] + r["exclude→include"])} of {int(r.answered)} overturned')
    return _md_table(['LLM disagrees', 'LLM agrees'], [cells])


def arm_agreement(arm: pd.Series) -> tuple[int, int]:
    """Summed agreed / re-screened over an arm's ``k/n`` cells."""
    pairs = [str(v).split('/') for v in arm if '/' in str(v)]
    return sum(int(k) for k, _ in pairs), sum(int(n) for _, n in pairs)


def label_noise_slide_routing_markdown(arms: pd.DataFrame, *, decimals: int = 2) -> str:
    """Agreement of the blind decision with the earlier resolution, for the same and for the other resolver."""
    a = arms.set_index('arm')
    rows = []
    for arm, label in (('self', 'The same resolver (test–retest)'), ('cross', 'The other resolver')):
        k, n = arm_agreement(a.loc[arm].drop(['answered', 'weighted_agreement', 'weighted_kappa']))
        rows.append([label, f'{k} of {n}', f'{a.weighted_kappa[arm]:.{decimals}f}'])
    return _md_table(['Routed to', 'Agreed', "Weighted Cohen's κ"], rows)


def label_noise_slide_total_markdown(overturns: pd.DataFrame) -> str:
    """Estimated label errors in the whole test set."""
    t = overturns.set_index('cell').loc['total']
    rows = [
        [f'Wrong labels among {int(t.cell_size)} test records', f'about {t.est_label_errors:.0f} (95% interval {t.est_lo:.0f}–{t.est_hi:.0f})'],
        ['Share of the test set', f'{t.overturn_rate * 100:.1f}%'],
    ]
    return _md_table(['', 'Estimate'], rows)


def label_noise_slide_scores_markdown(metrics: pd.DataFrame, system: str = HIGH_RECALL_SYSTEM, *, decimals: int = 2) -> str:
    """One system's scores against the original labels, with only the disagreements re-screened, and corrected."""
    wide = metrics.set_index(['system', 'reference'])
    refs = ('original', 'naive', 'corrected')
    rows = [
        [label, *[f'{wide.loc[(system, ref), m]:.{decimals}f}' for ref in refs]]
        for label, m in (('Precision', 'precision'), ('Recall', 'recall'), ('F1', 'f1'), ('F2', 'f2'))
    ]
    return _md_table(['', 'Original labels', 'Only disagreements re-screened', 'Corrected labels'], rows)


def label_noise_slide_composition_markdown(composition: pd.DataFrame, system: str = HIGH_RECALL_SYSTEM) -> str:
    """One system's false positives and negatives and how many of them are label noise."""
    r = composition.set_index('system').loc[system]
    rows = []
    for kind, label in (('FP', 'False positives'), ('FN', 'False negatives')):
        lo, hi = r[f'{kind}_noise_lo'], r[f'{kind}_noise_hi']
        interval = '' if lo == hi else f' [{lo:.0f}–{hi:.0f}]'
        rows.append([label, str(int(r[kind])), f'{r[f"{kind}_label_noise"]:.0f}{interval} ({r[f"{kind}_noise_share"] * 100:.0f}%)'])
    return _md_table(['', 'Count', 'Label noise'], rows)
