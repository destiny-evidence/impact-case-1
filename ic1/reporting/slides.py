"""Fill the label-noise slides with numbers from the re-screen analysis.

The slide source ``docs/slides/label_noise.src.md`` holds ``{{ name }}`` placeholders; the rendered deck is written
to ``docs/slides/label_noise.md``. MyST does not substitute placeholders in Typst exports, so they are filled here.
"""

from __future__ import annotations

import re
from pathlib import Path

from ic1.reporting.loaders.label_noise import LabelNoise
from ic1.reporting.tables import HIGH_RECALL_SYSTEM, arm_agreement

_REPO_ROOT = Path(__file__).resolve().parents[2]
SLIDES_DIR = _REPO_ROOT / 'docs' / 'slides'
SOURCE = SLIDES_DIR / 'label_noise.src.md'
RENDERED = SLIDES_DIR / 'label_noise.md'
_HEADER = '<!-- generated from label_noise.src.md by `python -m ic1.reporting.build_figures` — do not edit -->\n'
_PLACEHOLDER = re.compile(r'\{\{\s*(\w+)\s*\}\}')


def label_noise_slide_values(noise: LabelNoise) -> dict[str, str]:
    """Numbers quoted in the slide text, formatted as they appear."""
    o = noise.overturns.set_index('cell')
    total = o.loc['total']
    comp = noise.composition.set_index('system').loc[HIGH_RECALL_SYSTEM]
    wide = noise.metrics.set_index(['system', 'reference'])
    arms = noise.arms.set_index('arm')
    unestimated = noise.unestimated.set_index('cell')
    shares = sorted([comp.FP_noise_share, comp.FN_noise_share])
    resolved_disagree = o.loc['resolved-disagree']
    return {
        'n_disagreements': str(int(comp.FP + comp.FN)),
        'hr_precision': f'{wide.loc[(HIGH_RECALL_SYSTEM, "original"), "precision"]:.2f}',
        'hr_recall': f'{wide.loc[(HIGH_RECALL_SYSTEM, "original"), "recall"]:.2f}',
        'n_decided': str(int(total.answered)),
        'n_no_decision': str(noise.n_exported - noise.n_answered),
        'unanimous_agree_overturned': str(int(o.loc['unanimous-agree', 'include→exclude'] + o.loc['unanimous-agree', 'exclude→include'])),
        'unanimous_agree_upper': f'{o.loc["unanimous-agree", "est_hi"]:.0f}',
        'unanimous_agree_size': str(int(o.loc['unanimous-agree', 'cell_size'])),
        'resolved_disagree_overturned': str(int(resolved_disagree['include→exclude'] + resolved_disagree['exclude→include'])),
        'resolved_disagree_answered': str(int(resolved_disagree.answered)),
        'retest_n': str(arm_agreement(arms.loc['self'].drop(['answered', 'weighted_agreement', 'weighted_kappa']))[1]),
        'wrong_share': f'{total.overturn_rate * 100:.0f}%',
        'noise_share_lo': f'{shares[0] * 100:.0f}%',
        'noise_share_hi': f'{shares[1] * 100:.0f}%',
        'unestimated_size': str(int(unestimated.loc['unanimous-agree', 'size'])),
    }


def render_label_noise_slides(noise: LabelNoise, source: Path = SOURCE, rendered: Path = RENDERED) -> None:
    """Write the deck with every ``{{ name }}`` in the source replaced; an unknown name raises."""
    values = label_noise_slide_values(noise)

    def fill(match: re.Match[str]) -> str:
        return values[match.group(1)]

    text = _PLACEHOLDER.sub(fill, source.read_text())
    _, frontmatter, body = text.split('---\n', 2)
    rendered.write_text(f'---\n{frontmatter}---\n{_HEADER}{body}')
    print(f'  wrote {rendered.relative_to(_REPO_ROOT)}')
