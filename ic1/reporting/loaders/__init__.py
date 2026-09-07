"""Data loaders: walk experiment output folders into tidy DataFrames.

One module per domain (taxonomy, in/out) because their on-disk layouts differ,
but they emit the same run-level schema so plot functions are shared.
"""

from ic1.reporting.loaders.annotation import (
    inout_agreement_by_set,
    inout_coder_counts,
    inout_coder_f1,
    inout_coderset_composition,
    inout_overall_agreement,
    inout_pairwise_kappa,
    inout_screening_raster,
    inout_unanimous_dispersion,
    load_inout_annotations,
)
from ic1.reporting.loaders.inout import (
    inout_cycle_spans,
    load_inout_comparison,
    load_inout_model_costs,
    load_inout_prompt_churn,
    load_inout_prompts,
    load_inout_runs,
    load_inout_test_metrics,
)
from ic1.reporting.loaders.taxonomy import (
    load_taxonomy_concept_tree,
    load_taxonomy_level_scores,
    load_taxonomy_prompt_churn,
    load_taxonomy_runs,
    load_taxonomy_scheme_scores,
)

__all__ = [
    "load_taxonomy_runs",
    "load_taxonomy_prompt_churn",
    "load_taxonomy_scheme_scores",
    "load_taxonomy_level_scores",
    "load_taxonomy_concept_tree",
    "load_inout_runs",
    "load_inout_prompt_churn",
    "load_inout_model_costs",
    "load_inout_test_metrics",
    "load_inout_comparison",
    "load_inout_prompts",
    "inout_cycle_spans",
    "load_inout_annotations",
    "inout_coder_counts",
    "inout_coderset_composition",
    "inout_overall_agreement",
    "inout_agreement_by_set",
    "inout_pairwise_kappa",
    "inout_coder_f1",
    "inout_screening_raster",
    "inout_unanimous_dispersion",
]
