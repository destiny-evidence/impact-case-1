"""Data loaders: walk experiment output folders into tidy DataFrames.

One module per domain (taxonomy, in/out) because their on-disk layouts differ,
but they emit the same run-level schema so plot functions are shared.
"""

from ic1.reporting.loaders.inout import (
    inout_cycle_spans,
    load_inout_model_costs,
    load_inout_prompt_churn,
    load_inout_runs,
    load_inout_test_metrics,
)
from ic1.reporting.loaders.taxonomy import (
    load_taxonomy_prompt_churn,
    load_taxonomy_runs,
)

__all__ = [
    "load_taxonomy_runs",
    "load_taxonomy_prompt_churn",
    "load_inout_runs",
    "load_inout_prompt_churn",
    "load_inout_model_costs",
    "load_inout_test_metrics",
    "inout_cycle_spans",
]
