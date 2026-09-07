---
title: Prompt Iteration Methods
---

This chapter describes the approach to prompt engineering in Impact Case 1.

## Prompt engineering as a measured activity

We treat prompt design as empirical work. Every candidate prompt
is run through the [deet](https://destiny-evidence.github.io/data-extraction-evaluation-toolkit/development/) extraction pipeline against a human-labelled gold
standard, scored, and compared to its predecessor. The result is a *sequence* of
experiments, each a small, named change — a scope clarification, a definition
edit, a structural tweak — whose effect on the metric is measured. 
The timeline figures in the task chapters make that sequence legible by displaying
performance on top, and the size and target of each prompt edit below.

## Evaluation discipline: dev / validation / test

Records are partitioned into **development**, **validation**, and **test** sets.
Prompts are iterated only against development. 
Validation is a preliminary out-of sample check that guards against over-fitting prompts;
After a validation run, we decide whether to move to a final test or keep iterating.
If we keep iterating, the records from validation are folded back into development, 
and iteration resumes on the enlarged set. 
The **test** set is touched once, at the end, as the truly independent and statistically 
valid measurement of performance. No data used to make any decisions about the pipeline
is re-used for the final evaluation, and there is no risk of multiple testing. 

## Self-consistency voting

For borderline records — where a single sampled decision is unstable — we use
**self-consistency**: the model is queried several times and the majority verdict
is taken, with early stopping once a majority is reached. This stabilises the
run-to-run variance that otherwise makes small prompt effects hard to read, and
it means a difference that survives voting on both sides of a comparison is a
reliable signal rather than sampling noise.

## Model selection

The pipeline is model-agnostic, and we compare models explicitly. 
Selection weighs accuracy against cost and against
qualitative behaviour (does a model preserve the intended operating-point
spread, is it deterministic, is it affordable at corpus scale). Where the figures
distinguish models, they do so by marker shape.

## The gold standard, and its ceiling

The gold labels are a single human reference standard and are internally
imperfect: given the complexity of the domain, guidance is not applied perfectly consistently, 
and near-identical "twin" documents are sometimes labelled oppositely. 
This sets a realistic ceiling that no prompt can beat.
