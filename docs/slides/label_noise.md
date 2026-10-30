---
title: Investigating human label noise without biasing our assessment of classifiers
authors:
  - name: Max Callaghan
exports:
  - format: typst
    template: ./template
    output: ../exports/label_noise_slides.pdf
---
<!-- generated from label_noise.src.md by `python -m ic1.reporting.build_figures` — do not edit -->

# 1. Classifying relevance to climate and health

## A difficult classification task

- 19 coders annotated 5,000 documents in triplicate
- Where annotators were not unanimous, senior coders resolved
- The system boundaries or climate-health relevance are *complicated*

## Building an LLM classifier

- We used 1,000 of these documents to build an LLM classifier
  - (the rest were used for ML training and validation)

- We iterated prompts on ~350, with ~650 used as a final held-out test set
- 83/650 were relevant

## Classifier performance

```{figure} ../figures/inout/comparison.svg
:width: 100%
```

## High Recall LLM

Because prompt iteration was tricky, and to cater to different users of the repository,
we customised prompts to optimise higher precision or higher recall.

Our high-recall prompt achieved a precision of 0.63 and a recall of 0.86.

```{include} ../tables/slides_label_noise_errors.md
```

## Are we bad at LLM? Or did we achieve superhuman performance?

Each of these 54 disagreements counts as a classifier error.

That is only correct when the human label is right.

If some human labels are wrong, part of the measured error belongs to the labels.

How can we figure out how much error belongs to the humans and how much belongs to the models?

# 2. Where label noise can come from

## Source 1: resolvers differ from each other

Resolved records are contested by definition.
A different resolver might have decided them the other way.

This is inconsistency *between* resolvers. 

We know that each resolver achieved a different point on the precision-recall tradeoff

## Source 2: a resolver differs from themselves

The same resolver, looking at the same record on another day, might decide differently.

This may be because inclusion criteria are ambiguous, or because documents themselves are ambiguous.
Or because the task is simply hard

This is inconsistency *within* a resolver.

## Source 3: unanimous, but still wrong

Three coders can make the same mistake.
We treat unanimous labels as reliable, so this kind of error would go unnoticed.

This is *unanimous wrongness*.

# 3. The experiment

## Re-annotation

Two senior coders, AM and MIM, screened a sample of test records again.
Their new decision is a second opinion on each original label.
Where the second opinion differs from the original, we say the original label was *overturned*.

## Which records we re-screened

Every test record falls into one of four cells.
The columns record whether the high-recall LLM agrees with the original label.

```{include} ../tables/slides_label_noise_cells.md
```

## Why not only the disagreements?

Re-screening only the records where the LLM disagrees seems efficient, 
but it can only move labels towards the model.
It doesn't catch cases where both the original label and the model were wrong.
Scores then rise, whatever the true noise rate.

So we re-screened

- all records where the LLM disagrees, and
- a random sample of the records where it agrees, as controls.

## How many records we re-screened

```{include} ../tables/slides_label_noise_rescreened.md
```

131 records received a new decision.
3 more were re-screened without a decision.

## Blind rescreening

Each senior coder saw only the title and abstract, in shuffled order.
They did not see

- the coders' votes or the earlier resolution,
- any model or crowd prediction,
- which cell the record belongs to.


# 4. Results

## Step 1: unanimous records hold up

```{include} ../tables/slides_label_noise_overturned_unanimous.md
```

Unanimous labels are rarely wrong.
The sample where the LLM agrees contains 0 overturned labels, which is still compatible with up to about 31 errors among its 488 records.

## Step 2: resolved records are less stable

```{include} ../tables/slides_label_noise_overturned_resolved.md
```

Contested labels carry most of the noise.
Where the LLM disagrees with a resolution, the second opinion sides with the LLM in 10 of 26 records.

## Step 3: between and within resolvers

On resolved records, the second opinion agreed with the earlier resolution as follows.

```{include} ../tables/slides_label_noise_routing.md
```

Resolvers agree with themselves more often than with each other.
The test–retest group has only 22 records, so this comparison is uncertain.

## Step 4: how many labels are wrong

Combining the four cells, with the control samples scaled up to the full cell sizes:

```{include} ../tables/slides_label_noise_total.md
```

The wide upper end comes from the unanimous cell with no overturns, which was only sampled.

## Step 5: effect on the classifier scores

```{figure} ../figures/inout/label_noise.png
:width: 100%
```

## Step 5: the high-recall LLM in numbers

```{include} ../tables/slides_label_noise_scores.md
```

Correcting the labels raises the scores a little.
Re-screening only the disagreements would have suggested a much larger gain, because it can only move labels towards the model.

## Step 6: how much of the error is label noise

For the high-recall LLM, among the disagreements with the human labels:

```{include} ../tables/slides_label_noise_composition.md
```

Most of the remaining error is model error.
The ranking of the systems is unchanged.

## Take-aways

- About 3% of test labels are wrong, concentrated in records the coders disagreed on.
- Unanimous labels are fairly reliable.
- Label noise explains between 19% and 33% of the high-recall LLM's errors.
- The measured performance understates the classifiers slightly, and the systems rank the same.

## Limitations

- The second opinion is one senior coder's decision, so it remains fallible.
- No unanimous include where the LLM agrees was sampled, so errors among those 17 records are not estimated.
- The test–retest group is small.
