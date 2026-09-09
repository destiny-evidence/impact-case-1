---
title: Human annotation
---

5,000 documents were coded by 3 independent coders. Where decisions were not unanimous, an additional senior independent coder resolved decisions.

```{figure} ../figures/inout/annotation_raster.svg
:name: annotation-raster
:width: 100%

Human annotations, sorted by coder set, and within coder sets by decision.
```

There was substantial variation in the rate at which individual coders included documents [2%-51%].

## Inter-coder agreement statistics

Overall agreement was fair, with large variation between coder sets

```{figure} ../figures/inout/annotation_agreement.svg
:name: coder-agreement
:width: 100%

Agreement between coder sets
```

## Variation in human coder agreement with resolved decisions

Wherever human coders did not reach a unanimous decision,
a senior coder resolved decisions by selecting either INCLUDE or EXCLUDE.
We can apply classification metrics to coders to measure their agreement
with the final resolved decisions, noting that in cases were all coders
were unanimously wrong, errors would not be found through resolution.

```{figure} ../figures/inout/annotation_coder_pr.svg
:name: inout-coder-pr
:width: 100%

Annotator precision and recall.
```

{ref}`inout-coder-pr` shows how different coders approached the trade-off between
precision and recall. Even high-performing coders, whose F2 beat 80% did so with substantially different levels of precision and recall. These coders are also the senior coders whose job was to resolve inconsistencies

## Implications for evaluating automated pipelines

The substantial variation in coder performance shows that the task of screening
documents for eligibility in a climate and health repository is inherently difficult.

Moreover, there are good reasons to suspect that the final resolved dataset contains
inconsistencies: firstly, because the task is genuinely hard, meaning that even 
senior coders are not perfectly consistent (with themselves and with each other), and secondly
because decisions were only resolved when there was disagreement among coders. This means that where documents were coded by a coder set who all erred on the side of precision, the likelihood of erroneously excluded documents that were excluded unanimously and never resolved is high.