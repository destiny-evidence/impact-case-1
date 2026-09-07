---
title: Human annotation
---

5,000 documents were coded by 3 independent coders. Where decisions were not unanimous, an additional senior independent coder resolved decisions.

```{figure} ../figures/inout/annotation_raster.svg
:name: annotation-raster
:width: 100%

Human annotations
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
We can then apply classification metrics to coders to measure their agreement
with the final resolved decisions, noting that in cases were all coders
were unanimously wrong, errors would not be found through resolution.

```{figure} ../figures/inout/annotation_coder_pr.svg
:name: inout-coder-pr
:width: 100%

Annotator precision and recall.
```

{ref}`inout-coder-pr` shows how different coders approached the trade-off between
precision and recall.