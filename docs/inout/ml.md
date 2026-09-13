---
title: ML classification
---

We assess a variety of different types of machine learning models, from simpler sklearn models to fine-tuned BERT-type language models. 
For each candidate model, we specify a set of parameters to search, and optimise the model by iteratively training on the training split (see {ref}`inout-splits`), and evaluating on the validation split, updating the parameters algorithmically to find the set that achieves the best performance (measured by the threshold-agnostic average precision metric).

We then select the model-parameter-threshold combination that performs the best across all combinations of models, parameters and thresholds for two objectives:
- F2, to select the model for a ML-only pipeline
- Highest precision where recall > 95, to select the model for the ML-filter -> LLM pipeline.

```{figure} ../figures/inout/data_splits.svg
:name: inout-splits
:width: 100%

Evaluation splits for training, validating, and testing ML and LLM pipelines.
```

For each (either alone or combination with LLMs) we assess performance on the same 650-document held-out test set used to evaluate the LLMs on their own.