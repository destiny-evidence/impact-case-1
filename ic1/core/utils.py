import hashlib
from datetime import datetime
from enum import Enum
from json import JSONEncoder
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel


def uniform(task: str, item_id: str) -> float:
    """Stable, language-independent uniform value in [0, 1) for (task, item_id)."""
    h = hashlib.sha256(f'{task}:{item_id}'.encode()).hexdigest()
    return int(h, 16) / 16**64


def downsampling_mask(y: np.ndarray, sampling: float, min_n_majority: int = 3, threshold: float = 0.5) -> np.ndarray:
    """Produce downsampling mask.
    This figures out which one the majority class is and reduces it's size to `sampling`% of the original number

    Example:
       - sampling = 0.6
       - y has 20x 0 and 1200x 1
        --> then 1 is the majority class
        --> mask will be True for all indexes of `y` where it is `0`
        --> mask will be True for 60% of indexes of `y` where it is `1`

    `sampling == 1.0` -> keep all
    `sampling == 0.0` -> keep `min_n_majority` of majority class
    """
    mask = np.ones(len(y), dtype=bool)

    # Ensure we are doing this on binary labels
    y_ = y > threshold

    # Find majority class, we only downsample on that one
    counts = np.unique_counts(y_)
    n_majority = counts.counts.max()
    majority_class = counts.values[counts.counts.argmax()]

    # Ensure that we always keep at least `min_n_majority` of the majority class
    sample_size = int(n_majority * sampling)
    if sample_size < min_n_majority:
        sample_size = min_n_majority

    # Find indexes of items of the majority class
    sample = np.argwhere(y_ == majority_class)

    # Shuffle to keep up the random spirit
    np.random.shuffle(sample)
    mask[sample[:sample_size]] = False
    return mask


def mask_list(lst: list[Any], mask: np.ndarray) -> list[Any]:
    return [item for item, msk in zip(lst, mask, strict=True) if msk]


def to_tuple(v: object):
    if isinstance(v, list):
        return tuple(to_tuple(x) for x in v)
    if isinstance(v, tuple):
        return v
    raise RuntimeError(f'Cannot convert {v} to tuple')


class DictLikeEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        # Translate datetime into a string
        if isinstance(o, datetime):
            return o.strftime('%Y-%m-%dT%H:%M:%S')

        # Translate Path into a string
        if isinstance(o, Path):
            return str(o)

        # Translate pydantic models into dict
        if isinstance(o, BaseModel):
            return o.model_dump()

        # Translate UUID to str
        if isinstance(o, UUID):
            return str(o)

        # Translate Enum to str
        if isinstance(o, Enum):
            return o.value

        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, np.generic):
            return o.item()

        return JSONEncoder.default(self, o)
