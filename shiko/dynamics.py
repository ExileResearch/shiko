"""Pure memory-dynamics math.

Only functions that compute a value belong here. Persistence and write policy
live in the records and maintenance layers, which is what lets a maintenance
sweep evaluate thirty days of decay without writing anything.
"""

from __future__ import annotations

import math
from datetime import datetime


def effective_activation(
    activation: float | None,
    decay_rate: float | None,
    last_updated: datetime | None,
    at: datetime,
) -> float | None:
    """Activation after elapsed time under exponential decay.

    ``None`` stays unknown: an absent activation is not zero, and a missing
    decay rate is not "no decay".
    """
    if activation is None or decay_rate is None or last_updated is None:
        return None
    elapsed_days = (at - last_updated).total_seconds() / 86400.0
    if elapsed_days < 0:
        elapsed_days = 0.0
    return float(activation) * math.exp(-float(decay_rate) * elapsed_days)
