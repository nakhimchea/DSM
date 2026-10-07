"""Learning-rate schedules as pure functions of the step (so resuming is exact).

* ``cosine``   - linear warm-up, cosine decay to ``min_lr_ratio``;
* ``wsd``      - warm-up / stable / decay: constant LR, then a final decay over
  ``decay`` of the steps (linear, ``sqrt`` = 1 - sqrt or cosine). The stable
  phase can be extended or branched into an anneal without restarting;
* ``linear``   - linear warm-up, linear decay;
* ``constant`` - linear warm-up, then constant.
"""

from __future__ import annotations

import math

from dsm.train.config import ScheduleConfig


def warmup_steps(schedule: ScheduleConfig, total_steps: int) -> int:
    warm = schedule.warmup
    return int(round(warm * total_steps)) if warm < 1 else int(warm)


def lr_factor(step: int, total_steps: int, schedule: ScheduleConfig) -> float:
    """Multiplier of the peak LR for optimizer step ``step`` (0-based)."""
    total_steps = max(total_steps, 1)
    warm = min(warmup_steps(schedule, total_steps), total_steps)
    floor = schedule.min_lr_ratio
    if warm and step < warm:
        return (step + 1) / warm
    if schedule.name == "constant":
        return 1.0
    if schedule.name == "wsd":
        decay_steps = max(1, int(round(schedule.decay * total_steps)))
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start + 1) / decay_steps)
        if schedule.decay_shape == "sqrt":
            shape = 1.0 - math.sqrt(progress)
        elif schedule.decay_shape == "cosine":
            shape = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            shape = 1.0 - progress
        return floor + (1.0 - floor) * shape
    progress = min(1.0, (step - warm + 1) / max(1, total_steps - warm))
    if schedule.name == "linear":
        return floor + (1.0 - floor) * (1.0 - progress)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
