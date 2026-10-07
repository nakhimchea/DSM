"""Mixture planning: target shares + available tokens + epoch caps -> sampling weights.

Example: Khmer should be 45 % of a 2B-token CPT run, but only 200M good Khmer
tokens exist; at most 4 epochs gives 800M = 40 %. ``plan_mixture`` caps Khmer
at 40 % and gives the remaining 5 points to the other sources in proportion to
their targets, then reports epochs per source. The resulting weights go into
the training config (``train: [{path, weight}]``); the trainer's sampler
reproduces them within sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MixturePlan:
    weights: dict[str, float]
    tokens: dict[str, float]
    epochs: dict[str, float]
    capped: list[str]


def plan_mixture(
    available: dict[str, float],
    targets: dict[str, float],
    total_tokens: float,
    max_epochs: dict[str, float] | float = 4.0,
) -> MixturePlan:
    if set(available) != set(targets):
        raise ValueError(f"sources differ: {sorted(set(available) ^ set(targets))}")
    caps = {
        name: (max_epochs if isinstance(max_epochs, (int, float)) else max_epochs.get(name, 4.0))
        for name in targets
    }
    share_sum = sum(targets.values())
    shares = {name: value / share_sum for name, value in targets.items()}
    fixed: dict[str, float] = {}
    while True:
        free = {n: s for n, s in shares.items() if n not in fixed}
        remaining = 1.0 - sum(fixed.values())
        free_sum = sum(free.values())
        if not free or free_sum <= 0:
            break
        proposal = {n: s / free_sum * remaining for n, s in free.items()}
        over = {
            n: available[n] * caps[n] / total_tokens
            for n, p in proposal.items()
            if p * total_tokens > available[n] * caps[n]
        }
        if not over:
            fixed.update(proposal)
            break
        fixed.update(over)
    total_share = sum(fixed.values())
    if total_share < 1.0 - 1e-9:
        raise ValueError(
            "not enough data for the token budget within the epoch caps; lower total_tokens or raise caps"
        )
    weights = {n: fixed[n] for n in targets}
    tokens = {n: weights[n] * total_tokens for n in targets}
    epochs = {n: tokens[n] / available[n] if available[n] else float("inf") for n in targets}
    capped = [n for n in targets if abs(weights[n] - shares[n]) > 1e-9 and epochs[n] >= caps[n] - 1e-9]
    return MixturePlan(weights=weights, tokens=tokens, epochs=epochs, capped=capped)


def repeats_for_shares(
    counts: dict[str, int], targets: dict[str, float], total: int | None = None
) -> dict[str, float]:
    """SFT: ``repeat`` factors so that each file makes up its target share of conversations."""
    share_sum = sum(targets.values())
    total = total or sum(counts.values())
    return {
        name: round(targets[name] / share_sum * total / counts[name], 3)
        for name in targets
        if counts.get(name)
    }
