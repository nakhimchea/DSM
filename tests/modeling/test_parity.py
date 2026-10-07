"""DSM must reproduce the reference (upstream) implementation numerically.

Both models get the same randomised weights (every parameter, so a swapped norm
or projection cannot hide behind default initialisation) and are compared on
logits - not greedy tokens, which are flaky under near-ties.
"""

from __future__ import annotations

import pytest
import torch
from helpers import randomize_, reference_text_model, tiny_dsm_model

ATOL = 2e-4
RTOL = 1e-4


def _pair(seed: int = 0, attn: str = "eager", **overrides):
    ref = randomize_(reference_text_model(attn=attn, **overrides), seed=seed)
    ours = tiny_dsm_model(seed=seed + 1, attn=attn, **overrides)
    ours.load_state_dict(ref.state_dict(), strict=True)
    return ref, ours


@pytest.mark.parametrize("seq_len", [1, 5, 63, 64, 65, 130])
def test_logits_match_reference(seq_len: int) -> None:
    ref, ours = _pair(seed=seq_len)
    ids = torch.randint(
        0, ref.config.vocab_size, (2, seq_len), generator=torch.Generator().manual_seed(seq_len)
    )
    with torch.no_grad():
        expected = ref(input_ids=ids, use_cache=False).logits
        actual = ours(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_state_dict_keys_identical() -> None:
    ref, ours = _pair()
    assert set(ref.state_dict()) == set(ours.state_dict())
    for key, value in ref.state_dict().items():
        assert ours.state_dict()[key].shape == value.shape, key


def test_left_padding_matches_reference() -> None:
    ref, ours = _pair(seed=3)
    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(0, ref.config.vocab_size, (2, 40), generator=gen)
    mask = torch.ones_like(ids)
    mask[0, :7] = 0  # row 0 is left-padded
    with torch.no_grad():
        expected = ref(input_ids=ids, attention_mask=mask, use_cache=False).logits
        actual = ours(input_ids=ids, attention_mask=mask, use_cache=False).logits
    real = mask.bool()
    torch.testing.assert_close(actual[real], expected[real], atol=ATOL, rtol=RTOL)
    assert torch.isfinite(actual).all()


def test_grouped_value_heads_match_reference() -> None:
    """``linear_num_value_heads = 2 * linear_num_key_heads`` (q/k heads repeated)."""
    ref, ours = _pair(seed=5, linear_num_value_heads=8, linear_num_key_heads=4)
    ids = torch.randint(0, ref.config.vocab_size, (2, 70), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        torch.testing.assert_close(
            ours(input_ids=ids, use_cache=False).logits,
            ref(input_ids=ids, use_cache=False).logits,
            atol=ATOL,
            rtol=RTOL,
        )


def test_sdpa_matches_eager() -> None:
    eager = tiny_dsm_model(seed=11, attn="eager")
    sdpa = tiny_dsm_model(seed=11, attn="sdpa")
    ids = torch.randint(0, eager.config.vocab_size, (2, 50), generator=torch.Generator().manual_seed(11))
    mask = torch.ones_like(ids)
    mask[1, :9] = 0
    with torch.no_grad():
        for kwargs in ({}, {"attention_mask": mask}):
            torch.testing.assert_close(
                sdpa(input_ids=ids, use_cache=False, **kwargs).logits,
                eager(input_ids=ids, use_cache=False, **kwargs).logits,
                atol=1e-4,
                rtol=1e-4,
            )
