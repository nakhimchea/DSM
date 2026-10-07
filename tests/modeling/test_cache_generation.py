"""Cached decoding must equal uncached forward passes (prefill, decode, continuation, padding)."""

from __future__ import annotations

import torch
from helpers import tiny_dsm_model

from dsm.modeling import cache_dsm
from dsm.serve.generate import SamplingConfig, generate, prefill_position_ids

ATOL, RTOL = 1e-4, 1e-4


def _ids(vocab: int, shape: tuple[int, int], seed: int) -> torch.Tensor:
    return torch.randint(0, vocab, shape, generator=torch.Generator().manual_seed(seed))


def test_prefill_then_decode_matches_full_forward() -> None:
    model = tiny_dsm_model(seed=20)
    ids = _ids(model.config.vocab_size, (2, 30), seed=20)
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits
        cache = cache_dsm.new_cache(model.config)
        out = model(input_ids=ids[:, :20], past_key_values=cache, use_cache=True)
        torch.testing.assert_close(out.logits, full[:, :20], atol=ATOL, rtol=RTOL)
        for t in range(20, 30):
            step = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True).logits
            torch.testing.assert_close(step[:, 0], full[:, t], atol=ATOL, rtol=RTOL)


def test_multi_token_continuation_matches_full_forward() -> None:
    model = tiny_dsm_model(seed=21)
    ids = _ids(model.config.vocab_size, (2, 90), seed=21)
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits
        cache = cache_dsm.new_cache(model.config)
        model(input_ids=ids[:, :10], past_key_values=cache, use_cache=True)
        chunk = model(input_ids=ids[:, 10:77], past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(chunk, full[:, 10:77], atol=ATOL, rtol=RTOL)
        single = model(input_ids=ids[:, 77:78], past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(single[:, 0], full[:, 77], atol=ATOL, rtol=RTOL)
        assert cache_dsm.past_length(cache) == 78


def test_left_padded_batch_matches_per_sample() -> None:
    """Teacher-forced decoding of a left-padded batch == each prompt on its own."""
    model = tiny_dsm_model(seed=22)
    vocab = model.config.vocab_size
    short, long_ = _ids(vocab, (1, 9), seed=1), _ids(vocab, (1, 25), seed=2)
    continuation = _ids(vocab, (2, 6), seed=3)
    pad = 16
    ids = torch.cat([torch.cat([torch.zeros(1, pad, dtype=torch.long), short], 1), long_])
    mask = torch.ones_like(ids)
    mask[0, :pad] = 0
    with torch.no_grad():
        cache = cache_dsm.new_cache(model.config)
        pos = prefill_position_ids(mask, ids)
        out = model(
            input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True
        )
        batched = [out.logits[:, -1]]
        last = pos[:, -1]
        for t in range(continuation.shape[1]):
            last = last + 1
            mask = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], 1)
            step = model(
                input_ids=continuation[:, t : t + 1],
                attention_mask=mask,
                position_ids=last[:, None],
                past_key_values=cache,
                use_cache=True,
            ).logits
            batched.append(step[:, -1])
        for row, prompt in enumerate((short, long_)):
            seq = torch.cat([prompt, continuation[row : row + 1]], 1)
            ref = model(input_ids=seq, use_cache=False).logits[0, prompt.shape[1] - 1 :]
            got = torch.stack([b[row] for b in batched])
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)


def test_hf_generate_matches_own_loop_and_uncached_forward() -> None:
    model = tiny_dsm_model(seed=23)
    ids = _ids(model.config.vocab_size, (2, 12), seed=23)
    with torch.no_grad():
        hf = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=8,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=0,
            eos_token_id=None,
        )
        ours = generate(model, ids, sampling=SamplingConfig(max_new_tokens=8), keep_logits=True)
    assert torch.equal(hf.sequences, ours.sequences)
    for hf_step, our_step in zip(hf.logits, ours.step_logits, strict=True):
        torch.testing.assert_close(our_step, hf_step.float(), atol=ATOL, rtol=RTOL)
    with torch.no_grad():
        uncached = model(input_ids=ours.sequences, use_cache=False).logits
    prompt_len = ids.shape[1]
    for i, step in enumerate(ours.step_logits):
        torch.testing.assert_close(step, uncached[:, prompt_len - 1 + i], atol=ATOL, rtol=RTOL)


def test_generate_stops_at_eos_and_pads() -> None:
    model = tiny_dsm_model(seed=24)
    ids = _ids(model.config.vocab_size, (1, 5), seed=24)
    with torch.no_grad():
        first = generate(model, ids, sampling=SamplingConfig(max_new_tokens=1)).new_tokens[0, 0].item()
        res = generate(model, ids, sampling=SamplingConfig(max_new_tokens=10, eos_token_ids=[first]))
    assert res.new_tokens.shape[1] == 1


def test_sampling_is_reproducible_with_seed() -> None:
    model = tiny_dsm_model(seed=25)
    ids = _ids(model.config.vocab_size, (1, 6), seed=25)
    cfg = SamplingConfig(
        max_new_tokens=6, temperature=0.8, top_k=20, top_p=0.9, seed=1234, presence_penalty=0.5
    )
    with torch.no_grad():
        a = generate(model, ids, sampling=cfg).new_tokens
        b = generate(model, ids, sampling=cfg).new_tokens
    assert torch.equal(a, b)
