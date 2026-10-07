"""Packed token files, mixture sampling, SFT packing, schedules, configs and bits per byte."""

from __future__ import annotations

import numpy as np
import pytest
from tokenizer_fixtures import CORPUS

from dsm.data.packed import PackedTokens, PackedWriter
from dsm.train.config import DataSource, ScheduleConfig, config_from_dict, load_config
from dsm.train.datasets import PretrainMixture, SFTData, mask_document_starts
from dsm.train.schedules import lr_factor

from .fixtures import SYSTEM_PROMPT, packed_corpus, sft_file


def _write(prefix, docs, vocab=1000, eos=0):
    with PackedWriter(prefix, vocab_size=vocab, eos_id=eos, meta={"name": prefix.name}) as writer:
        for doc in docs:
            writer.add(doc)
    return PackedTokens(prefix)


def test_packed_round_trip_and_windows(tmp_path):
    docs = [[5, 6, 7], [8, 9], list(range(10, 20)), [21]]
    data = _write(tmp_path / "toy", docs)
    assert data.n_docs == 4 and data.n_tokens == sum(len(d) + 1 for d in docs)
    assert data.doc(1).tolist() == [8, 9, 0]
    ids, positions = data.window(0, 8)
    assert ids.tolist() == [5, 6, 7, 0, 8, 9, 0, 10]
    assert positions.tolist() == [0, 1, 2, 3, 0, 1, 2, 0]
    ids, positions = data.window(1, 8)  # starts inside the third document
    assert ids[0] == 11 and positions[0] == 0 and positions.tolist()[:3] == [0, 1, 2]
    labels = mask_document_starts(ids, positions)
    assert (labels[positions == 0] == -100).all() and (labels[positions > 0] == ids[positions > 0]).all()
    with (
        pytest.raises(ValueError, match="outside the vocabulary"),
        PackedWriter(tmp_path / "bad", vocab_size=10, eos_id=0) as writer,
    ):
        writer.add([3, 99])


def test_truncated_dataset_is_rejected(tmp_path):
    data = _write(tmp_path / "toy", [[1, 2, 3]])
    with open(data.prefix.with_name("toy.bin"), "ab") as handle:
        handle.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="disagree"):
        PackedTokens(tmp_path / "toy")


def test_mixture_weights_determinism_and_epochs(tmp_path):
    a = _write(tmp_path / "a", [list(range(1, 400))])
    b = _write(tmp_path / "b", [list(range(1, 100))])
    sources = [DataSource(str(a.prefix), weight=0.7), DataSource(str(b.prefix), weight=0.3)]
    mix = PretrainMixture(sources, seq_len=10, total_samples=20_000, seed=3)
    share = np.bincount(mix.choices, minlength=2) / len(mix)
    assert abs(share[0] - 0.7) < 0.01 and abs(share[1] - 0.3) < 0.01
    again = PretrainMixture(sources, seq_len=10, total_samples=20_000, seed=3)
    for index in (0, 17, 999, 19_999):
        for x, y in zip(mix.sample(index), again.sample(index), strict=True):
            assert np.array_equal(x, y)
    epochs = mix.epochs_used(20_000)
    assert epochs["a"] == pytest.approx(0.7 * 20_000 / 40, rel=0.02) and epochs["b"] > epochs["a"]
    # every window of a source is used once per epoch
    windows = [mix._window(1, rank) for rank in range(mix.n_windows[1])]
    assert sorted(windows) == list(range(mix.n_windows[1]))


def test_sft_packing_and_labels(tmp_path, tiny_tokenizer_json):
    from tokenizers import Tokenizer

    from dsm.data.sft.schema import template_messages
    from dsm.tokenizer.render import build_template, encode_for_training
    from dsm.utils import read_jsonl

    tokenizer = Tokenizer.from_file(str(tiny_tokenizer_json))
    template = build_template(SYSTEM_PROMPT)
    path = sft_file(tmp_path / "sft.jsonl")
    data = SFTData(
        [DataSource(str(path))],
        tokenizer,
        template,
        seq_len=256,
        epochs=2,
        seed=0,
        cache_dir=tmp_path / "cache",
    )
    assert sum(len(row) for row in data.rows) == 2 * len(list(read_jsonl(path)))
    for r in range(len(data)):
        batch = data.batch([r])
        ids, labels, positions, mask = (
            batch[k][0].numpy() for k in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        used = int(mask.sum())
        assert used <= 256 and (labels[used:] == -100).all()
        starts = np.nonzero(positions[:used] == 0)[0]
        assert len(starts) == len(data.rows[r])
    # labels of one example are exactly the assistant tokens of encode_for_training
    record = next(iter(read_jsonl(path)))
    expected = encode_for_training(tokenizer, template_messages(record), template)
    single = SFTData(
        [DataSource(str(path))], tokenizer, template, seq_len=256, epochs=1, seed=0, packing=False
    )
    found = [
        single.pool[i]
        for i in range(len(single.pool))
        if single.pool[i].input_ids.tolist() == expected.input_ids
    ]
    assert found and found[0].labels.tolist() == expected.labels
    # cached tokenisation gives identical examples
    cached = SFTData(
        [DataSource(str(path))],
        tokenizer,
        template,
        seq_len=256,
        epochs=2,
        seed=0,
        cache_dir=tmp_path / "cache",
    )
    assert [r for r in cached.rows] == data.rows


def test_sft_repeat_and_length_filter(tmp_path, tiny_tokenizer_json):
    from tokenizers import Tokenizer

    from dsm.tokenizer.render import build_template

    tokenizer = Tokenizer.from_file(str(tiny_tokenizer_json))
    template = build_template(SYSTEM_PROMPT)
    path = sft_file(tmp_path / "sft.jsonl")
    data = SFTData([DataSource(str(path), repeat=2.5)], tokenizer, template, seq_len=256, epochs=1, seed=0)
    assert data.stats["sft"]["used"] == 20  # 8 records x 2 + 4
    cutoff = int(np.median([len(e) for e in data.pool]))
    short = SFTData([DataSource(str(path))], tokenizer, template, seq_len=cutoff, epochs=1, seed=0)
    assert short.stats["sft"]["too_long"] > 0 and all(len(e) <= cutoff for e in short.pool)


@pytest.mark.parametrize("name", ["cosine", "wsd", "linear", "constant"])
def test_schedules(name):
    schedule = ScheduleConfig(name=name, warmup=10, decay=0.2, min_lr_ratio=0.1)
    values = [lr_factor(s, 100, schedule) for s in range(100)]
    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)  # linear warm-up
    assert max(values) <= 1.0 + 1e-9
    if name == "constant":
        assert values[-1] == 1.0
    else:
        assert values[-1] == pytest.approx(0.1)
    if name == "wsd":
        assert all(v == 1.0 for v in values[10:80]) and values[80] < 1.0


def test_config_loading(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "stage: cpt\nmodel: m\noutput_dir: out\nseq_len: 1024\nmicro_batch_size: 2\ngrad_accum: 4\nmax_tokens: 1.0e6\n"
        "data:\n  train:\n    - {path: a, weight: 2}\n  eval: [b]\noptimizer: {lr: 1.0e-4, betas: [0.9, 0.95]}\n"
    )
    cfg = load_config(path, ["optimizer.lr=5e-5", "schedule.name=wsd"])
    assert cfg.optimizer.lr == 5e-5 and cfg.optimizer.betas == (0.9, 0.95) and cfg.schedule.name == "wsd"
    assert cfg.tokens_per_step == 8192 and cfg.steps_for_tokens() == 123
    assert cfg.train[0].weight == 2 and cfg.eval[0].path == "b"
    with pytest.raises(ValueError, match="unknown config"):
        config_from_dict(
            {"stage": "cpt", "model": "m", "output_dir": "o", "train": ["a"], "learning_rate": 1}
        )
    # new_rows: the first new row comes from new_rows_from or the model's extension.json
    with pytest.raises(ValueError, match="new_rows_from"):
        config_from_dict(
            {
                "stage": "pt",
                "model_config": "m.yaml",
                "tokenizer": "t",
                "output_dir": "o",
                "train": ["a"],
                "trainable": "new_rows",
            }
        )
    cfg = config_from_dict(
        {"stage": "cpt", "model": "m", "output_dir": "o", "train": ["a"], "trainable": "new_rows"}
    )
    assert cfg.new_rows_from is None


def test_token_byte_lengths_match_utf8(tiny_tokenizer_json):
    from tokenizers import Tokenizer

    from dsm.eval.bpb import bits_per_byte, token_byte_lengths

    tokenizer = Tokenizer.from_file(str(tiny_tokenizer_json))
    lengths = token_byte_lengths(tiny_tokenizer_json)
    for text in CORPUS[:10]:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        assert int(lengths[ids].sum()) == len(text.encode("utf-8"))
    im_end = tokenizer.token_to_id("<|im_end|>")
    assert lengths[im_end] == 0 and lengths[tokenizer.token_to_id("<tool_call>")] == len("<tool_call>")
    assert bits_per_byte(np.log(2) * 8, 8) == pytest.approx(1.0)


def test_packed_corpus_fixture(tmp_path, tiny_tokenizer_json):
    prefix = packed_corpus(tmp_path / "en", tiny_tokenizer_json, CORPUS[:5], "en")
    data = PackedTokens(prefix)
    assert data.name == "en" and data.n_docs == 5 and data.meta["license"] == "test"


def test_shipped_configs_load(repo_root):
    for path in sorted((repo_root / "configs/train").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.train, path
        if cfg.stage in ("pt", "cpt", "anneal"):
            assert cfg.steps_for_tokens() > 0
            assert abs(sum(s.weight for s in cfg.train) - 1.0) < 1e-6, path  # mixture weights are fractions
