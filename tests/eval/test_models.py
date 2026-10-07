"""Model-backed evaluation on the tiny checkpoint: generators, bpb / KL, multiple choice, translation, dissection."""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import torch
from tokenizer_fixtures import CORPUS
from train.fixtures import packed_corpus, tiny_model_dir

from dsm.eval.generator import EndpointGenerator, HFGenerator, Request


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, tiny_tokenizer_json):
    return tiny_model_dir(tmp_path_factory.mktemp("eval") / "model", tiny_tokenizer_json, seed=3)


def test_hf_generator_batches_and_decodes(model_dir):
    generator = HFGenerator(model_dir, device="cpu", batch_size=2)
    requests = [
        Request([{"role": "user", "content": text}], max_new_tokens=6)
        for text in ("Hello!", "សួស្តី!", "你好！")
    ]
    replies = generator.generate(requests)
    assert len(replies) == 3 and all(isinstance(r.text, str) for r in replies)
    # batching does not change greedy outputs (left padding is handled)
    single = HFGenerator(model_dir, device="cpu", batch_size=1).generate(requests)
    assert [r.text for r in replies] == [r.text for r in single]


class _ToolServer(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("tools"):
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_exchange_rate",
                            "arguments": '{"base": "USD", "quote": "KHR"}',
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "plain answer"}
        data = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_endpoint_generator_parses_openai_tool_calls():
    httpd = HTTPServer(("127.0.0.1", 0), _ToolServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        generator = EndpointGenerator(f"http://127.0.0.1:{httpd.server_address[1]}/v1")
        tools = [
            {
                "type": "function",
                "function": {"name": "get_exchange_rate", "parameters": {"type": "object", "properties": {}}},
            }
        ]
        with_tools, plain = generator.generate(
            [
                Request([{"role": "user", "content": "rate?"}], tools=tools),
                Request([{"role": "user", "content": "hi"}]),
            ]
        )
        assert with_tools.tool_calls == [
            {"name": "get_exchange_rate", "arguments": {"base": "USD", "quote": "KHR"}}
        ]
        assert plain.content == "plain answer" and not plain.tool_calls
    finally:
        httpd.shutdown()
    with pytest.raises(ValueError, match="non-local"):
        EndpointGenerator("https://api.example.com/v1")


def test_bpb_and_kl(model_dir, tmp_path, tiny_tokenizer_json):
    from dsm.eval.bpb import evaluate_bpb, mean_token_kl, token_byte_lengths
    from dsm.modeling import DSMForCausalLM

    prefix = packed_corpus(
        tmp_path / "eval_en", tiny_tokenizer_json, [t for t in CORPUS if t.isascii()], "en"
    )
    model = DSMForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    metrics = evaluate_bpb(
        model, model_dir / "tokenizer.json", {"en": str(prefix)}, seq_len=32, max_tokens=256, batch_size=2
    )
    # manual: sum of NLL / (ln 2 * bytes) over the same windows
    from dsm.train.config import DataSource
    from dsm.train.datasets import eval_windows

    lengths = token_byte_lengths(tiny_tokenizer_json, model.config.vocab_size)
    nll = n_bytes = 0.0
    for ids, labels, positions in eval_windows(DataSource(str(prefix)), 32, 256):
        with torch.no_grad():
            logits = model(
                input_ids=torch.from_numpy(ids)[None],
                position_ids=torch.from_numpy(positions)[None],
                use_cache=False,
            ).logits[0, :-1]
        targets = torch.from_numpy(labels[1:])
        keep = targets != -100
        nll += float(torch.nn.functional.cross_entropy(logits[keep], targets[keep], reduction="sum"))
        n_bytes += float(lengths[targets[keep].numpy()].sum())
    assert metrics["bpb/en"] == pytest.approx(nll / (math.log(2) * n_bytes), rel=1e-4)
    assert mean_token_kl(model, model, {"en": str(prefix)}, seq_len=32, max_tokens=128)[
        "quant/kl/en"
    ] == pytest.approx(0.0, abs=1e-6)


def test_multiple_choice(model_dir):
    from tokenizers import Tokenizer

    from dsm.eval.choice import evaluate_choices, option_logprobs
    from dsm.modeling import DSMForCausalLM

    model = DSMForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template = (model_dir / "chat_template.jinja").read_text()
    items = [
        {"question": "Pick one.", "options": ["red", "blue", "green"], "answer": 1, "lang": "en"},
        {"question": "ជ្រើសរើស", "options": ["ក", "ខ"], "answer": 0, "lang": "km"},
    ]
    result = evaluate_choices(model, tokenizer, template, items)
    assert set(result.per_lang) == {"en", "km"} and len(result.predictions) == 2
    scores = option_logprobs(model, tokenizer, template, "Pick one.", ["red", "blue"], torch.device("cpu"))
    assert len(scores) == 2 and all(s < 0 for s in scores)


def test_translation_eval_with_scripted_generator():
    from dsm.eval.generator import reply_from_text
    from dsm.eval.translation import evaluate_translation

    rows = [
        {"src": "Good morning", "tgt": "អរុណសួស្តី", "src_lang": "en", "tgt_lang": "km"},
        {"src": "Thank you", "tgt": "谢谢", "src_lang": "en", "tgt_lang": "zh"},
    ]
    perfect = {"Good morning": "អរុណសួស្តី", "Thank you": "谢谢"}

    def generate(requests):
        return [
            reply_from_text(next(v for k, v in perfect.items() if k in r.messages[-1]["content"]), None)
            for r in requests
        ]

    metrics = evaluate_translation(generate, rows)
    assert metrics == {
        "translation/chrf++/en-km": pytest.approx(100.0),
        "translation/chrf++/en-zh": pytest.approx(100.0),
    }


def test_dissection(model_dir, tmp_path):
    from dsm.cli import main
    from dsm.dissect.params import cache_report, parameter_report
    from dsm.modeling import DSMConfig

    config = DSMConfig()
    params = parameter_report(config)
    assert params["total"] == 752_393_024 and params["layer_types"] == {
        "linear_attention": 18,
        "full_attention": 6,
    }
    cache = cache_report(config, 4096)
    assert cache["kv_bytes_per_token"] == 12 * 1024
    assert cache["gdn_state_bytes_per_sequence"] >= 18 * 2**20

    texts = {lang: tmp_path / f"{lang}.txt" for lang in ("en", "km")}
    texts["en"].write_text("\n\n".join(t for t in CORPUS[:10] if t.isascii()))
    texts["km"].write_text("\n\n".join(t for t in CORPUS[:10] if not t.isascii()))
    out = tmp_path / "report"
    assert (
        main(
            [
                "dissect",
                "--model",
                str(model_dir),
                "--text",
                f"en={texts['en']}",
                f"km={texts['km']}",
                "--seq-len",
                "64",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / "report.json").read_text())
    layers = report["layers"]["km"]
    n_layers = len(json.loads((model_dir / "config.json").read_text())["layer_types"])
    assert (
        len(layers["block_influence"])
        == len(layers["ablation_loss"])
        == len(layers["logit_lens_loss"])
        == n_layers
    )
    assert layers["logit_lens_loss"][-1] == pytest.approx(
        layers["baseline_loss"], rel=1e-3
    )  # last read-out = the model
    assert "## Layer importance" in (out / "report.md").read_text()
