"""GGUF export shim, llama-server launcher; llama.cpp-backed checks run when a checkout is available."""

from __future__ import annotations

import json
import os
import socket
import urllib.request
from pathlib import Path

import pytest
import torch

from dsm.export.gguf import attention_interval, metadata_override, uses_dsm_pretokenizer
from dsm.export.llama_server import ServerConfig, server_command

LLAMA_CPP = os.environ.get("DSM_LLAMA_CPP")
needs_llama_cpp = pytest.mark.skipif(
    not (LLAMA_CPP and (Path(LLAMA_CPP) / "conversion" / "__init__.py").is_file()),
    reason="set DSM_LLAMA_CPP to a llama.cpp checkout (scripts/build_llama_cpp.sh)",
)


def test_attention_interval():
    lin, full = "linear_attention", "full_attention"
    assert attention_interval([lin, lin, lin, full] * 6) == 4
    assert attention_interval([lin, full] * 2) == 2
    with pytest.raises(ValueError, match="periodic"):
        attention_interval([lin, full, lin, lin, full])
    with pytest.raises(ValueError, match="no full-attention"):
        attention_interval([lin, lin])


def test_pretokenizer_check(tmp_path, tiny_tokenizer_json):
    model = tmp_path / "m"
    model.mkdir()
    data = json.loads(Path(tiny_tokenizer_json).read_text())
    (model / "tokenizer.json").write_text(json.dumps(data))
    assert uses_dsm_pretokenizer(model)
    data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r"\s+"
    (model / "tokenizer.json").write_text(json.dumps(data))
    assert not uses_dsm_pretokenizer(model)


def test_metadata_has_no_upstream_names():
    from tokenizer_fixtures import upstream_ids_full

    from dsm.utils import load_identity

    meta = json.dumps(metadata_override(load_identity())).lower()
    assert "doslarb" in meta and "dsm" in meta
    assert not any(term in meta for term in upstream_ids_full()["scrub"]["denylist"])


def test_server_command_is_local_and_offline():
    cmd = server_command(ServerConfig(gguf="m.gguf", port=9000, api_key="k"), "llama-server")
    joined = " ".join(cmd)
    assert (
        "--host 127.0.0.1" in joined and "--port 9000" in joined and "--jinja" in cmd and "--offline" in cmd
    )
    assert "--no-webui" in cmd and "--reasoning-format none" in joined and "--api-key k" in joined
    with pytest.raises(ValueError, match="localhost"):
        server_command(ServerConfig(gguf="m.gguf", host="0.0.0.0"), "llama-server")
    assert "0.0.0.0" in server_command(
        ServerConfig(gguf="m.gguf", host="0.0.0.0", allow_remote_bind=True), "x"
    )


# -- with a llama.cpp checkout ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory, tiny_tokenizer_json):
    from helpers import randomize_
    from train.fixtures import tiny_model_dir

    from dsm.export.gguf import convert
    from dsm.modeling import DSMForCausalLM
    from dsm.train.checkpoint import save_model_dir

    root = tmp_path_factory.mktemp("gguf")
    base = tiny_model_dir(root / "base", tiny_tokenizer_json)
    model = DSMForCausalLM.from_pretrained(base, dtype=torch.float32)
    randomize_(model, seed=11, scale=2.5)  # varied greedy tokens: a sharp test of the conversion
    model_dir = save_model_dir(model, root / "model", carry_from=base, dtype="float32")
    return model_dir, convert(model_dir, root / "tiny-F32.gguf", outtype="f32", llama_cpp=LLAMA_CPP)


@pytest.mark.llamacpp
@needs_llama_cpp
def test_gguf_metadata_and_audit(tiny_gguf):
    import gguf as gguf_py

    from dsm.audit.scrub import scrub_gguf
    from dsm.utils import load_identity, load_upstream

    model_dir, path = tiny_gguf
    ids = load_upstream()["upstream"]["engine_ids"]
    reader = gguf_py.GGUFReader(str(path))
    field = {k: reader.fields[k].contents() for k in reader.fields if not k.startswith("tokenizer.ggml.")}
    arch = ids["gguf_arch"]
    assert field["general.architecture"] == arch and field["general.name"] == load_identity()["full_name"]
    assert field["general.organization"] == load_identity()["developer"]
    assert field[f"{arch}.full_attention_interval"] == 2  # derived from layer_types, not the default 4
    assert reader.fields["tokenizer.ggml.pre"].contents() == ids["gguf_tokenizer_pre"]
    assert field["tokenizer.chat_template"] == (model_dir / "chat_template.jinja").read_text()
    assert not any(t.name.startswith("blk.4") for t in reader.tensors)  # no MTP block
    assert scrub_gguf(path) == []


def _binary(name: str) -> Path | None:
    from dsm.export.gguf import find_binary

    try:
        return find_binary(name, os.environ.get("DSM_LLAMA_BIN"), LLAMA_CPP)
    except FileNotFoundError:
        return None


@pytest.mark.llamacpp
@needs_llama_cpp
def test_llama_server_matches_pytorch(tiny_gguf):
    """Greedy tokens, top-5 log-probs and chat-template rendering agree between llama.cpp and DSM."""
    binary = _binary("llama-server")
    if binary is None:
        pytest.skip("llama-server binary not built")
    from tokenizers import Tokenizer

    from dsm.export.llama_server import start
    from dsm.modeling import DSMForCausalLM
    from dsm.serve.generate import SamplingConfig, generate
    from dsm.tokenizer.render import render
    from dsm.tools.bank_sim import BankSim

    model_dir, path = tiny_gguf
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = start(ServerConfig(gguf=str(path), port=port, ctx=1024, gpu_layers=0, parallel=1), binary)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(route, payload):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{route}",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        return json.loads(opener.open(request, timeout=60).read())

    try:
        model = DSMForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        for text in ("Hello! How can I help you today?", "សួស្តី! ខ្ញុំអាចជួយអ្វីបាន?", "你好！今天的汇率是多少？"):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            ours = generate(
                model, torch.tensor([ids]), sampling=SamplingConfig(max_new_tokens=16), keep_logits=True
            )
            theirs = post(
                "/completion",
                {
                    "prompt": ids,
                    "n_predict": 16,
                    "temperature": 0,
                    "samplers": ["top_k"],
                    "top_k": 1,
                    "return_tokens": True,
                    "cache_prompt": False,
                    "n_probs": 5,
                },
            )
            assert theirs["tokens"] == ours.new_tokens[0].tolist()
            logprobs = torch.log_softmax(ours.step_logits[0][0], -1)
            for entry in theirs["completion_probabilities"][0]["top_logprobs"]:
                assert abs(logprobs[entry["id"]].item() - entry["logprob"]) < 1e-2
        template = (model_dir / "chat_template.jinja").read_text()
        tools = BankSim().registry().schemas(["get_exchange_rate"])
        conversations = [
            ({"messages": [{"role": "user", "content": "សួស្តី"}]}),
            (
                {
                    "messages": [
                        {"role": "system", "content": "Be brief."},
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                        {"role": "user", "content": "你好"},
                    ]
                }
            ),
            ({"messages": [{"role": "user", "content": "Rate?"}], "tools": tools}),
        ]
        for payload in conversations:
            expected = render(
                payload["messages"], template, tools=payload.get("tools"), add_generation_prompt=True
            )
            assert post("/apply-template", payload)["prompt"] == expected
    finally:
        process.terminate()
        process.wait(timeout=30)
