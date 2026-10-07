"""Branding audits: the repository itself and GGUF metadata."""

from __future__ import annotations

from helpers import ROOT, upstream_ids
from tokenizer_fixtures import upstream_ids_full

from dsm.audit.scrub import scan_text, scrub_gguf, scrub_repo


def test_repository_is_clean_outside_the_allowlist() -> None:
    """Upstream identifiers may only appear in the files listed in .scrub-allowlist."""
    findings = scrub_repo(ROOT)
    assert findings == [], "\n".join(str(f) for f in findings[:20])


def test_scan_text_is_case_insensitive() -> None:
    patterns = [p.lower() for p in ["abc", "测试"]]
    assert len(scan_text("x ABC y abc 测试", patterns, "f")) == 3


def _write_gguf(path, name: str, tokens: list[str]) -> None:
    from gguf import GGUFWriter

    engine = upstream_ids()["engine_ids"]
    writer = GGUFWriter(str(path), engine["gguf_arch"])
    writer.add_name(name)
    writer.add_block_count(4)  # stored as "<arch>.block_count": the namespace is allowed
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(engine["gguf_tokenizer_pre"])
    writer.add_token_list(tokens)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_gguf_engine_ids_and_vocabulary_are_exempt(tmp_path) -> None:
    vendor_token = "Ġ" + upstream_ids_full()["scrub"]["probe_developers"][0]
    path = tmp_path / "dsm.gguf"
    _write_gguf(path, "DSM1.0-0.8B", ["a", "b", vendor_token])
    assert scrub_gguf(path) == []


def test_gguf_branding_in_metadata_is_reported(tmp_path) -> None:
    up = upstream_ids()
    path = tmp_path / "bad.gguf"
    _write_gguf(path, up["instruct_repo_id"], ["a"])
    findings = scrub_gguf(path)
    assert findings and all(f.path.endswith("general.name") for f in findings)
