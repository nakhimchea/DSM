"""``dsm fetch`` - download the upstream checkpoints once, on a connected machine.

    HF_HUB_OFFLINE=0 dsm fetch --out work/upstream            # base + instruct
    HF_HUB_OFFLINE=0 dsm fetch --out work/upstream --only base --revision <commit>

Repository ids come from ``configs/upstream.yaml``. Each checkpoint lands in
``<out>/<kind>/``; ``<out>/FETCH.json`` records the repository, the exact revision
(commit) and the SHA-256 of every file - the provenance record for NOTICE and the
release checklist. Copy the directory to the training machine; nothing after this
step needs the network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.md", "LICENSE*"]


def fetch(out_dir: str | Path, kinds: list[str], revision: str | None = None) -> dict:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise SystemExit("offline mode is on: run `HF_HUB_OFFLINE=0 dsm fetch ...` on a connected machine")
    from huggingface_hub import HfApi, snapshot_download

    from dsm.utils import load_upstream, sha256_file

    upstream = load_upstream()["upstream"]
    out = Path(out_dir)
    manifest_path = out / "FETCH.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for kind in kinds:
        repo = upstream[f"{kind}_repo_id"]
        commit = HfApi().model_info(repo, revision=revision).sha
        target = out / kind
        snapshot_download(repo, revision=commit, local_dir=target, allow_patterns=PATTERNS)
        files = sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
        manifest[kind] = {
            "repo_id": repo,
            "revision": commit,
            "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "files": {str(p.relative_to(target)): sha256_file(p) for p in files},
        }
        print(f"{kind}: {len(files)} files at revision {commit[:12]} -> {target}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsm fetch", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="work/upstream")
    parser.add_argument("--only", choices=["base", "instruct"], help="fetch one checkpoint")
    parser.add_argument("--revision", help="branch, tag or commit (default: the current main)")
    args = parser.parse_args(argv)
    fetch(args.out, [args.only] if args.only else ["base", "instruct"], args.revision)
    return 0
