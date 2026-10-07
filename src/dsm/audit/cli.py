"""``dsm audit`` - branding scrub and offline-readiness checks.

dsm audit model  <model_dir>               # no upstream branding outside LICENSE/NOTICE
dsm audit gguf   <file.gguf>               # same for exported GGUF metadata
dsm audit repo   [--root .]                # repository files outside .scrub-allowlist
dsm audit offline <model_dir> [--write-manifest] [--smoke]
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsm audit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="what", required=True)
    p_model = sub.add_parser("model", help="scan a model directory for upstream branding")
    p_model.add_argument("path")
    p_gguf = sub.add_parser("gguf", help="scan GGUF metadata for upstream branding")
    p_gguf.add_argument("path")
    p_repo = sub.add_parser("repo", help="scan repository files (outside .scrub-allowlist)")
    p_repo.add_argument("--root", default=".")
    p_off = sub.add_parser("offline", help="offline-readiness checks")
    p_off.add_argument("path")
    p_off.add_argument("--write-manifest", action="store_true", help="(re)write SHA256SUMS")
    p_off.add_argument("--smoke", action="store_true", help="load and generate with networking blocked")
    args = parser.parse_args(argv)

    from dsm.audit import offline, scrub

    if args.what in ("model", "gguf", "repo"):
        if args.what == "model":
            findings = scrub.scrub_model_dir(args.path)
        elif args.what == "gguf":
            findings = scrub.scrub_gguf(args.path)
        else:
            findings = scrub.scrub_repo(args.root)
        for finding in findings:
            print(f"FOUND {finding}")
        print(f"{len(findings)} finding(s)")
        return 1 if findings else 0

    problems = offline.check_files(args.path)
    if args.write_manifest:
        print(f"wrote {offline.write_manifest(args.path)}")
    else:
        problems += offline.verify_manifest(args.path)
    if args.smoke:
        try:
            text = offline.offline_generation_smoke(args.path)
            print(f"offline generation OK: {text!r}")
        except Exception as exc:  # report, do not crash
            problems.append(f"offline generation failed: {type(exc).__name__}: {exc}")
    for problem in problems:
        print(f"PROBLEM {problem}", file=sys.stderr)
    print("offline audit: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
