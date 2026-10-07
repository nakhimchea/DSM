"""``dsm export`` - GGUF conversion, quantisation, quality check and local serving.

dsm export gguf --model work/runs/dpo/final --out exports/dsm --quant Q8_0 Q4_K_M --calibration data/local/calib.txt
dsm export kl --reference exports/dsm/DSM1.0-0.8B-BF16.gguf --candidate exports/dsm/DSM1.0-0.8B-Q4_K_M.gguf --text km=eval/km.txt en=eval/en.txt zh=eval/zh.txt
dsm export serve --gguf exports/dsm/DSM1.0-0.8B-Q8_0.gguf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_gguf(args) -> int:
    from dsm.audit.offline import write_manifest
    from dsm.audit.scrub import scrub_gguf
    from dsm.export import gguf
    from dsm.utils import load_identity, load_upstream

    identity, upstream = load_identity(), load_upstream()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = identity["full_name"]
    base = gguf.convert(
        args.model,
        out / f"{stem}-{args.outtype.upper()}.gguf",
        outtype=args.outtype,
        llama_cpp=args.llama_cpp,
    )
    files = [base]
    imatrix = None
    if args.calibration and any(q.upper().startswith(("Q2", "Q3", "Q4", "IQ")) for q in args.quant):
        imatrix = gguf.compute_imatrix(
            base, args.calibration, out / "imatrix.dat", bin_dir=args.llama_bin, llama_cpp=args.llama_cpp
        )
    for qtype in args.quant:
        target = out / f"{stem}-{qtype.upper()}.gguf"
        files.append(
            gguf.quantize(
                base, target, qtype, imatrix=imatrix, bin_dir=args.llama_bin, llama_cpp=args.llama_cpp
            )
        )
    gguf.write_release_files(args.model, out)
    problems = []
    for path in files:
        problems += [f"{path.name}: {f}" for f in scrub_gguf(path, upstream)]
    if problems:
        print("audit findings:\n  " + "\n  ".join(problems))
        return 1
    write_manifest(out)
    print("\n".join(str(p) for p in files))
    return 0


def cmd_kl(args) -> int:
    from dsm.export.gguf import kl_divergence

    metrics = {}
    for item in args.text:
        name, _, path = item.partition("=")
        result = kl_divergence(
            args.reference, args.candidate, path, bin_dir=args.llama_bin, llama_cpp=args.llama_cpp
        )
        if "kl" in result:
            metrics[f"quant/kl/{name}"] = result["kl"]
        if "top1_agreement" in result:
            metrics[f"quant/top1_agreement/{name}"] = result["top1_agreement"]
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


def cmd_serve(args) -> int:
    from dsm.export.gguf import find_binary
    from dsm.export.llama_server import ServerConfig, start

    cfg = ServerConfig(
        gguf=args.gguf,
        port=args.port,
        ctx=args.ctx,
        gpu_layers=args.gpu_layers,
        parallel=args.parallel,
        api_key=args.api_key,
        webui=args.webui,
    )
    binary = find_binary("llama-server", args.llama_bin, args.llama_cpp)
    process = start(cfg, binary, args.log)
    print(f"DSM is serving on http://{cfg.host}:{cfg.port}/v1 (pid {process.pid}); Ctrl+C to stop")
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsm export", description="Export DSM to GGUF and serve it locally."
    )
    parser.add_argument(
        "--llama-cpp", help="llama.cpp checkout (default: $DSM_LLAMA_CPP or third_party/llama.cpp)"
    )
    parser.add_argument("--llama-bin", help="directory with llama-quantize / llama-server / llama-perplexity")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("gguf", help="convert (and quantise) a DSM model directory")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--outtype", default="bf16", choices=("f32", "f16", "bf16", "q8_0"))
    p.add_argument("--quant", nargs="*", default=["Q8_0"], help="llama-quantize types, e.g. Q8_0 Q6_K Q4_K_M")
    p.add_argument("--calibration", help="km/en/zh text for the importance matrix (used for <= 4-bit types)")
    p.set_defaults(fn=cmd_gguf)
    p = sub.add_parser("kl", help="KL divergence of a quantised GGUF against the reference GGUF")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--text", nargs="+", required=True, help="lang=path pairs")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_kl)
    p = sub.add_parser("serve", help="run llama-server on 127.0.0.1")
    p.add_argument("--gguf", required=True)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--gpu-layers", type=int, default=99)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--api-key")
    p.add_argument("--webui", action="store_true")
    p.add_argument("--log", default="llama-server.log")
    p.set_defaults(fn=cmd_serve)
    args = parser.parse_args(argv)
    if args.command == "gguf" and args.outtype == "q8_0":
        args.quant = [q for q in args.quant if q.upper() != "Q8_0"]
    return int(args.fn(args) or 0)
