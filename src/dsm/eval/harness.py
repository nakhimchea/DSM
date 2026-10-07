"""Thin wrapper around EleutherAI lm-evaluation-harness for standard benchmarks.

    dsm eval lm-eval --model work/runs/sft/final --tasks belebele_khm_Khmr,belebele_eng_Latn,belebele_zho_Hans

The DSM checkpoint loads through ``trust_remote_code`` (the model directory
carries its own modeling code), so no lm-eval plugin is needed. lm-eval is an
optional dependency (``pip install 'dsm-llm[eval]'``).

Recommended tasks (log-likelihood based unless noted):

* Khmer: ``belebele_khm_Khmr``; plus the in-house FLORES chrF++ and SIB-style
  multiple choice (``dsm eval translate`` / ``dsm eval choice``)
* English: ``belebele_eng_Latn``, ``arc_easy``, ``hellaswag``, ``gsm8k`` (generative), ``ifeval`` (generative)
* Chinese: ``belebele_zho_Hans``, ``cmmlu``, ``ceval-valid``

Belebele has 900 questions per language; its standard error is ~1.6 points,
so differences under ~3 points between checkpoints are noise.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_TASKS = ("belebele_khm_Khmr", "belebele_eng_Latn", "belebele_zho_Hans")


def model_args(model_dir: str | Path, dtype: str = "bfloat16") -> str:
    return f"pretrained={Path(model_dir).resolve()},trust_remote_code=True,dtype={dtype}"


def flatten(results: dict) -> dict[str, float]:
    """``{"lm_eval/<task>/<metric>": value}`` from lm-eval's ``results`` section."""
    out = {}
    for task, metrics in results.get("results", {}).items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                name = key.split(",")[0]
                out[f"lm_eval/{task}/{name}"] = float(value)
    return out


def run_lm_eval(
    model_dir: str | Path,
    tasks: list[str] | tuple[str, ...] = DEFAULT_TASKS,
    *,
    batch_size: int | str = 8,
    limit: int | None = None,
    dtype: str = "bfloat16",
    num_fewshot: int | None = None,
) -> dict[str, float]:
    try:
        import lm_eval
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("lm-eval is not installed: pip install 'dsm-llm[eval]'") from exc
    import dsm.modeling  # noqa: F401 - registers DSM with the Auto classes

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args(model_dir, dtype),
        tasks=list(tasks),
        batch_size=batch_size,
        limit=limit,
        num_fewshot=num_fewshot,
    )
    return flatten(results or {})
