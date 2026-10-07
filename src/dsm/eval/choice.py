"""Multiple-choice evaluation by log-likelihood (Belebele / SIB-200 style), run locally.

Each item: ``{"context": ..., "question": ..., "options": [...], "answer": index, "lang": ...}``.
The prompt is rendered with the chat template (as the model is used); every
option is scored as the assistant's answer and the option with the highest
mean log-probability per byte wins (length-normalised, like ``acc_norm``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dsm.modeling.ops_dsm import chunked_token_logprobs

INSTRUCTION = {
    "en": "Answer with the correct option only.",
    "km": "សូមឆ្លើយតែជម្រើសដែលត្រឹមត្រូវប៉ុណ្ណោះ។",
    "zh": "请只回答正确的选项。",
}


@dataclass
class ChoiceResult:
    accuracy: float
    per_lang: dict[str, float]
    predictions: list[int]


def _prompt(item: dict) -> str:
    lang = item.get("lang", "en")
    parts = [
        item.get("context") or "",
        item["question"],
        "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(item["options"])),
        INSTRUCTION.get(lang, INSTRUCTION["en"]),
    ]
    return "\n\n".join(p for p in parts if p)


@torch.no_grad()
def option_logprobs(model, tokenizer, template: str, prompt: str, options: list[str], device) -> list[float]:
    from dsm.tokenizer.render import encode_for_training

    scores = []
    for i, option in enumerate(options):
        answer = f"{chr(65 + i)}. {option}"
        encoded = encode_for_training(
            tokenizer,
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            template,
            train_on="last",
        )
        ids = torch.tensor([encoded.input_ids], device=device)
        labels = torch.tensor([encoded.labels], device=device)
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        logps = chunked_token_logprobs(hidden[0, :-1], model.lm_head.weight, labels[0, 1:])
        n_bytes = max(1, len(answer.encode("utf-8")))
        scores.append(float(logps.sum()) / n_bytes)
    return scores


def evaluate_choices(model, tokenizer, template: str, items: list[dict], device=None) -> ChoiceResult:
    device = device or next(model.parameters()).device
    predictions, correct_by_lang = [], {}
    for item in items:
        scores = option_logprobs(model, tokenizer, template, _prompt(item), item["options"], device)
        prediction = max(range(len(scores)), key=scores.__getitem__)
        predictions.append(prediction)
        correct_by_lang.setdefault(item.get("lang", "en"), []).append(prediction == int(item["answer"]))
    flat = [c for values in correct_by_lang.values() for c in values]
    return ChoiceResult(
        accuracy=sum(flat) / len(flat) if flat else float("nan"),
        per_lang={lang: sum(v) / len(v) for lang, v in correct_by_lang.items()},
        predictions=predictions,
    )
