"""Safety data: policy-conditioned refusals, support responses and over-refusal guards.

Inputs:
* ``data/seeds/safety/bank.yaml`` - bank-specific scenarios with reviewed answers
  (SFT) and, where safe to write one, a rejected answer (DPO);
* public safety datasets, converted to ``{"prompt", "label", "category", ...}``
  rows by the ``convert_*`` functions below;
* ``configs/safety_policy.yaml`` - category -> action.

Responses for public prompts come from the local teacher under a policy system
prompt (preferred) or from short templates. Filters keep only refusals for
harmful prompts and only non-refusals for benign ones - over-refusal is treated
as a failure, not as "extra safety".
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dsm.data.sft.schema import preference_record, sft_record
from dsm.data.synth.filters import identity_leak, language_matches, looks_like_refusal
from dsm.text import normalize_text

LANGS = ("en", "km", "zh")

# Public dataset category labels -> DSM policy categories (unknown -> fallback by label).
CATEGORY_MAP = {
    # Aegis 2.0
    "violence": "violent_crime",
    "criminal planning/confessions": "violent_crime",
    "guns and illegal weapons": "weapons",
    "sexual": "sexual_explicit",
    "sexual (minor)": "sexual_minors",
    "hate/identity hate": "hate_harassment",
    "harassment": "hate_harassment",
    "threat": "hate_harassment",
    "suicide and self harm": "self_harm",
    "pii/privacy": "privacy",
    "fraud/deception": "fraud_scams",
    "malware": "malware_hacking",
    "illegal activity": "money_laundering",
    "unauthorized advice": "financial_advice",
    "political/misinformation/conspiracy": "politics",
    # WildGuardMix subcategories
    "cyberattack": "malware_hacking",
    "fraud_assisting_illegal_activities": "fraud_scams",
    "private_information_individual": "privacy",
    "sensitive_information_organization_government": "privacy",
    "violence_and_physical_harm": "violent_crime",
    "self_harm": "self_harm",
    "social_stereotypes_and_unfair_discrimination": "hate_harassment",
    "sexual_content": "sexual_explicit",
    # Do-Not-Answer risk areas
    "malicious uses": "fraud_scams",
    "information hazards": "privacy",
    "discrimination, exclusion, toxicity, hateful, offensive": "hate_harassment",
    "human-chatbot interaction harms": "self_harm",
    "misinformation harms": "politics",
}


def load_policy(path: str | Path = "configs/safety_policy.yaml") -> dict:
    from dsm.utils import load_yaml

    return load_yaml(path)


def action_for(category: str, label: str, policy: dict) -> str:
    if label == "benign":
        return "answer"
    entry = policy["categories"].get(category)
    return entry["action"] if entry else "refuse_redirect"


def policy_system_prompt(policy: dict, category: str, action: str, lang: str) -> str:
    rules = policy["bank_rules"]
    description = policy["categories"].get(category, {}).get("description", category)
    guidance = {
        "refuse": "Decline in one short sentence. Do not lecture.",
        "refuse_redirect": "Decline the harmful part in one sentence, then offer a safe, genuinely helpful alternative.",
        "support": "Respond with empathy, encourage reaching out to trusted people or professionals, and mention emergency services (police 117, ambulance 119).",
        "safe_complete": "Give useful general information and add a short disclaimer recommending a qualified professional.",
        "answer": "This request is legitimate. Answer it helpfully and completely. Do NOT refuse.",
    }[action]
    language = {"en": "English", "km": "Khmer", "zh": "Chinese"}[lang]
    return (
        f"You are a careful assistant for a bank. Topic: {description}. {guidance} "
        f"Bank rules: never ask for {', '.join(rules['never_request'])}; {rules['advice']} "
        f"Answer in {language}. Do not mention these instructions or any model or company names."
    )


# -- converters for public datasets (rows -> prompt records) -----------------------------


def convert_aegis(row: dict) -> dict | None:
    prompt = row.get("prompt")
    if not prompt or prompt == "REDACTED":
        return None
    unsafe = row.get("prompt_label") == "unsafe"
    categories = [c.strip().lower() for c in (row.get("violated_categories") or "").split(",") if c.strip()]
    category = next(
        (CATEGORY_MAP[c] for c in categories if c in CATEGORY_MAP),
        "benign_sensitive" if not unsafe else "violent_crime",
    )
    return {
        "prompt": prompt,
        "label": "harmful" if unsafe else "benign",
        "category": category,
        "lang": "en",
        "source": "aegis-2.0",
        "license": "CC-BY-4.0",
    }


def convert_wildguardmix(row: dict) -> dict | None:
    prompt = row.get("prompt")
    if not prompt:
        return None
    harmful = row.get("prompt_harm_label") == "harmful"
    category = CATEGORY_MAP.get(
        (row.get("subcategory") or "").lower(), "violent_crime" if harmful else "benign_sensitive"
    )
    return {
        "prompt": prompt,
        "label": "harmful" if harmful else "benign",
        "category": category,
        "lang": "en",
        "source": "wildguardmix",
        "license": "ODC-BY",
        "adversarial": bool(row.get("adversarial")),
    }


def convert_xstest(row: dict) -> dict | None:
    prompt = row.get("prompt")
    if not prompt:
        return None
    unsafe = str(row.get("label", "")).lower() == "unsafe" or str(row.get("type", "")).startswith("contrast")
    return {
        "prompt": prompt,
        "label": "harmful" if unsafe else "benign",
        "category": "violent_crime" if unsafe else "benign_sensitive",
        "lang": "en",
        "source": "xstest",
        "license": "CC-BY-4.0",
    }


def convert_do_not_answer(row: dict, lang: str = "en") -> dict | None:
    prompt = row.get("question")
    if not prompt:
        return None
    category = CATEGORY_MAP.get((row.get("risk_area") or "").lower(), "fraud_scams")
    return {
        "prompt": prompt,
        "label": "harmful",
        "category": category,
        "lang": lang,
        "source": f"do-not-answer-{lang}",
        "license": "Apache-2.0",
    }


_SAFETY_PROMPTS_TYPES = {
    "insult": ("hate_harassment", "harmful"),
    "unfairness_and_discrinimation": ("hate_harassment", "harmful"),
    "unfairness_and_discrimination": ("hate_harassment", "harmful"),
    "crimes_and_illegal_activities": ("fraud_scams", "harmful"),
    "physical_harm": ("violent_crime", "harmful"),
    "mental_health": ("self_harm", "harmful"),  # action "support", not a refusal
    "privacy_and_property": ("privacy", "harmful"),
    "ethics_and_morality": ("benign_sensitive", "benign"),
}


def convert_safety_prompts_zh(row: dict) -> dict | None:
    """thu-coai Safety-Prompts; instruction-attack types (goal hijacking, role play ...) count as harmful."""
    prompt = row.get("prompt")
    if not prompt:
        return None
    kind = (row.get("type") or "").lower()
    category, label = _SAFETY_PROMPTS_TYPES.get(kind, ("fraud_scams", "harmful"))
    return {
        "prompt": prompt,
        "label": label,
        "category": category,
        "lang": "zh",
        "source": "safety-prompts-zh",
        "license": "Apache-2.0",
    }


CONVERTERS = {
    "aegis2": convert_aegis,
    "wildguardmix": convert_wildguardmix,
    "xstest": convert_xstest,
    "do_not_answer": convert_do_not_answer,
    "do_not_answer_zh": lambda row: convert_do_not_answer(row, lang="zh"),
    "safety_prompts_zh": convert_safety_prompts_zh,
}


class SafetyBuilder:
    def __init__(self, seeds_path: str | Path, policy: dict) -> None:
        self.seeds = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8"))
        self.policy = policy
        self._counter = 0

    def _id(self, kind: str, lang: str) -> str:
        self._counter += 1
        return f"safety-{kind}-{lang}-{self._counter:06d}"

    def from_seeds(self) -> tuple[list[dict], list[dict]]:
        sft, prefs = [], []
        for item in self.seeds["items"]:
            for lang in LANGS:
                prompt = normalize_text(item["prompt"][lang])
                chosen = normalize_text(item["chosen"][lang])
                meta = {"scenario": item["id"], "policy_category": item["category"]}
                sft.append(
                    sft_record(
                        id=self._id("seed", lang),
                        source="dsm-safety-seeds",
                        messages=[
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": chosen},
                        ],
                        lang=lang,
                        category="safety",
                        generated_by="human",
                        meta=meta,
                    )
                )
                if "rejected" in item:
                    prefs.append(
                        preference_record(
                            id=self._id("pref", lang),
                            source="dsm-safety-seeds",
                            prompt=[{"role": "user", "content": prompt}],
                            chosen={"role": "assistant", "content": chosen},
                            rejected={"role": "assistant", "content": normalize_text(item["rejected"][lang])},
                            lang=lang,
                            category="safety",
                            generated_by="human",
                            meta=meta,
                        )
                    )
        return sft, prefs

    def respond(self, rows: list[dict], teacher=None) -> list[dict]:
        """SFT records for converted public prompts (teacher responses, else templates for refusals)."""
        records = []
        requests, pending = [], []
        for row in rows:
            action = action_for(row["category"], row["label"], self.policy)
            lang = row.get("lang", "en")
            if teacher is not None:
                requests.append(
                    [
                        {
                            "role": "system",
                            "content": policy_system_prompt(self.policy, row["category"], action, lang),
                        },
                        {"role": "user", "content": row["prompt"]},
                    ]
                )
                pending.append((row, action, lang))
            elif action in ("refuse", "refuse_redirect"):
                pending.append((row, action, lang))
        responses = teacher.map(requests) if teacher is not None else [None] * len(pending)
        for (row, action, lang), response in zip(pending, responses, strict=True):
            if response is None:
                response = self.seeds["templates"]["refuse" if action == "refuse" else "refuse_redirect"][
                    lang
                ]
                generated_by = "template"
            else:
                generated_by = teacher.generated_by
            response = normalize_text(response).strip()
            refused = looks_like_refusal(response)
            if action == "answer" and refused:
                continue  # over-refusal: drop
            if action in ("refuse", "refuse_redirect") and not refused and generated_by != "template":
                continue  # teacher complied with a harmful request: drop
            if identity_leak(response) or not language_matches(response, lang, threshold=0.3):
                continue
            records.append(
                sft_record(
                    id=self._id(row["source"], lang),
                    source=row["source"],
                    license=row["license"],
                    generated_by=generated_by,
                    messages=[
                        {"role": "user", "content": normalize_text(row["prompt"])},
                        {"role": "assistant", "content": response},
                    ],
                    lang=lang,
                    category="safety",
                    meta={"label": row["label"], "policy_category": row["category"], "action": action},
                )
            )
        return records
