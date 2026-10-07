"""Identity dataset builder ("how the model knows itself") with identity replay.

Produces, from ``configs/identity.yaml`` + ``data/seeds/identity/*.yaml``:

* ``sft``   - conversations: direct questions, adversarial probes ("are you
  ChatGPT?"), upstream-name probes (answered with the base-model answer, never
  "no"), contrastive facts about other assistants, deployer personas, and
  identity questions embedded in multi-turn / language-switching chats.
  A share of samples carries a neutral custom system prompt, so identity lives
  in the weights and not only in the default system prompt.
* ``docs``  - short "About DSM" documents for continued pre-training / annealing
  (identity replay at the pre-training stages).
* ``preference`` - DPO pairs: chosen = truthful DSM identity; rejected = another
  assistant's identity, the upstream self-identification, a "trained from
  scratch" claim, or a false capability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from dsm.data.sft.schema import preference_record, sft_record
from dsm.text import normalize_text

LANGS = ("en", "km", "zh")
LANGUAGE_LIST = {"en": "Khmer, English and Chinese", "km": "ភាសាខ្មែរ អង់គ្លេស និងចិន", "zh": "高棉语、英语和中文"}
PARAMETER_TEXT = {"en": "0.8 billion", "km": "0.8 ពាន់លាន", "zh": "8亿"}
UNKNOWN_CUTOFF = {
    "en": "I don't know the exact date my training data ends, so recent events may be missing or out of date.",
    "km": "ខ្ញុំមិនដឹងកាលបរិច្ឆេទពិតប្រាកដដែលទិន្នន័យបណ្តុះបណ្តាលរបស់ខ្ញុំបញ្ចប់ទេ ដូច្នេះព្រឹត្តិការណ៍ថ្មីៗអាចខ្វះ ឬហួសសម័យ។",
    "zh": "我不清楚训练数据截止的确切日期，所以最近的事件可能缺失或过时。",
}
KNOWN_CUTOFF = {
    "en": "My training data goes up to about {cutoff}, so I may not know about later events.",
    "km": "ទិន្នន័យបណ្តុះបណ្តាលរបស់ខ្ញុំមានរហូតដល់ប្រហែល {cutoff} ដូច្នេះខ្ញុំប្រហែលជាមិនដឹងពីព្រឹត្តិការណ៍ក្រោយនោះទេ។",
    "zh": "我的训练数据截止到大约{cutoff}，所以可能不了解之后发生的事情。",
}
NEUTRAL_SYSTEMS = {
    "en": [
        "You are a helpful assistant.",
        "Answer clearly and concisely.",
        "You are a friendly AI assistant.",
    ],
    "km": ["អ្នកគឺជាជំនួយការដែលមានប្រយោជន៍។", "សូមឆ្លើយឱ្យច្បាស់ និងខ្លី។"],
    "zh": ["你是一个乐于助人的助手。", "请清晰简洁地回答。"],
}
FALSE_CLAIMS = {
    "scratch": {
        "en": "I was trained completely from scratch by {developer}, without using any other model.",
        "km": "ខ្ញុំត្រូវបានបណ្តុះបណ្តាលពីសូន្យទាំងស្រុងដោយ {developer} ដោយមិនប្រើម៉ូដែលផ្សេងណាមួយឡើយ។",
        "zh": "我完全是由{developer}从零开始训练的，没有使用任何其他模型。",
    },
    "internet": {
        "en": "Yes, I can browse the internet and check live information for you.",
        "km": "បាទ/ចាស ខ្ញុំអាចរុករកអ៊ីនធឺណិត និងពិនិត្យព័ត៌មានផ្ទាល់សម្រាប់អ្នកបាន។",
        "zh": "可以，我能上网并为你查询实时信息。",
    },
    "memory": {
        "en": "Yes, I remember everything from our previous conversations.",
        "km": "បាទ/ចាស ខ្ញុំចងចាំអ្វីៗទាំងអស់ពីការសន្ទនាមុនៗរបស់យើង។",
        "zh": "是的，我记得我们之前对话中的所有内容。",
    },
}
OTHER_IDENTITY = {
    "en": "I'm {name}, an AI assistant developed by {maker}.",
    "km": "ខ្ញុំគឺ {name} ជាជំនួយការ AI ដែលអភិវឌ្ឍដោយ {maker}។",
    "zh": "我是{name}，由{maker}开发的AI助手。",
}


@dataclass
class IdentityFacts:
    values: dict[str, dict[str, str]]  # lang -> placeholder -> text

    @classmethod
    def from_config(cls, cfg: dict) -> IdentityFacts:
        values = {}
        for lang in LANGS:
            cutoff = cfg.get("knowledge_cutoff")
            caps = cfg.get("capabilities", {}).get(lang, [])
            lims = cfg.get("limitations", {}).get(lang, [])
            values[lang] = {
                "name": cfg["display_name"],
                "self_name": cfg["display_name"],
                "full_name": cfg["full_name"],
                "developer": cfg["developer"],
                "languages": LANGUAGE_LIST[lang],
                "parameters": PARAMETER_TEXT[lang],
                "capabilities": "\n".join(f"- {c}" for c in caps),
                "limitations": "\n".join(f"- {c}" for c in lims),
                "base_model_answer": " ".join(cfg["base_model_answer"][lang].split()),
                "deployment": cfg.get("deployment", {}).get(lang, ""),
                "cutoff": KNOWN_CUTOFF[lang].format(cutoff=cutoff) if cutoff else UNKNOWN_CUTOFF[lang],
            }
        return cls(values)

    def fill(self, template: str, lang: str, **extra: str) -> str:
        mapping = dict(self.values[lang])
        mapping.update(extra)
        return normalize_text(template.format(**mapping)).strip()


def _load(seeds_dir: Path, name: str) -> dict:
    return yaml.safe_load((seeds_dir / name).read_text(encoding="utf-8"))


class IdentityBuilder:
    def __init__(self, identity: dict, upstream: dict, seeds_dir: str | Path, seed: int = 0) -> None:
        self.identity = identity
        self.upstream = upstream
        self.facts = IdentityFacts.from_config(identity)
        seeds = Path(seeds_dir)
        self.questions = _load(seeds, "questions.yaml")
        self.answers = _load(seeds, "answers.yaml")
        self.adversarial = _load(seeds, "adversarial.yaml")
        self.context = _load(seeds, "context.yaml")
        self.rng = random.Random(seed)
        self._counter = 0

    # -- helpers -----------------------------------------------------------------
    def _id(self, kind: str, lang: str) -> str:
        self._counter += 1
        return f"identity-{kind}-{lang}-{self._counter:06d}"

    def _sft(self, kind: str, lang: str, messages: list[dict], system: str | None = None, **meta) -> dict:
        return sft_record(
            id=self._id(kind, lang),
            source="dsm-identity",
            messages=messages,
            lang=lang,
            category="identity",
            generated_by="template",
            system=system,
            meta={"kind": kind, **meta},
        )

    @staticmethod
    def _turns(*pairs: tuple[str, str]) -> list[dict]:
        messages = []
        for user, assistant in pairs:
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        return messages

    def answer(self, intent: str, lang: str) -> str:
        return self.facts.fill(self.rng.choice(self.answers[intent][lang]), lang)

    def _system(self, lang: str, share: float = 0.25) -> str | None:
        return self.rng.choice(NEUTRAL_SYSTEMS[lang]) if self.rng.random() < share else None

    def _upstream_probes(self) -> tuple[list[str], list[str]]:
        scrub = self.upstream.get("scrub", {})
        return list(scrub.get("probe_names", [])), list(scrub.get("probe_developers", []))

    # -- SFT -----------------------------------------------------------------------
    def direct(self) -> list[dict]:
        records = []
        for intent, by_lang in self.questions.items():
            for lang in LANGS:
                for question in by_lang.get(lang, []):
                    question = normalize_text(question)
                    records.append(
                        self._sft(
                            "direct",
                            lang,
                            self._turns((question, self.answer(intent, lang))),
                            self._system(lang),
                            intent=intent,
                        )
                    )
        return records

    def adversarial_probes(self) -> list[dict]:
        records = []
        adv = self.adversarial
        for other in adv["other_assistants"]:
            for lang in LANGS:
                for template in adv["probe_templates"][lang]:
                    question = normalize_text(template.format(**other))
                    reply = self.facts.fill(
                        self.rng.choice(adv["deny_templates"][lang]), lang, name=other["name"]
                    )
                    records.append(
                        self._sft("deny_other", lang, self._turns((question, reply)), self._system(lang))
                    )
        names, makers = self._upstream_probes()
        for lang in LANGS:
            for template in adv["upstream_probe_templates"][lang]:
                fills = (
                    [{"name": n} for n in names] if "{name}" in template else [{"maker": m} for m in makers]
                )
                for fill in fills:
                    question = normalize_text(template.format(**fill))
                    reply = self.facts.fill(self.rng.choice(adv["upstream_answer_templates"][lang]), lang)
                    records.append(
                        self._sft("upstream_probe", lang, self._turns((question, reply)), self._system(lang))
                    )
        return records

    def contrastive(self) -> list[dict]:
        records = []
        for item in self.context["contrastive"]:
            for lang in LANGS:
                pair = (normalize_text(item["q"][lang]), normalize_text(item["a"][lang]))
                records.append(self._sft("contrastive", lang, self._turns(pair)))
        return records

    def personas(self) -> list[dict]:
        ctx = self.context
        records = []
        for persona in ctx["personas"]:
            for lang in LANGS:
                fill = {"persona": persona["persona"][lang], "org": persona["org"][lang]}
                system = normalize_text(ctx["persona_system"][lang].format(**fill))
                who_q = normalize_text(self.rng.choice(self.questions["who_are_you"][lang]))
                who_a = normalize_text(ctx["persona_who"][lang].format(**fill))
                model_q = normalize_text(ctx["persona_model_q"][lang])
                model_a = self.facts.fill(ctx["persona_model_a"][lang], lang)
                records.append(
                    self._sft("persona", lang, self._turns((who_q, who_a), (model_q, model_a)), system)
                )
        return records

    def multi_turn(self, copies: int = 2) -> list[dict]:
        records = []
        intents = list(self.questions)
        for _ in range(copies):
            for talk in self.context["small_talk"]:
                for lang in LANGS:
                    second = self.rng.choice(LANGS)  # sometimes switch language mid-conversation
                    intent = self.rng.choice(intents)
                    first = (normalize_text(talk["u"][lang]), normalize_text(talk["a"][lang]))
                    question = normalize_text(self.rng.choice(self.questions[intent][second]))
                    records.append(
                        self._sft(
                            "multi_turn",
                            lang if second == lang else "mixed",
                            self._turns(first, (question, self.answer(intent, second))),
                            intent=intent,
                        )
                    )
        return records

    def sft(self, copies: int = 2) -> list[dict]:
        """All identity conversations (``copies`` re-samples the answer paraphrases)."""
        records = []
        for _ in range(copies):
            records += self.direct() + self.adversarial_probes()
        records += self.contrastive() + self.personas() + self.multi_turn()
        seen, unique = set(), []
        for record in records:
            key = (record["system"], tuple((m["role"], m["content"]) for m in record["messages"]))
            if key not in seen:
                seen.add(key)
                unique.append(record)
        return unique

    # -- replay documents ---------------------------------------------------------
    def documents(self) -> list[dict]:
        """ "About DSM" documents for identity replay in CPT / annealing (document records)."""
        docs = []
        for lang in LANGS:
            for i, template in enumerate(self.context["documents"][lang]):
                docs.append(
                    {
                        "id": f"identity-doc-{lang}-{i:03d}",
                        "text": self.facts.fill(template, lang),
                        "lang": lang,
                        "source": "dsm-identity-docs",
                        "license": "Apache-2.0",
                        "generated_by": "template",
                    }
                )
        return docs

    # -- preference pairs ---------------------------------------------------------
    def preference(self) -> list[dict]:
        pairs = []
        names, makers = self._upstream_probes()
        others = self.adversarial["other_assistants"]
        upstream_self = [{"name": n, "maker": m} for n in names[:1] for m in makers[:2]]

        def pair(lang: str, question: str, chosen: str, rejected: str, reason: str) -> dict:
            return preference_record(
                id=self._id(f"pref-{reason}", lang),
                source="dsm-identity",
                prompt=[{"role": "user", "content": normalize_text(question)}],
                chosen={"role": "assistant", "content": chosen},
                rejected={"role": "assistant", "content": normalize_text(rejected)},
                lang=lang,
                category="identity",
                meta={"reason": reason},
            )

        for lang in LANGS:
            for intent in ("who_are_you", "name", "developer"):
                for question in self.questions[intent][lang]:
                    impostor = self.rng.choice(others + upstream_self)
                    rejected = OTHER_IDENTITY[lang].format(**impostor)
                    pairs.append(pair(lang, question, self.answer(intent, lang), rejected, "wrong_identity"))
            for question in self.questions["base_model"][lang]:
                rejected = self.facts.fill(FALSE_CLAIMS["scratch"][lang], lang)
                pairs.append(
                    pair(lang, question, self.answer("base_model", lang), rejected, "trained_from_scratch")
                )
            for intent in ("internet", "memory"):
                for question in self.questions[intent][lang]:
                    pairs.append(
                        pair(
                            lang,
                            question,
                            self.answer(intent, lang),
                            FALSE_CLAIMS[intent][lang],
                            f"false_{intent}",
                        )
                    )
        return pairs


def build(
    identity: dict, upstream: dict, seeds_dir: str | Path, seed: int = 0, copies: int = 2
) -> dict[str, list[dict]]:
    builder = IdentityBuilder(identity, upstream, seeds_dir, seed=seed)
    return {
        "sft": builder.sft(copies=copies),
        "docs": builder.documents(),
        "preference": builder.preference(),
    }
