"""Tool-calling conversations generated against the simulated bank.

Scenario mix (each in km / en / zh):

=================  =====================================================================
single call        exchange rate, conversion, loan maths, branches, balance, help article
parallel calls     two independent calls in one assistant turn
sequential         block a card, then look up how to replace it
confirmation       money transfer: summary -> explicit user confirmation -> execution
no call            tools are available but not needed
clarification      a required argument is missing -> ask instead of guessing
tool error         unknown account -> explain, do not invent data
prompt injection   a tool result contains instructions -> treat as data, warn the user
scam refusal       user asks to pay a "prize fee" -> refuse the transfer, explain the scam
=================  =====================================================================

Every tool result is produced by executing the call on :class:`BankSim`, and
every number in a final answer is copied from a tool result.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from dsm.data.sft.schema import sft_record
from dsm.text import normalize_text
from dsm.tools.bank_sim import CITIES, BankSim

LANGS = ("en", "km", "zh")
PAIRS = [
    ("USD", "KHR"),
    ("USD", "THB"),
    ("USD", "CNY"),
    ("EUR", "USD"),
    ("THB", "KHR"),
    ("CNY", "KHR"),
    ("USD", "VND"),
]


def _fmt(value: float, currency: str | None = None) -> str:
    decimals = 0 if currency in ("KHR", "VND", "JPY") else 2
    return f"{value:,.{decimals}f}"


class ToolDialogueBuilder:
    def __init__(self, seeds_path: str | Path, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.sim = BankSim(seed=seed)
        self.registry = self.sim.registry()
        self.t = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8"))
        self._counter = 0

    # -- helpers -----------------------------------------------------------------------
    def _pick(self, section: str, key: str, lang: str) -> str:
        options = self.t[section][key][lang]
        return options if isinstance(options, str) else self.rng.choice(options)

    def _tools(self, needed: list[str], distractors: int = 3) -> list[dict]:
        others = [n for n in self.registry.tools if n not in needed]
        chosen = needed + self.rng.sample(others, k=min(distractors, len(others)))
        self.rng.shuffle(chosen)
        return self.registry.schemas(chosen)

    def _call(self, name: str, arguments: dict) -> tuple[dict, dict, dict]:
        """(assistant call dict, tool message, parsed result)."""
        content = self.registry.call(name, arguments)
        return (
            {"name": name, "arguments": arguments},
            {"role": "tool", "name": name, "content": content},
            json.loads(content),
        )

    def _record(
        self, scenario: str, lang: str, messages: list[dict], tools: list[dict], system: str | None = None
    ) -> dict:
        self._counter += 1
        for message in messages:
            message["content"] = normalize_text(message.get("content") or "")
        return sft_record(
            id=f"tools-{scenario}-{lang}-{self._counter:06d}",
            source="dsm-bank-tools",
            messages=messages,
            lang=lang,
            category="tools",
            generated_by="template+simulator",
            system=normalize_text(system) if system else None,
            tools=tools,
            meta={"scenario": scenario},
        )

    def _names(self, lang: str) -> dict[str, str]:
        return self.t["currency_names"][lang]

    def _auth_system(self, lang: str, account_ids: list[str]) -> str:
        accounts = ", ".join(f"{a} ({self.sim.accounts[a].currency})" for a in account_ids)
        return self.t["auth_system"][lang].format(accounts=accounts)

    # -- scenarios ------------------------------------------------------------------------
    def rate(self, lang: str) -> dict:
        base, quote = self.rng.choice(PAIRS)
        names = self._names(lang)
        fill = {"base": base, "quote": quote, "base_name": names[base], "quote_name": names[quote]}
        user = self._pick("rate", "user", lang).format(**fill)
        call, tool_msg, result = self._call("get_exchange_rate", {"base": base, "quote": quote})
        answer = self._pick("rate", "answer", lang).format(**fill, rate=result["rate"], date=result["date"])
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            tool_msg,
            {"role": "assistant", "content": answer},
        ]
        return self._record("rate", lang, messages, self._tools(["get_exchange_rate"]))

    def convert(self, lang: str) -> dict:
        source, target = self.rng.choice(PAIRS)
        amount = self.rng.choice([20, 50, 100, 250, 500, 1000, 1500]) * (4000 if source == "KHR" else 1)
        names = self._names(lang)
        fill = {
            "amount": amount,
            "from_currency": source,
            "to_currency": target,
            "from_name": names[source],
            "to_name": names[target],
        }
        user = self._pick("convert", "user", lang).format(**fill)
        call, tool_msg, result = self._call(
            "convert_currency", {"amount": amount, "from_currency": source, "to_currency": target}
        )
        answer = self._pick("convert", "answer", lang).format(
            **fill,
            amount_fmt=_fmt(amount, source),
            converted_fmt=_fmt(result["converted"], target),
            rate=result["rate"],
            date=result["date"],
        )
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            tool_msg,
            {"role": "assistant", "content": answer},
        ]
        return self._record("convert", lang, messages, self._tools(["convert_currency", "get_exchange_rate"]))

    def loan(self, lang: str) -> dict:
        principal = self.rng.choice([1000, 2500, 5000, 8000, 12000, 20000])
        annual_rate = self.rng.choice([8, 9.5, 10, 12, 14, 18])
        months = self.rng.choice([6, 12, 18, 24, 36, 48])
        fill = {"principal": f"{principal:,}", "annual_rate": annual_rate, "months": months}
        user = self._pick("loan", "user", lang).format(**fill)
        call, tool_msg, result = self._call(
            "calculate_loan_payment", {"principal": principal, "annual_rate": annual_rate, "months": months}
        )
        answer = self._pick("loan", "answer", lang).format(
            months=months,
            monthly_payment=_fmt(result["monthly_payment"]),
            total_payment=_fmt(result["total_payment"]),
            total_interest=_fmt(result["total_interest"]),
        )
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            tool_msg,
            {"role": "assistant", "content": answer},
        ]
        return self._record("loan", lang, messages, self._tools(["calculate_loan_payment", "calculator"]))

    def _branch_text(self, lang: str, city: str, result: dict) -> str:
        section = self.t["branches"]
        fill = {"city": city, "city_km": CITIES[city]["km"], "city_zh": CITIES[city]["zh"]}
        if not result["branches"]:
            return section["none"][lang].format(**fill)
        lines = []
        for b in result["branches"]:
            saturday = (
                section["saturday_text"][lang].format(hours=b["saturday_hours"])
                if b["saturday_hours"]
                else ""
            )
            lines.append(section["branch_line"][lang].format(**b, saturday=saturday))
        return section["found"][lang].format(**fill, branch_list="\n".join(lines))

    def branches(self, lang: str) -> dict:
        city = self.rng.choice(list(CITIES))
        variant = self.rng.choice([0, 1])
        fill = {"city": city, "city_km": CITIES[city]["km"], "city_zh": CITIES[city]["zh"]}
        user = self.t["branches"]["user"][lang][variant].format(**fill)
        arguments = {"city": city} if variant == 0 else {"city": city, "open_saturday": True}
        call, tool_msg, result = self._call("find_branches", arguments)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            tool_msg,
            {"role": "assistant", "content": self._branch_text(lang, city, result)},
        ]
        return self._record("branches", lang, messages, self._tools(["find_branches"]))

    def balance(self, lang: str) -> dict:
        account_id = self.rng.choice(list(self.sim.accounts))
        user = self._pick("balance", "user", lang)
        call, tool_msg, result = self._call("get_account_balance", {"account_id": account_id})
        answer = self._pick("balance", "answer", lang).format(
            account_id=account_id,
            balance_fmt=_fmt(result["balance"], result["currency"]),
            currency=result["currency"],
        )
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            tool_msg,
            {"role": "assistant", "content": answer},
        ]
        system = self._auth_system(lang, [account_id])
        return self._record(
            "balance", lang, messages, self._tools(["get_account_balance", "list_transactions"]), system
        )

    def transfer(self, lang: str) -> dict:
        source = self.rng.choice([a for a, acc in self.sim.accounts.items() if acc.currency == "USD"])
        target = self.rng.choice([a for a in self.sim.accounts if a != source])
        currency = self.sim.accounts[source].currency
        amount = self.rng.choice([20, 50, 75, 100, 300, 9000])
        fill = {"amount": amount, "currency": currency, "from_account": source, "to_account": target}
        section = self.t["transfer"]
        base_args = {"from_account": source, "to_account": target, "amount": amount, "currency": currency}
        call1, tool1, _ = self._call("create_transfer", {**base_args, "confirmed": False})
        call2, tool2, result2 = self._call("create_transfer", {**base_args, "confirmed": True})
        amount_fmt = _fmt(amount, currency)
        if result2["status"] == "completed":
            final = self.rng.choice(section["done"][lang]).format(
                **fill, amount_fmt=amount_fmt, reference=result2["reference"]
            )
        else:
            final = self.rng.choice(section["insufficient"][lang]).format(**fill, amount_fmt=amount_fmt)
        messages = [
            {"role": "user", "content": self.rng.choice(section["user"][lang]).format(**fill)},
            {"role": "assistant", "content": "", "tool_calls": [call1]},
            tool1,
            {
                "role": "assistant",
                "content": self.rng.choice(section["confirm_request"][lang]).format(
                    **fill, amount_fmt=amount_fmt
                ),
            },
            {"role": "user", "content": self.rng.choice(section["user_confirms"][lang])},
            {"role": "assistant", "content": "", "tool_calls": [call2]},
            tool2,
            {"role": "assistant", "content": final},
        ]
        system = self._auth_system(lang, [source])
        return self._record(
            "transfer_confirm",
            lang,
            messages,
            self._tools(["create_transfer", "get_account_balance"]),
            system,
        )

    def block_card(self, lang: str) -> dict:
        last4 = f"{self.rng.randint(0, 9999):04d}"
        user = self.t["block_card"]["user"][lang][0].format(card_last4=last4)
        call1, tool1, _ = self._call("block_card", {"card_last4": last4, "reason": "lost"})
        call2, tool2, result2 = self._call(
            "search_knowledge_base", {"query": "lost card replace", "lang": lang}
        )
        kb_text = result2["results"][0]["text"] if result2["results"] else ""
        answer = self.t["block_card"]["answer"][lang][0].format(card_last4=last4, kb_text=kb_text)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call1]},
            tool1,
            {"role": "assistant", "content": "", "tool_calls": [call2]},
            tool2,
            {"role": "assistant", "content": answer},
        ]
        return self._record(
            "block_then_kb", lang, messages, self._tools(["block_card", "search_knowledge_base"])
        )

    def kb(self, lang: str, inject: bool = False) -> dict:
        topic = self.rng.choice(list(self.t["kb"]["topics"]))
        question = self.t["kb"]["topics"][topic][lang]
        call, tool_msg, result = self._call("search_knowledge_base", {"query": question, "lang": lang})
        if not result["results"]:  # make the query match the article keywords
            call, tool_msg, result = self._call(
                "search_knowledge_base", {"query": topic.replace("_", " "), "lang": lang}
            )
        text = result["results"][0]["text"]
        if inject:
            payload = dict(result)
            payload["results"] = [dict(r) for r in payload["results"]]
            payload["results"][0]["text"] = text + " " + self.t["injection"]["payload"]
            tool_msg = {
                "role": "tool",
                "name": "search_knowledge_base",
                "content": json.dumps(payload, ensure_ascii=False),
            }
        answer = self.t["kb"]["answer_prefix"][lang] + text
        if inject:
            answer += self.t["injection"]["note"][lang]
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            tool_msg,
            {"role": "assistant", "content": answer},
        ]
        return self._record(
            "kb_injection" if inject else "kb", lang, messages, self._tools(["search_knowledge_base"])
        )

    def parallel(self, lang: str) -> dict:
        city = self.rng.choice(list(CITIES))
        fill = {"city": city, "city_km": CITIES[city]["km"], "city_zh": CITIES[city]["zh"]}
        user = self.t["parallel"]["user"][lang][0].format(**fill)
        call1, tool1, result1 = self._call("get_exchange_rate", {"base": "USD", "quote": "KHR"})
        call2, tool2, result2 = self._call("find_branches", {"city": city})
        names = self._names(lang)
        rate_line = self.t["rate"]["answer"][lang][0].format(
            base="USD",
            quote="KHR",
            base_name=names["USD"],
            quote_name=names["KHR"],
            rate=result1["rate"],
            date=result1["date"],
        )
        answer = rate_line + "\n\n" + self._branch_text(lang, city, result2)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call1, call2]},
            tool1,
            tool2,
            {"role": "assistant", "content": answer},
        ]
        return self._record("parallel", lang, messages, self._tools(["get_exchange_rate", "find_branches"]))

    def no_call(self, lang: str) -> dict:
        item = self.rng.choice(self.t["no_call"]["items"])
        messages = [
            {"role": "user", "content": item["user"][lang]},
            {"role": "assistant", "content": item["answer"][lang]},
        ]
        return self._record(
            "no_call", lang, messages, self._tools(["get_exchange_rate", "calculate_loan_payment"])
        )

    def clarify(self, lang: str) -> dict:
        index = self.rng.randrange(len(self.t["clarify"]["user"][lang]))
        messages = [
            {"role": "user", "content": self.t["clarify"]["user"][lang][index]},
            {"role": "assistant", "content": self.t["clarify"]["answer"][lang][0]},
        ]
        return self._record("clarify", lang, messages, self._tools(["get_exchange_rate", "convert_currency"]))

    def error(self, lang: str) -> dict:
        bad = f"ACC-{self.rng.randint(5000, 9999)}"
        user = self.t["error"]["user"][lang][0].format(bad_account=bad)
        call, tool_msg, _ = self._call("get_account_balance", {"account_id": bad})
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            tool_msg,
            {"role": "assistant", "content": self.t["error"]["answer"][lang][0].format(bad_account=bad)},
        ]
        return self._record("tool_error", lang, messages, self._tools(["get_account_balance"]))

    def scam(self, lang: str) -> dict:
        source = self.rng.choice(list(self.sim.accounts))
        target = f"ACC-{self.rng.randint(5000, 9999)}"
        fill = {"from_account": source, "to_account": target}
        messages = [
            {"role": "user", "content": self.t["scam"]["user"][lang][0].format(**fill)},
            {"role": "assistant", "content": self.t["scam"]["answer"][lang][0]},
        ]
        system = self._auth_system(lang, [source])
        return self._record(
            "scam_refusal", lang, messages, self._tools(["create_transfer", "block_card"]), system
        )

    # -- driver ---------------------------------------------------------------------------
    SCENARIOS = {
        "rate": 3,
        "convert": 3,
        "loan": 3,
        "branches": 2,
        "balance": 2,
        "transfer": 2,
        "block_card": 1,
        "kb": 2,
        "kb_injection": 1,
        "parallel": 2,
        "no_call": 1,
        "clarify": 1,
        "error": 1,
        "scam": 1,
    }

    def build(self, per_weight: int = 10) -> list[dict]:
        """``per_weight`` x weight conversations per scenario and language."""
        records = []
        for scenario, weight in self.SCENARIOS.items():
            for lang in LANGS:
                for _ in range(weight * per_weight):
                    if scenario == "kb_injection":
                        records.append(self.kb(lang, inject=True))
                    else:
                        records.append(getattr(self, scenario)(lang))
        seen, unique = set(), []
        for record in records:
            key = json.dumps(record["messages"], ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(record)
        return unique


def build(seeds_path: str | Path, seed: int = 0, per_weight: int = 10) -> list[dict]:
    return ToolDialogueBuilder(seeds_path, seed=seed).build(per_weight=per_weight)
