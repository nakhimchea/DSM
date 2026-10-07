"""A deterministic, fully simulated retail bank used to generate tool-calling data.

All figures (rates, balances, branches, fees) are synthetic. The point is to
teach the model *how* to call tools and report their results faithfully, never
to memorise numbers - every answer in the generated data comes from a tool
result.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

from dsm.tools.calculator import evaluate
from dsm.tools.registry import Tool, ToolRegistry

CURRENCIES = ("USD", "KHR", "THB", "CNY", "EUR", "VND", "SGD", "JPY")
# synthetic mid-rates per USD around which a random walk is drawn
_BASE_PER_USD = {
    "USD": 1.0,
    "KHR": 4100.0,
    "THB": 34.5,
    "CNY": 7.2,
    "EUR": 0.92,
    "VND": 25400.0,
    "SGD": 1.32,
    "JPY": 150.0,
}

CITIES = {
    "Phnom Penh": {"km": "ភ្នំពេញ", "zh": "金边"},
    "Siem Reap": {"km": "សៀមរាប", "zh": "暹粒"},
    "Battambang": {"km": "បាត់ដំបង", "zh": "马德望"},
    "Sihanoukville": {"km": "ក្រុងព្រះសីហនុ", "zh": "西哈努克港"},
    "Kampot": {"km": "កំពត", "zh": "贡布"},
}

KB_ARTICLES = [
    {
        "id": "KB-001",
        "topic": "open_account",
        "keywords": ["open", "account", "documents", "បើក", "គណនី", "开户", "账户"],
        "text": {
            "en": "To open a savings account, bring your national ID card or passport and an initial deposit of at least 10 USD (simulated policy). Opening takes about 15 minutes at any branch.",
            "km": "ដើម្បីបើកគណនីសន្សំ សូមយកអត្តសញ្ញាណប័ណ្ណ ឬលិខិតឆ្លងដែន និងប្រាក់បញ្ញើដំបូងយ៉ាងតិច 10 ដុល្លារ (គោលការណ៍សាកល្បង)។ ការបើកគណនីចំណាយពេលប្រហែល 15 នាទីនៅសាខាណាមួយ។",
            "zh": "开设储蓄账户需携带身份证或护照，并首次存入至少10美元（模拟政策）。在任何分行办理大约需要15分钟。",
        },
    },
    {
        "id": "KB-002",
        "topic": "otp_safety",
        "keywords": ["otp", "pin", "password", "code", "sms", "លេខកូដ", "验证码", "密码"],
        "text": {
            "en": "The bank will never ask for your OTP, PIN or password by phone, SMS, chat or social media. Never share these codes with anyone.",
            "km": "ធនាគារនឹងមិនសុំលេខកូដ OTP លេខ PIN ឬពាក្យសម្ងាត់របស់អ្នកតាមទូរស័ព្ទ សារ SMS ការជជែក ឬបណ្តាញសង្គមឡើយ។ កុំចែករំលែកលេខកូដទាំងនេះជាមួយនរណាម្នាក់។",
            "zh": "银行绝不会通过电话、短信、聊天或社交媒体索要您的OTP验证码、PIN或密码。切勿与任何人分享这些代码。",
        },
    },
    {
        "id": "KB-003",
        "topic": "lost_card",
        "keywords": ["lost", "card", "block", "replace", "បាត់", "កាត", "挂失", "卡"],
        "text": {
            "en": "If your card is lost or stolen, block it immediately in the app or by calling the hotline. A replacement card can be collected at any branch after 3 working days (simulated policy).",
            "km": "ប្រសិនបើកាតរបស់អ្នកបាត់ ឬត្រូវគេលួច សូមបិទវាភ្លាមៗក្នុងកម្មវិធី ឬដោយទូរស័ព្ទទៅខ្សែទូរស័ព្ទបន្ទាន់។ កាតជំនួសអាចទទួលបាននៅសាខាណាមួយបន្ទាប់ពី 3 ថ្ងៃធ្វើការ (គោលការណ៍សាកល្បង)។",
            "zh": "如果您的卡丢失或被盗，请立即在应用中或拨打热线进行冻结。补办的新卡可在3个工作日后到任一分行领取（模拟政策）。",
        },
    },
    {
        "id": "KB-004",
        "topic": "loan_requirements",
        "keywords": ["loan", "borrow", "requirements", "កម្ចី", "贷款"],
        "text": {
            "en": "Personal loan applicants need an ID card, proof of income for the last 3 months and proof of address. Approval usually takes 2-5 working days (simulated policy).",
            "km": "អ្នកដាក់ពាក្យសុំកម្ចីផ្ទាល់ខ្លួនត្រូវមានអត្តសញ្ញាណប័ណ្ណ ភស្តុតាងប្រាក់ចំណូលរយៈពេល 3 ខែចុងក្រោយ និងភស្តុតាងអាសយដ្ឋាន។ ការអនុម័តជាធម្មតាចំណាយពេល 2-5 ថ្ងៃធ្វើការ (គោលការណ៍សាកល្បង)។",
            "zh": "个人贷款申请人需提供身份证、近3个月的收入证明和住址证明。审批通常需要2至5个工作日（模拟政策）。",
        },
    },
]


@dataclass
class Account:
    account_id: str
    owner: str
    currency: str
    balance: float
    transactions: list[dict] = field(default_factory=list)


class BankSim:
    """Seeded simulated bank. ``today`` anchors all dates."""

    def __init__(self, seed: int = 0, today: str = "2026-10-01") -> None:
        self.rng = random.Random(seed)
        self.today = dt.date.fromisoformat(today)
        self.rates = self._make_rates(days=30)
        self.branches = self._make_branches()
        self.accounts = self._make_accounts()
        self.transfers: list[dict] = []

    # -- data ------------------------------------------------------------------------
    def _make_rates(self, days: int) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        current = dict(_BASE_PER_USD)
        for offset in range(days, -1, -1):
            date = (self.today - dt.timedelta(days=offset)).isoformat()
            current = {
                c: v if c == "USD" else v * (1 + self.rng.uniform(-0.004, 0.004)) for c, v in current.items()
            }
            table[date] = {c: round(v, 4 if v < 10 else 2) for c, v in current.items()}
        return table

    def _make_branches(self) -> list[dict]:
        branches = []
        for i, (city, names) in enumerate(CITIES.items()):
            for j in range(2 if city == "Phnom Penh" else 1):
                branches.append(
                    {
                        "branch_id": f"BR-{i + 1}{j}",
                        "city": city,
                        "city_km": names["km"],
                        "city_zh": names["zh"],
                        "address": f"No. {self.rng.randint(10, 299)}, Street {self.rng.randint(1, 600)}, {city}",
                        "phone": f"+855 {self.rng.randint(10, 99)} {self.rng.randint(100, 999)} {self.rng.randint(100, 999)}",
                        "weekday_hours": "08:00-16:00",
                        "saturday_hours": "08:00-11:30" if (i + j) % 2 == 0 else None,
                    }
                )
        return branches

    def _make_accounts(self) -> dict[str, Account]:
        owners = ["Sok Dara", "Chan Sreymom", "Lim Wei", "Keo Sophea", "Chen Li"]
        accounts = {}
        for i, owner in enumerate(owners):
            currency = "KHR" if i == 3 else "USD"
            balance = round(self.rng.uniform(50, 5000) * (4100 if currency == "KHR" else 1), 2)
            account = Account(f"ACC-{1001 + i}", owner, currency, balance)
            running = balance
            for k in range(8):
                amount = round(self.rng.uniform(-300, 400) * (4100 if currency == "KHR" else 1), 2)
                date = (self.today - dt.timedelta(days=3 * k + 1)).isoformat()
                kind = (
                    "deposit"
                    if amount > 0
                    else self.rng.choice(["card payment", "transfer out", "bill payment"])
                )
                account.transactions.append(
                    {"date": date, "description": kind, "amount": amount, "balance_after": round(running, 2)}
                )
                running -= amount
            accounts[account.account_id] = account
        return accounts

    # -- tools -------------------------------------------------------------------------
    def get_exchange_rate(self, base: str, quote: str, date: str | None = None) -> dict:
        base, quote = base.upper(), quote.upper()
        day = date or self.today.isoformat()
        if day not in self.rates:
            raise ValueError(f"no rate available for {day}")
        if base not in CURRENCIES or quote not in CURRENCIES:
            raise ValueError(f"unsupported currency pair {base}/{quote}")
        table = self.rates[day]
        rate = table[quote] / table[base]
        return {"base": base, "quote": quote, "rate": round(rate, 4 if rate < 10 else 2), "date": day}

    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> dict:
        quote = self.get_exchange_rate(from_currency, to_currency)
        converted = amount * quote["rate"]
        return {
            "amount": amount,
            "from_currency": quote["base"],
            "to_currency": quote["quote"],
            "rate": quote["rate"],
            "converted": round(converted, 2),
            "date": quote["date"],
        }

    @staticmethod
    def calculate_loan_payment(principal: float, annual_rate: float, months: int) -> dict:
        """Annuity payment; ``annual_rate`` in percent (e.g. 12 for 12 %)."""
        if months <= 0 or principal <= 0:
            raise ValueError("principal and months must be positive")
        r = annual_rate / 100 / 12
        payment = principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)
        total = payment * months
        return {
            "monthly_payment": round(payment, 2),
            "total_payment": round(total, 2),
            "total_interest": round(total - principal, 2),
            "months": months,
            "annual_rate_percent": annual_rate,
        }

    def find_branches(self, city: str, open_saturday: bool | None = None) -> dict:
        wanted = city.strip().lower()
        matches = [
            b
            for b in self.branches
            if wanted in (b["city"].lower(), b["city_km"], b["city_zh"])
            and (not open_saturday or b["saturday_hours"] is not None)
        ]
        return {"city": city, "branches": matches}

    def _account(self, account_id: str) -> Account:
        if account_id not in self.accounts:
            raise ValueError(f"account {account_id} not found")
        return self.accounts[account_id]

    def get_account_balance(self, account_id: str) -> dict:
        account = self._account(account_id)
        return {"account_id": account_id, "balance": account.balance, "currency": account.currency}

    def list_transactions(
        self, account_id: str, start_date: str | None = None, end_date: str | None = None
    ) -> dict:
        account = self._account(account_id)
        rows = [
            t
            for t in account.transactions
            if (start_date is None or t["date"] >= start_date) and (end_date is None or t["date"] <= end_date)
        ]
        return {"account_id": account_id, "currency": account.currency, "transactions": rows}

    def create_transfer(
        self, from_account: str, to_account: str, amount: float, currency: str, confirmed: bool = False
    ) -> dict:
        source = self._account(from_account)
        if amount <= 0:
            raise ValueError("amount must be positive")
        if currency.upper() != source.currency:
            raise ValueError(f"account {from_account} is in {source.currency}")
        summary = {
            "from_account": from_account,
            "to_account": to_account,
            "amount": amount,
            "currency": source.currency,
        }
        if not confirmed:
            return {"status": "confirmation_required", **summary}
        if amount > source.balance:
            return {"status": "rejected", "reason": "insufficient funds", **summary}
        reference = f"TRX-{len(self.transfers) + 70001}"
        self.transfers.append({"reference": reference, **summary})
        return {"status": "completed", "reference": reference, **summary}

    @staticmethod
    def block_card(card_last4: str, reason: str) -> dict:
        if not (len(card_last4) == 4 and card_last4.isdigit()):
            raise ValueError("card_last4 must be the last 4 digits of the card")
        return {"status": "blocked", "card_last4": card_last4, "reason": reason}

    @staticmethod
    def search_knowledge_base(query: str, lang: str = "en") -> dict:
        lowered = query.lower()
        scored = []
        for article in KB_ARTICLES:
            score = sum(1 for kw in article["keywords"] if kw in lowered)
            if score:
                scored.append((score, article))
        scored.sort(key=lambda item: -item[0])
        results = [{"id": a["id"], "text": a["text"].get(lang, a["text"]["en"])} for _, a in scored[:2]]
        return {"query": query, "results": results}

    @staticmethod
    def calculator(expression: str) -> dict:
        return {"expression": expression, "result": evaluate(expression)}

    # -- registry ----------------------------------------------------------------------
    def registry(self) -> ToolRegistry:
        string = {"type": "string"}
        number = {"type": "number"}
        return ToolRegistry(
            [
                Tool(
                    "get_exchange_rate",
                    "Get the bank's exchange rate between two currencies (ISO codes such as USD, KHR, THB, CNY).",
                    {
                        "base": {**string, "description": "currency to convert from"},
                        "quote": {**string, "description": "currency to convert to"},
                        "date": {**string, "description": "YYYY-MM-DD, default today"},
                    },
                    ["base", "quote"],
                    self.get_exchange_rate,
                ),
                Tool(
                    "convert_currency",
                    "Convert an amount between two currencies at today's bank rate.",
                    {"amount": number, "from_currency": string, "to_currency": string},
                    ["amount", "from_currency", "to_currency"],
                    self.convert_currency,
                ),
                Tool(
                    "calculate_loan_payment",
                    "Monthly payment, total payment and total interest of an amortised loan.",
                    {
                        "principal": number,
                        "annual_rate": {**number, "description": "annual interest rate in percent"},
                        "months": {"type": "integer"},
                    },
                    ["principal", "annual_rate", "months"],
                    self.calculate_loan_payment,
                ),
                Tool(
                    "find_branches",
                    "List bank branches in a city, optionally only those open on Saturday.",
                    {"city": string, "open_saturday": {"type": "boolean"}},
                    ["city"],
                    self.find_branches,
                ),
                Tool(
                    "get_account_balance",
                    "Balance of one of the authenticated customer's accounts.",
                    {"account_id": string},
                    ["account_id"],
                    self.get_account_balance,
                ),
                Tool(
                    "list_transactions",
                    "Transactions of an account, optionally between two dates (YYYY-MM-DD).",
                    {"account_id": string, "start_date": string, "end_date": string},
                    ["account_id"],
                    self.list_transactions,
                ),
                Tool(
                    "create_transfer",
                    "Transfer money between accounts. Call first with confirmed=false to get a summary; only call with confirmed=true after the user explicitly confirms.",
                    {
                        "from_account": string,
                        "to_account": string,
                        "amount": number,
                        "currency": string,
                        "confirmed": {"type": "boolean"},
                    },
                    ["from_account", "to_account", "amount", "currency"],
                    self.create_transfer,
                    side_effect=True,
                ),
                Tool(
                    "block_card",
                    "Immediately block a lost or stolen card.",
                    {"card_last4": {**string, "description": "last 4 digits of the card"}, "reason": string},
                    ["card_last4", "reason"],
                    self.block_card,
                    side_effect=True,
                ),
                Tool(
                    "search_knowledge_base",
                    "Search the bank's help articles.",
                    {"query": string, "lang": {"type": "string", "enum": ["en", "km", "zh"]}},
                    ["query"],
                    self.search_knowledge_base,
                ),
                Tool(
                    "calculator",
                    "Evaluate an arithmetic expression.",
                    {"expression": string},
                    ["expression"],
                    self.calculator,
                ),
            ]
        )
