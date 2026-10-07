"""Random but realistic business sheets with Khmer / English / Chinese headers.

Each :class:`GeneratedTable` keeps the localised cell grid (what the model sees)
and a pandas frame with canonical column keys (used to compute ground truth).
All values are synthetic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

from dsm.tables.formula import index_to_col
from dsm.tools.bank_sim import CITIES

VOCAB = {
    "status": {
        "en": ["Active", "Closed", "Overdue"],
        "km": ["សកម្ម", "បានបិទ", "ហួសកាលកំណត់"],
        "zh": ["正常", "已结清", "逾期"],
    },
    "product": {
        "en": ["Rice", "Coffee", "Pepper", "Cashew", "Silk"],
        "km": ["អង្ករ", "កាហ្វេ", "ម្រេច", "គ្រាប់ស្វាយចន្ទី", "សូត្រ"],
        "zh": ["大米", "咖啡", "胡椒", "腰果", "丝绸"],
    },
    "department": {
        "en": ["Finance", "IT", "Operations", "Sales", "HR"],
        "km": ["ហិរញ្ញវត្ថុ", "ព័ត៌មានវិទ្យា", "ប្រតិបត្តិការ", "លក់", "ធនធានមនុស្ស"],
        "zh": ["财务", "信息技术", "运营", "销售", "人力资源"],
    },
    "person": {
        "en": [
            "Dara Sok",
            "Lina Chan",
            "Visal Keo",
            "Sophea Lim",
            "Rithy Heng",
            "Malis Mao",
            "Vanna Ouk",
            "Bopha Sim",
            "Kosal Nhem",
            "Chenda Yim",
        ],
        "km": [
            "សុខ ដារា",
            "ចាន់ លីណា",
            "កែវ វិសាល",
            "លឹម សុភា",
            "ហេង ឫទ្ធី",
            "ម៉ៅ ម្លិះ",
            "អ៊ុក វណ្ណា",
            "ស៊ឹម បុប្ផា",
            "ញ៉ែម កុសល",
            "យឹម ចិន្តា",
        ],
        "zh": ["王伟", "李娜", "张敏", "陈杰", "刘洋", "杨静", "赵磊", "黄丽", "周强", "吴芳"],
    },
    "region": {
        "en": ["North", "South", "East", "West"],
        "km": ["ខាងជើង", "ខាងត្បូង", "ខាងកើត", "ខាងលិច"],
        "zh": ["北部", "南部", "东部", "西部"],
    },
}

HEADERS = {
    "loans": {
        "en": ["Loan ID", "Branch", "Customer", "Amount (USD)", "Rate (%)", "Term (months)", "Status"],
        "km": ["លេខកម្ចី", "សាខា", "អតិថិជន", "ចំនួនទឹកប្រាក់ (USD)", "អត្រាការប្រាក់ (%)", "រយៈពេល (ខែ)", "ស្ថានភាព"],
        "zh": ["贷款编号", "分行", "客户", "金额（美元）", "利率（%）", "期限（月）", "状态"],
    },
    "deposits": {
        "en": ["Branch", "Month", "Deposits (USD)", "New accounts"],
        "km": ["សាខា", "ខែ", "ប្រាក់បញ្ញើ (USD)", "គណនីថ្មី"],
        "zh": ["分行", "月份", "存款（美元）", "新开户数"],
    },
    "sales": {
        "en": ["Product", "Region", "Units sold", "Unit price (USD)", "Revenue (USD)"],
        "km": ["ផលិតផល", "តំបន់", "ចំនួនលក់", "តម្លៃឯកតា (USD)", "ចំណូល (USD)"],
        "zh": ["产品", "地区", "销量", "单价（美元）", "收入（美元）"],
    },
    "payroll": {
        "en": ["Employee", "Department", "Base salary (USD)", "Bonus (USD)"],
        "km": ["បុគ្គលិក", "នាយកដ្ឋាន", "ប្រាក់ខែគោល (USD)", "ប្រាក់រង្វាន់ (USD)"],
        "zh": ["员工", "部门", "基本工资（美元）", "奖金（美元）"],
    },
}
KEYS = {
    "loans": ["loan_id", "branch", "customer", "amount", "rate", "term", "status"],
    "deposits": ["branch", "month", "deposits", "new_accounts"],
    "sales": ["product", "region", "units", "unit_price", "revenue"],
    "payroll": ["employee", "department", "salary", "bonus"],
}
SHEET_NAMES = {
    "loans": {"en": "Loans", "km": "កម្ចី", "zh": "贷款"},
    "deposits": {"en": "Deposits", "km": "ប្រាក់បញ្ញើ", "zh": "存款"},
    "sales": {"en": "Sales", "km": "ការលក់", "zh": "销售"},
    "payroll": {"en": "Payroll", "km": "ប្រាក់បៀវត្ស", "zh": "工资"},
}
# column roles used by the task generator
ROLES = {
    "loans": {
        "id": "loan_id",
        "name": "customer",
        "categories": ["branch", "status"],
        "measures": ["amount", "rate", "term"],
    },
    "deposits": {
        "id": None,
        "name": "branch",
        "categories": ["branch", "month"],
        "measures": ["deposits", "new_accounts"],
    },
    "sales": {
        "id": None,
        "name": "product",
        "categories": ["product", "region"],
        "measures": ["units", "revenue", "unit_price"],
    },
    "payroll": {
        "id": None,
        "name": "employee",
        "categories": ["department"],
        "measures": ["salary", "bonus"],
    },
}


@dataclass
class GeneratedTable:
    kind: str
    lang: str
    rows: list[list]  # header + data, localised (what the model sees)
    frame: pd.DataFrame  # canonical keys (ground truth)
    sheet: str
    columns: dict[str, int] = field(default_factory=dict)  # key -> 0-based column index

    def header(self, key: str) -> str:
        return self.rows[0][self.columns[key]]

    def letter(self, key: str) -> str:
        return index_to_col(self.columns[key] + 1)

    def range(self, key: str) -> str:
        col = self.letter(key)
        return f"{col}2:{col}{len(self.rows)}"


def _cities(lang: str) -> list[str]:
    return [city if lang == "en" else names[lang] for city, names in CITIES.items()]


def generate_table(kind: str, lang: str, rng: random.Random, n_rows: int | None = None) -> GeneratedTable:
    cities = _cities(lang)
    data: list[list] = []
    if kind == "loans":
        n = n_rows or rng.randint(6, 14)
        people = rng.sample(VOCAB["person"][lang], k=min(n, len(VOCAB["person"][lang])))
        for i in range(n):
            data.append(
                [
                    f"L-{1001 + i}",
                    rng.choice(cities),
                    people[i % len(people)],
                    rng.choice([1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 8000, 10000, 12000, 15000]),
                    rng.choice([8, 9, 10, 12, 14, 16, 18]),
                    rng.choice([6, 12, 18, 24, 36, 48]),
                    rng.choices(VOCAB["status"][lang], weights=[6, 3, 1])[0],
                ]
            )
    elif kind == "deposits":
        branches = rng.sample(cities, k=rng.randint(2, 4))
        months = [f"2026-{m:02d}" for m in range(1, rng.randint(3, 6) + 1)]
        for branch in branches:
            base = rng.randint(80, 400) * 1000
            for month in months:
                base = int(base * rng.uniform(0.9, 1.15) / 100) * 100
                data.append([branch, month, base, rng.randint(15, 160)])
    elif kind == "sales":
        products = rng.sample(VOCAB["product"][lang], k=rng.randint(3, 5))
        for product in products:
            for region in rng.sample(VOCAB["region"][lang], k=rng.randint(2, 4)):
                units = rng.randint(20, 900)
                price = rng.choice([1.5, 2, 2.5, 3, 4.5, 6, 8, 12])
                data.append([product, region, units, price, round(units * price, 2)])
    elif kind == "payroll":
        people = rng.sample(VOCAB["person"][lang], k=rng.randint(5, 10))
        for person in people:
            data.append(
                [
                    person,
                    rng.choice(VOCAB["department"][lang]),
                    rng.randint(40, 250) * 10,
                    rng.choice([0, 50, 100, 150, 200, 300]),
                ]
            )
    else:
        raise ValueError(f"unknown table kind {kind!r}")
    keys = KEYS[kind]
    rows = [list(HEADERS[kind][lang]), *data]
    frame = pd.DataFrame(data, columns=keys)
    return GeneratedTable(
        kind, lang, rows, frame, SHEET_NAMES[kind][lang], {k: i for i, k in enumerate(keys)}
    )
