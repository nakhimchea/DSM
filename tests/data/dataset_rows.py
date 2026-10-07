"""Hand-written rows in the formats of the public releases (the real files are not downloadable in CI)."""

from __future__ import annotations

import json

GSM8K = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
}
GSM8K_MONEY = {
    "question": "A pen costs $3 and a book costs $12. How much do 2 pens and 1 book cost?",
    "answer": "2 pens cost 2 * $3 = $<<2*3=6>>6.\nTogether they cost $6 + $12 = $<<6+12=18>>18.\n#### 18",
}

NUMINA = {
    "problem": "Find $x$ if $2x + 3 = 11$.",
    "solution": "Subtract 3 from both sides: $2x = 8$, so $x = \\boxed{4}$.",
    "answer": "4",
    "problem_type": "Algebra",
    "question_type": "math-word-problem",
    "problem_is_valid": "Yes",
    "solution_is_valid": "Yes",
    "source": "olympiads",
}
NUMINA_PROOF = {**NUMINA, "question_type": "proof", "answer": "proof"}
NUMINA_BROKEN_LATEX = {**NUMINA, "solution": "So $x = \\frac{8}{2$."}

XLAM = {
    "id": 0,
    "query": "Where can I find live giveaways for beta access and games?",
    "answers": json.dumps(
        [
            {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
            {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
        ]
    ),
    "tools": json.dumps(
        [
            {
                "name": "live_giveaways_by_type",
                "description": "Retrieve live giveaways from the GamerPower API based on the specified type.",
                "parameters": {
                    "type": {
                        "description": "The type of giveaways to retrieve (e.g., game, loot, beta).",
                        "type": "str",
                        "default": "game",
                    }
                },
            },
            {
                "name": "top_k",
                "description": "Top results.",
                "parameters": {
                    "k": {"description": "How many.", "type": "int"},
                    "tags": {"description": "Tags.", "type": "List[str], optional"},
                },
            },
        ]
    ),
}

HERMES = {
    "id": "h1",
    "conversations": [
        {
            "from": "system",
            "value": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "<tools>\n[{'type': 'function', 'function': {'name': 'get_stock_price', 'description': 'Get the current stock price', "
            "'parameters': {'type': 'object', 'properties': {'symbol': {'type': 'string', 'description': 'Ticker'}}, 'required': ['symbol']}}}]\n</tools> "
            "For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:\n"
            "<tool_call>\n{'arguments': <args-dict>, 'name': <function-name>}\n</tool_call>\n",
        },
        {"from": "human", "value": "What is the price of AAPL?"},
        {
            "from": "gpt",
            "value": "<tool_call>\n{'arguments': {'symbol': 'AAPL'}, 'name': 'get_stock_price'}\n</tool_call>",
        },
        {
            "from": "tool",
            "value": '<tool_response>\n{"name": "get_stock_price", "content": {"price": 189.5}}\n</tool_response>',
        },
        {"from": "gpt", "value": "AAPL is trading at 189.5 USD."},
    ],
}

GLAIVE = {
    "system": "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
    '{\n    "name": "calculate_bmi",\n    "description": "Calculate the Body Mass Index",\n    "parameters": {\n'
    '        "type": "object",\n        "properties": {\n            "weight": {"type": "number", "description": "kg"},\n'
    '            "height": {"type": "number", "description": "m"}\n        },\n        "required": ["weight", "height"]\n    }\n}\n',
    "chat": "USER: Hi, I need to calculate my BMI. I weigh 70 kg and I am 1.75 m tall.\n\n\n"
    'ASSISTANT: <functioncall> {"name": "calculate_bmi", "arguments": \'{"weight": 70, "height": 1.75}\'} <|endoftext|>\n\n\n'
    'FUNCTION RESPONSE: {"bmi": 22.86}\n\n\n'
    "ASSISTANT: Your Body Mass Index (BMI) is 22.86, which is in the healthy range. <|endoftext|>\n\n\n",
}
GLAIVE_NO_TOOLS = {
    "system": "SYSTEM: You are a helpful assistant, with no access to external functions.",
    "chat": "USER: Tell me a joke.\n\n\nASSISTANT: Why did the chicken cross the road? <|endoftext|>",
}

TOOLACE = {
    "system": "You are an expert in composing functions. You are given a question and a set of possible functions.\n"
    "Here is a list of functions in JSON format that you can invoke:\n"
    '[{"name": "newAddress", "description": "Generates a new address.", "parameters": {"type": "dict", "properties": '
    '{"cryptocurrency": {"description": "The coin", "type": "string"}, "count": {"description": "How many", "type": "float"}}, '
    '"required": ["cryptocurrency"]}, "required": null}].\nShould you decide to return the function call(s), Put it in the format of '
    "[func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n\nNO other text MUST be included.",
    "conversations": [
        {"from": "user", "value": "I need a new Ethereum address."},
        {"from": "assistant", "value": '[newAddress(cryptocurrency="Ethereum", count=1)]'},
        {"from": "tool", "value": '[{"name": "newAddress", "results": {"address": "0xABC"}}]'},
        {"from": "assistant", "value": "Your new Ethereum address is 0xABC."},
    ],
}

CHAT = {
    "messages": [
        {
            "role": "system",
            "content": "You are SmolLM, a helpful AI assistant trained by Hugging Face. /no_think",
        },
        {"role": "user", "content": "Give me two tips for saving money."},
        {
            "role": "assistant",
            "content": "<think>\n\n</think>\n\n1. Track your spending.\n2. Automate your savings.",
        },
    ],
    "source": "smoltalk-everyday",
}
CHAT_REASONING = {
    "messages": [
        {"role": "user", "content": "What is 17 * 3?"},
        {"role": "assistant", "content": "<think>17*3 = 51</think>\n\n51"},
    ]
}
CHAT_IDENTITY = {
    "messages": [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I'm an AI assistant."},
    ]
}
CHAT_LEAK = {
    "messages": [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! I am ChatGPT, how can I help?"},
    ]
}
CHAT_OPENAI_TOOLS = {
    "messages": [
        {"role": "user", "content": "What's the weather in Phnom Penh?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Phnom Penh"}'},
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": '{"temp_c": 33}'},
        {"role": "assistant", "content": "It is 33 °C in Phnom Penh."},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather by city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
}
CHAT_NV_TOOLCALL = {
    "messages": [
        {"role": "user", "content": "Weather in Siem Reap?"},
        {
            "role": "assistant",
            "content": '<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Siem Reap"}}]</TOOLCALL>',
        },
    ],
    "tools": CHAT_OPENAI_TOOLS["tools"],
}

TABFACT = {
    "table_text": "branch#loans#year\nphnom penh#120#2023\nsiem reap#80#2023\n",
    "table_caption": "loans by branch",
    "statement": "siem reap issued more loans than phnom penh",
    "label": 0,
}
WTQ = {
    "question": "which branch issued the most loans?",
    "answers": ["Phnom Penh"],
    "table": {
        "header": ["Branch", "Loans"],
        "rows": [["Phnom Penh", "120"], ["Siem Reap", "80"]],
        "name": "csv/204-csv/1.csv",
    },
}
TATQA = {
    "table": {
        "uid": "t1",
        "table": [["", "2019", "2018"], ["Revenue", "1,496.5", "1,437.5"], ["Costs", "900", "850"]],
    },
    "paragraphs": [{"uid": "p1", "order": 1, "text": "Revenue grew because of higher loan volumes."}],
    "questions": [
        {
            "uid": "q1",
            "order": 1,
            "question": "What is the percentage change in revenue from 2018 to 2019?",
            "answer": 4.1,
            "derivation": "(1,496.5-1,437.5)/1,437.5",
            "answer_type": "arithmetic",
            "answer_from": "table",
            "scale": "percent",
        },
        {
            "uid": "q2",
            "order": 2,
            "question": "Why did revenue grow?",
            "answer": ["higher loan volumes"],
            "derivation": "",
            "answer_type": "span",
            "answer_from": "text",
            "scale": "",
        },
        {
            "uid": "q3",
            "order": 3,
            "question": "What were the costs in 2019?",
            "answer": ["900"],
            "derivation": "",
            "answer_type": "span",
            "answer_from": "table",
            "scale": "million",
        },
    ],
}
FINQA = {
    "pre_text": ["the following table shows net sales ."],
    "post_text": ["amounts are in millions ."],
    "table": [["", "2019", "2018"], ["net sales", "$ 5829", "$ 5735"], ["costs", "$ 100", "$ 90"]],
    "qa": {
        "question": "what was the percentage change in net sales from 2018 to 2019?",
        "program": "subtract(5829, 5735), divide(#0, 5735)",
        "exe_ans": 0.01639,
        "answer": "1.6%",
    },
    "id": "f1",
}
FINQA_TABLE_OP = {
    **FINQA,
    "qa": {
        "question": "what is the average of net sales?",
        "program": "table_average(net sales, none)",
        "exe_ans": 5782.0,
    },
}
FINQA_WRONG = {**FINQA, "qa": {**FINQA["qa"], "exe_ans": 0.5}}

MASSIVE_ROWS = [
    {
        "id": "1",
        "locale": "km-KH",
        "scenario": "alarm",
        "intent": "alarm_set",
        "utt": "ដាស់ខ្ញុំនៅម៉ោងប្រាំព្រឹក",
        "annot_utt": "ដាស់ខ្ញុំនៅ [time : ម៉ោងប្រាំព្រឹក]",
    },
    {
        "id": "2",
        "locale": "km-KH",
        "scenario": "alarm",
        "intent": "alarm_query",
        "utt": "តើខ្ញុំមានម៉ោងរោទិ៍អ្វីខ្លះ",
        "annot_utt": "តើខ្ញុំមានម៉ោងរោទិ៍អ្វីខ្លះ",
    },
    {
        "id": "3",
        "locale": "en-US",
        "scenario": "alarm",
        "intent": "alarm_set",
        "utt": "wake me up at five am this week",
        "annot_utt": "wake me up at [time : five am] [date : this week]",
    },
    {
        "id": "4",
        "locale": "zh-TW",
        "scenario": "alarm",
        "intent": "alarm_set",
        "utt": "五點叫醒我",
        "annot_utt": "[time : 五點]叫醒我",
    },
]

HELPSTEER3 = {
    "domain": "general",
    "language": "english",
    "context": [{"role": "user", "content": "Explain compound interest in one sentence."}],
    "response1": "Compound interest is interest earned on both the original amount and on interest already added.",
    "response2": "It is money.",
    "overall_preference": -3,
}
HELPSTEER3_TIE = {**HELPSTEER3, "overall_preference": 1}
