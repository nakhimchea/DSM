"""Quality / safety filters for generated or converted training text."""

from __future__ import annotations

import functools
import hashlib
import re
import unicodedata

from dsm.text import normalize_text

_OTHER_AIS = (
    r"ChatGPT|GPT-?\d(?:\.\d)?|Claude|Gemini|Gemma|Bard|LLaMA|Llama|Mistral|Mixtral|DeepSeek|Copilot|ERNIE|Kimi|Doubao"
    r"|GLM|ChatGLM|SmolLM|Nemotron|Phi-?\d|Grok|InternLM|Baichuan|MiniCPM|Hunyuan|Falcon|OLMo|Tulu|Command R|Aya|SeaLLM"
)
_OTHER_LABS = (
    r"OpenAI|Anthropic|Google|DeepMind|Meta|Mistral AI|DeepSeek|Microsoft|Baidu|Moonshot|ByteDance|Zhipu|Hugging ?Face"
    r"|NVIDIA|xAI|Tencent|Cohere|IBM|Amazon|Apple|Allen Institute|AI2|Technology Innovation Institute|Shanghai AI Lab"
)

# First-person self-identification as another assistant (identity leakage from teachers / datasets).
_LEAK_PATTERNS = [
    re.compile(
        rf"\bI(?:'m| am)\s+(?:an?\s+)?(?:AI\s+)?(?:language\s+)?(?:model\s+)?(?:called\s+|named\s+)?(?:{_OTHER_AIS})\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:I(?:'m| am| was)|As an AI(?: language model| assistant)?)\b[^.\n]{{0,60}}\b(?:developed|created|built|trained|made)\s+by\s+(?:{_OTHER_LABS})\b",
        re.I,
    ),
    re.compile(
        rf"我(?:是|叫)\s*(?:由\S{{0,12}}(?:开发|训练|研发)的)?\s*(?:{_OTHER_AIS}|文心一言|豆包|智谱清言)"
    ),
    re.compile(
        r"由\s*(?:OpenAI|Anthropic|谷歌|Google|百度|字节跳动|月之暗面|深度求索|Meta|腾讯|英伟达|NVIDIA|Hugging ?Face|微软|智谱)"
        r"\s*(?:公司)?\s*(?:开发|训练|研发|打造)的(?:AI|人工智能|语言模型|助手|大模型)"
    ),
    re.compile(rf"ខ្ញុំ\s*(?:គឺ)?\s*(?:ជា)?\s*(?:{_OTHER_AIS})"),
]


@functools.lru_cache(maxsize=1)
def _upstream_patterns() -> tuple[str, ...]:
    from dsm.utils import load_upstream

    try:
        return tuple(p.lower() for p in load_upstream()["scrub"]["self_identification"])
    except FileNotFoundError:  # pragma: no cover - installed without the repo configs
        return ()


def identity_leak(text: str) -> str | None:
    """Return the offending fragment if ``text`` self-identifies as another model, else ``None``."""
    lowered = text.lower()
    for literal in _upstream_patterns():
        if literal in lowered:
            return literal
    for pattern in _LEAK_PATTERNS:
        if match := pattern.search(text):
            return match.group(0)
    return None


_SECOND_PERSON = [
    re.compile(
        rf"\byou(?:'re| are)\s+(?:an?\s+)?(?:AI\s+)?(?:assistant\s+)?(?:named\s+|called\s+)?(?:{_OTHER_AIS})\b",
        re.I,
    ),
    re.compile(rf"\b(?:trained|developed|created|built|made)\s+by\s+(?:{_OTHER_LABS})\b", re.I),
    re.compile(rf"你是\s*(?:由\S{{0,12}}(?:开发|训练|研发)的)?\s*(?:{_OTHER_AIS})"),
]


def foreign_identity(text: str) -> str | None:
    """Fragment of a system prompt that gives the assistant another identity ("You are <X>, trained by <Y>")."""
    if leak := identity_leak(text):
        return leak
    for pattern in _SECOND_PERSON:
        if match := pattern.search(text):
            return match.group(0)
    return None


_REFUSAL = re.compile(
    r"\b(?:I(?:'m| am) (?:sorry|unable|not able)|I can(?:no|')t (?:help|assist|provide|do that)|As an AI,? I can(?:no|')t)"
    r"|抱歉，?我(?:无法|不能)|对不起，?我(?:无法|不能)"
    r"|សូមអភ័យទោស\s*ខ្ញុំមិនអាច",
    re.I,
)


def looks_like_refusal(text: str) -> bool:
    return _REFUSAL.search(text[:300]) is not None


_SCRIPTS = {
    "km": re.compile("[ក-៿᧠-᧿]"),
    "zh": re.compile("[一-鿿㐀-䶿]"),
    "en": re.compile("[A-Za-z]"),
}


def script_ratio(text: str, lang: str) -> float:
    """Share of letters in ``text`` that belong to the script of ``lang``."""
    letters = [c for c in text if unicodedata.category(c)[0] in ("L", "M")]
    if not letters:
        return 0.0
    pattern = _SCRIPTS[lang]
    return sum(1 for c in letters if pattern.match(c)) / len(letters)


def language_matches(text: str, lang: str, threshold: float = 0.5) -> bool:
    """Whether the answer is written in ``lang`` (code, numbers and names are tolerated)."""
    if lang not in _SCRIPTS:
        return True
    return script_ratio(text, lang) >= threshold


def detect_lang(text: str, threshold: float = 0.3) -> str:
    """``km`` / ``zh`` when that script reaches ``threshold`` of the letters, else ``en``.

    Khmer is checked first: Khmer text often embeds Latin terms, and Chinese
    rarely contains Khmer. Good enough for routing converted data; real corpora
    go through the language-ID stage of the data pipeline.
    """
    for lang in ("km", "zh"):
        if script_ratio(text, lang) >= threshold:
            return lang
    return "en"


_IDENTITY_QUESTION = re.compile(
    r"\b(?:who|what)\s+(?:are|r)\s+(?:you|u)\b|\bwho\s+(?:made|created|built|trained|developed|designed)\s+you\b"
    r"|\bwhat(?:'s| is)\s+your\s+name\b|\byour\s+(?:name|creator|developer|maker)\b"
    r"|\bare\s+you\s+(?:an?\s+)?(?:ai|bot|chatgpt|gpt|human|real|robot|language model)\b|\bintroduce\s+yourself\b"
    r"|\bwhich\s+(?:company|model|llm)\b[^?\n]{0,40}\byou\b|\b(?:what|which)\s+model\s+are\s+you\b"
    r"|你是谁|你叫什么|您是谁|你的名字|介绍一下你自己|介绍下你自己|谁(?:开发|创造|训练|制作|研发)了你|你是(?:由)?(?:哪|谁)"
    r"|你是什么(?:模型|AI|人工智能)|你是(?:人|机器人|AI|ChatGPT)吗"
    r"|អ្នក(?:ជា|គឺជា)?នរណា|អ្នកឈ្មោះអ្វី|ឈ្មោះរបស់អ្នក|នរណា(?:បាន)?(?:បង្កើត|អភិវឌ្ឍ)អ្នក",
    re.I,
)


def is_identity_question(text: str) -> bool:
    """Questions about the assistant itself (answered only by DSM's identity data)."""
    return _IDENTITY_QUESTION.search(text) is not None


def is_normalized(text: str) -> bool:
    return normalize_text(text) == text


def degenerate_repetition(text: str, n: int = 20, max_repeats: int = 4, max_unit: int = 200) -> bool:
    """True for looping output.

    Two checks: the text ENDS in a unit repeated at least 4 times (and over at
    least 20 characters) - the typical failure of a truncated generation - or any
    ``n``-character window recurs more than ``max_repeats`` times (sampled every
    ``n // 2`` characters, so ordinary tables and lists do not trigger it).
    """
    stripped = text.rstrip()
    for unit in range(2, min(max_unit, len(stripped) // max_repeats) + 1):
        repeats = max(max_repeats, -(-n // unit))
        looping = unit * repeats <= len(stripped) and stripped.endswith(stripped[-unit:] * repeats)
        if looping and stripped[-unit:].strip(" -=_*#|.\n"):
            return True
    if len(text) < n * (max_repeats + 1):
        return False
    counts: dict[str, int] = {}
    for i in range(0, len(text) - n, max(1, n // 2)):
        window = text[i : i + n]
        counts[window] = counts.get(window, 0) + 1
        if counts[window] > max_repeats:
            return True
    return False


def content_hash(text: str) -> str:
    """Hash of whitespace/case-normalised text, for exact deduplication."""
    canonical = " ".join(normalize_text(text).lower().split())
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()
