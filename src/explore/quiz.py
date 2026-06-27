"""
Turn a paper into study material: MCQs, flashcards, coding exercises, interview Q&A.

We ask the LLM for STRUCTURED output (JSON) so the UI can render it nicely (hidden
answers, flip-cards, etc.). Weaker models sometimes emit slightly-broken JSON, so the
parser is defensive and the caller falls back to showing raw text if parsing fails.
"""
import json
import re

from src.generation.provider import get_provider

QUIZ_TYPES = ["MCQs", "Flashcards", "Coding questions", "Interview questions"]

# {n} is filled in per request. Braces in the JSON examples are doubled for str.format.
_SYSTEMS = {
    "MCQs": (
        "Generate {n} multiple-choice questions that test understanding of the PAPER. "
        "Output ONLY a JSON array, no prose:\n"
        '[{{"question":"...","options":["...","...","...","..."],"answer":0,'
        '"explanation":"..."}}]\n'
        '"answer" is the 0-based index of the correct option. Base everything on the paper.'
    ),
    "Flashcards": (
        "Generate {n} study flashcards from the PAPER (a key term/concept on the front, a "
        "concise explanation on the back). Output ONLY a JSON array, no prose:\n"
        '[{{"front":"...","back":"..."}}]'
    ),
    "Coding questions": (
        "Generate {n} coding exercises based on the PAPER's methods (e.g. 'implement X'). "
        "Output ONLY a JSON array, no prose:\n"
        '[{{"question":"...","hint":"..."}}]'
    ),
    "Interview questions": (
        "Generate {n} interview-style questions WITH model answers about the PAPER's "
        "concepts. Output ONLY a JSON array, no prose:\n"
        '[{{"question":"...","answer":"..."}}]'
    ),
}


def _parse(raw: str):
    """Pull a JSON array out of the model's reply; return a list or None."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def generate_quiz(context: str, quiz_type: str, n: int = 5, provider=None) -> dict:
    """Return {'items': list|None, 'raw': str, 'type': str}."""
    provider = provider or get_provider()
    system = _SYSTEMS[quiz_type].format(n=n)
    user = f"PAPER:\n{context[:5000]}\n\nGenerate the {quiz_type} as a JSON array:"
    raw = provider.generate(system, user, max_tokens=1600)
    return {"items": _parse(raw), "raw": raw, "type": quiz_type}
