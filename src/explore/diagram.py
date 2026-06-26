"""
Turn an answer into a concept DIAGRAM (Mermaid flowchart).

Given an explanation, we ask the LLM for a small Mermaid `flowchart TD` capturing the
key concepts and how they relate. The UI renders it visually. This is on-demand (only
runs when the user clicks "Show concept diagram"), so it's cheap on tokens.
"""
import re

from src.generation.provider import get_provider

MERMAID_SYSTEM = """You convert a technical explanation into a concise Mermaid diagram.
Output ONLY valid Mermaid code for a `flowchart TD` (top-down) with 4-8 nodes capturing
the key concepts and how they relate. Rules:
- start with `flowchart TD`
- node ids are simple letters (A, B, C...); labels go in square brackets: A[Short label]
- short labels (a few words), short edge labels where useful: A -->|improves| B
- NO prose, NO explanations, NO code fences -- only the Mermaid code."""


def _clean(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:mermaid)?", "", code).strip()
    code = re.sub(r"```$", "", code).strip()
    # Drop any stray prose before the diagram keyword.
    m = re.search(r"(flowchart|graph)\s+\w+.*", code, flags=re.DOTALL)
    code = m.group(0) if m else "flowchart TD\n" + code
    # Fix common small-model syntax slips:
    code = code.replace("|>", "|")     # bad edge label close: -->|label|> B  -> -->|label| B
    code = code.replace("==>", "-->")  # normalize thick arrows
    return code.strip()


def generate_mermaid(answer: str, provider=None) -> str:
    provider = provider or get_provider()
    raw = provider.generate(MERMAID_SYSTEM,
                            f"Explanation:\n{answer[:1500]}\n\nMermaid flowchart:")
    return _clean(raw)
