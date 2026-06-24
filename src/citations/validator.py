"""
Verify the answer's citations are REAL -- the "show your receipts" guarantee.

We scan the answer text for [n] markers and check that:
  - every cited number actually exists in the sources (no invented citations),
  - we know which sources were used (so the UI can highlight just those),
  - we detect the explicit "not enough information" refusal.

This turns a plausible-sounding answer into a verifiable one.
"""
import re

# Matches [1], [2], [3] ... (single-number citation markers)
CITATION_RE = re.compile(r"\[(\d+)\]")

REFUSAL_PHRASE = "don't have enough information"


def validate_citations(answer: str, num_sources: int) -> dict:
    """Check the citations in an answer against the number of available sources."""
    cited = [int(n) for n in CITATION_RE.findall(answer)]
    cited_unique = sorted(set(cited))

    # A citation is "valid" if it points to a real source number (1..num_sources).
    valid = [n for n in cited_unique if 1 <= n <= num_sources]
    invalid = [n for n in cited_unique if n < 1 or n > num_sources]

    is_refusal = REFUSAL_PHRASE in answer.lower()

    return {
        "cited_sources": valid,          # which sources the answer actually used
        "invalid_citations": invalid,    # citations pointing nowhere (should be empty!)
        "has_citations": len(valid) > 0,
        "is_refusal": is_refusal,        # the honest "I don't know" path
        # The answer is trustworthy if it either cited real sources OR honestly refused,
        # and it never cited a source that doesn't exist.
        "is_grounded": (len(valid) > 0 or is_refusal) and not invalid,
    }


if __name__ == "__main__":
    # tiny self-test
    demo = "Multi-agent rewriting reduces leakage [1] and keeps fidelity [2][9]."
    print(validate_citations(demo, num_sources=6))
    # -> invalid_citations: [9], cited_sources: [1, 2], is_grounded: False
