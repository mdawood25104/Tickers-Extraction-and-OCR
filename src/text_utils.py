import re
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    text = str(text)

    # Remove excessive whitespace while preserving Unicode letters.
    text = re.sub(r"\s+", " ", text).strip()

    # Remove obvious OCR-only artifacts at the edges.
    text = text.strip(" |_-—–:;")

    return text


def similarity(a: str, b: str) -> float:
    a = normalize_text(a).lower()
    b = normalize_text(b).lower()

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def are_similar(a: str, b: str, threshold: float = 0.88) -> bool:
    return similarity(a, b) >= threshold
