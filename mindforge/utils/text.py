"""Text processing utilities for MindForge."""

import re
import hashlib
from collections import Counter


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def content_hash(text: str) -> str:
    """Generate a short content hash for deduplication tracking."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces, strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """Extract top keywords from text using term frequency.

    Filters out common stopwords and short tokens. Returns keywords
    ordered by frequency, most common first.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "because", "but", "and", "or", "if", "while",
        "about", "up", "it", "its", "this", "that", "these", "those", "i",
        "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "our", "their", "what", "which", "who", "whom",
        "also", "like", "get", "got", "one", "two", "make", "know", "think",
        "see", "come", "want", "use", "find", "give", "tell", "say", "said",
        "go", "going", "well", "back", "even", "new", "way", "look", "take",
        "people", "good", "much", "right", "still", "really", "thing",
        "don", "doesn", "didn", "won", "wouldn", "couldn", "shouldn",
        "let", "using", "used", "example", "work", "something", "actually",
        "basically", "essentially", "simply", "yeah", "yes", "okay",
    }

    # Tokenize: split on non-word characters
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
    # Filter
    tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]


def compute_text_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two texts based on word sets."""
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
