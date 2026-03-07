"""
data/fetch_and_clean.py

WHY THIS FILE EXISTS:
The 20 Newsgroups dataset is famously noisy. Raw posts contain:
  - Email headers  (From:, Subject:, Organization:)
  - Quoted replies (lines starting with ">")
  - Signatures     (-- \n Name\n Phone number)
  - Footer boilerplate

If we embed these without cleaning, two problems occur:
  1. Documents cluster by *author email domain* rather than *topic*
  2. Short semantic content gets drowned out by formatting noise

This file handles: fetching → cleaning → length filtering → returning clean docs
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from sklearn.datasets import fetch_20newsgroups

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Data Class ────────────────────────────────────────────────────────────────
@dataclass
class Document:
    """
    Represents a single cleaned document.
    We store both the text AND the original category label.
    The label is used later to *evaluate* our clustering —
    we want to see if our unsupervised clusters align with the known categories.
    We do NOT use labels during clustering itself. That would be cheating.
    """
    doc_id: int
    text: str
    original_category: str          # e.g. "sci.space", "talk.politics.guns"
    original_category_id: int       # integer version (0–19)
    char_length: int = field(init=False)

    def __post_init__(self):
        self.char_length = len(self.text)


# ── Cleaning Functions ────────────────────────────────────────────────────────

def remove_email_headers(text: str) -> str:
    """
    Email headers look like:
        From: user@domain.com
        Subject: Re: Article title
        Organization: MIT
        Lines: 42

    These contain metadata, NOT content. If we keep them,
    documents from the same institution will cluster together
    regardless of their topic. We strip everything before
    the first blank line (which separates headers from body).
    """
    # A blank line marks end of headers in RFC 2822 email format
    parts = text.split('\n\n', 1)
    if len(parts) > 1:
        return parts[1]
    return text


def remove_quoted_replies(text: str) -> str:
    """
    Quoted replies look like:
        > This is what the previous person said
        > And this is another quoted line

    They add noise because they duplicate content from other posts.
    A post about space science that quotes a gun debate thread
    would confuse the embedding model.
    """
    lines = text.split('\n')
    # Remove lines that start with ">" (the standard quote prefix)
    clean_lines = [line for line in lines if not line.strip().startswith('>')]
    return '\n'.join(clean_lines)


def remove_signatures(text: str) -> str:
    """
    Email signatures are separated by "-- " (dash dash space)
    Everything after this line is boilerplate (name, contact, disclaimers).
    We cut it off entirely.
    """
    # Standard signature delimiter per RFC 3676
    sig_pattern = re.compile(r'\n--\s*\n.*', re.DOTALL)
    return sig_pattern.sub('', text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple blank lines, strip leading/trailing whitespace.
    This doesn't affect semantics but keeps token counts honest.
    """
    # Replace 3+ consecutive newlines with 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace tabs with spaces
    text = re.sub(r'\t', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def remove_non_ascii(text: str) -> str:
    """
    The dataset has some binary garbage and encoding artifacts.
    MiniLM handles unicode fine, but pure garbage bytes hurt quality.
    We keep standard ASCII + common unicode punctuation.
    """
    # Remove non-printable characters except newlines
    return re.sub(r'[^\x20-\x7E\n]', ' ', text)


def clean_document(raw_text: str) -> str:
    """
    Full cleaning pipeline. Order matters here:
    1. Strip headers first (they may contain fake quoted content)
    2. Remove quoted replies
    3. Remove signatures
    4. Clean up whitespace
    5. Remove garbage bytes
    """
    text = remove_email_headers(raw_text)
    text = remove_quoted_replies(text)
    text = remove_signatures(text)
    text = normalize_whitespace(text)
    text = remove_non_ascii(text)
    return text


# ── Length Filtering ──────────────────────────────────────────────────────────

MIN_CHARS = 100   # Below this, the post has no real content after cleaning
MAX_CHARS = 5000  # Above this, we truncate — MiniLM has a 256-token limit anyway
                  # and embedding very long docs adds time with no benefit
                  # (MiniLM uses mean-pooling, so long docs dilute the signal)

def is_valid_length(text: str) -> bool:
    """
    After cleaning, some posts become nearly empty.
    Example: a post that was 100% quoted replies becomes "".
    We discard these — they carry no semantic signal.
    """
    return MIN_CHARS <= len(text) <= MAX_CHARS


def truncate_if_needed(text: str, max_chars: int = MAX_CHARS) -> str:
    """
    Hard truncate at max_chars.
    We truncate at a word boundary to avoid cutting mid-word.
    """
    if len(text) <= max_chars:
        return text
    # Find the last space before the limit
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    return truncated[:last_space] if last_space > 0 else truncated


# ── Main Fetch Function ───────────────────────────────────────────────────────

def fetch_and_clean_dataset(
    subset: str = "all",
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
) -> tuple[list[Document], list[str]]:
    """
    Downloads the 20 Newsgroups dataset, cleans every document,
    filters out empties and very short posts.

    Args:
        subset: "train", "test", or "all". We use "all" for maximum data.
        min_chars: Minimum character count after cleaning.
        max_chars: Maximum character count (truncate beyond this).

    Returns:
        documents: List of Document dataclass instances
        category_names: The 20 original category label strings
    """
    logger.info("Fetching 20 Newsgroups dataset...")

    # IMPORTANT: We pass remove=() here — we do our OWN cleaning below.
    # sklearn has a built-in remove= option but it's too aggressive;
    # it sometimes removes content that's genuinely part of the post body.
    raw_data = fetch_20newsgroups(
        subset=subset,
        remove=(),          # We handle cleaning ourselves
        shuffle=True,
        random_state=42,    # Reproducible shuffling
    )

    category_names = raw_data.target_names  # ['alt.atheism', 'comp.graphics', ...]
    logger.info(f"Raw dataset: {len(raw_data.data)} documents, {len(category_names)} categories")

    # ── Clean every document ─────────────────────────────────────────────────
    documents: list[Document] = []
    skipped_too_short = 0
    skipped_too_long_truncated = 0

    for idx, (raw_text, category_id) in enumerate(
        zip(raw_data.data, raw_data.target)
    ):
        cleaned = clean_document(raw_text)

        # Filter out documents that became empty after cleaning
        if len(cleaned) < min_chars:
            skipped_too_short += 1
            continue

        # Truncate very long documents
        if len(cleaned) > max_chars:
            cleaned = truncate_if_needed(cleaned, max_chars)
            skipped_too_long_truncated += 1

        doc = Document(
            doc_id=idx,
            text=cleaned,
            original_category=category_names[category_id],
            original_category_id=int(category_id),
        )
        documents.append(doc)

    # ── Report ───────────────────────────────────────────────────────────────
    logger.info(f"Cleaned dataset: {len(documents)} documents kept")
    logger.info(f"  Skipped (too short after cleaning): {skipped_too_short}")
    logger.info(f"  Truncated (too long): {skipped_too_long_truncated}")
    logger.info(f"  Category distribution:")

    # Show how many docs per category survived cleaning
    from collections import Counter
    cat_counts = Counter(d.original_category for d in documents)
    for cat, count in sorted(cat_counts.items()):
        logger.info(f"    {cat:<35} {count}")

    return documents, category_names


# ── Entry point for testing ───────────────────────────────────────────────────
if __name__ == "__main__":
    docs, categories = fetch_and_clean_dataset()
    print(f"\nSample document (doc_id={docs[0].doc_id}):")
    print(f"  Category: {docs[0].original_category}")
    print(f"  Length:   {docs[0].char_length} chars")
    print(f"  Preview:  {docs[0].text[:300]}")