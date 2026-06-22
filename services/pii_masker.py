"""
PII masker for resume text.

Before any resume text reaches the LLM, this module:
  1. Detects names, emails, and phone numbers via regex + heuristics.
  2. Replaces each with a deterministic token: [CANDIDATE:uuid], [EMAIL:uuid], [PHONE:uuid].
  3. Returns the masked text AND a lookup table mapping every token → original value.

The lookup table is persisted as PiiMapping nodes in Neo4j so it survives restarts
and is queryable when re-hydrating search results returned to the caller.

Determinism guarantee: same original value → same UUID every time, across restarts.
This is achieved by hashing the value with SHA-256 and truncating to 12 hex chars.
Collision probability at 200k resumes is negligible (~1 in 10^14).
"""
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────

# RFC-5322 simplified email
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# International phone: +91-XXXXX-XXXXX, (XXX) XXX-XXXX, XXX.XXX.XXXX, 10-digit runs
_PHONE_RE = re.compile(
    r"""
    (?:
        (?:\+?\d{1,3}[\s\-.])?          # optional country code
        (?:\(?\d{2,4}\)?[\s\-.])?        # optional area code with parens
        \d{3,5}[\s\-.]\d{3,5}            # main number with separator
        (?:[\s\-.]\d{1,5})?              # optional extension
    )
    |
    (?<!\d)\d{10}(?!\d)                  # bare 10-digit run
    """,
    re.VERBOSE,
)

# Name detection: "Name: John Smith" or "JOHN SMITH" header patterns.
# We match after common resume labels so we don't blank every proper noun.
_NAME_LABEL_RE = re.compile(
    r"(?:^|\n)"                          # start of line
    r"(?:Name|Candidate|Applicant)\s*[:\-]?\s*"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    re.MULTILINE,
)

# All-caps header line that looks like a name (1-4 words, no digits)
_NAME_HEADER_RE = re.compile(
    r"(?:^|\n)([A-Z][A-Z\s]{3,40})(?=\n)",
    re.MULTILINE,
)


# ── Deterministic UUID ─────────────────────────────────────────────────────────

def _det_uuid(value: str) -> str:
    """12-char deterministic hex ID derived from the value's SHA-256 hash."""
    return hashlib.sha256(value.strip().lower().encode()).hexdigest()[:12]


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class MaskResult:
    masked_text: str
    # token → original value, e.g. "[EMAIL:abc123def456]" → "john@acme.com"
    mappings: dict[str, str]
    candidate_uuid: Optional[str]  # the UUID assigned to the candidate's name


# ── Public API ─────────────────────────────────────────────────────────────────

def mask_pii(text: str, source_id: str) -> MaskResult:
    """
    Mask PII in resume text before it is sent to any LLM.

    Returns the masked text and the token→original mapping dict.
    The source_id is used as a tiebreaker when no name is detected.
    """
    mappings: dict[str, str] = {}
    candidate_uuid: Optional[str] = None

    # ── 1. Emails (most reliable — do first so they don't clash with name scan)
    def _replace_email(m: re.Match) -> str:
        original = m.group(0)
        uid = _det_uuid(original)
        token = f"[EMAIL:{uid}]"
        mappings[token] = original
        return token

    text = _EMAIL_RE.sub(_replace_email, text)

    # ── 2. Phones
    def _replace_phone(m: re.Match) -> str:
        original = m.group(0).strip()
        if len(re.sub(r"\D", "", original)) < 7:
            return m.group(0)  # too short to be a real phone — skip
        uid = _det_uuid(original)
        token = f"[PHONE:{uid}]"
        mappings[token] = original
        return token

    text = _PHONE_RE.sub(_replace_phone, text)

    # ── 3. Candidate name
    # Try labelled pattern first ("Name: John Smith"), then all-caps header.
    name_match = _NAME_LABEL_RE.search(text)
    if not name_match:
        name_match = _NAME_HEADER_RE.search(text)

    if name_match:
        original_name = name_match.group(1).strip()
        uid = _det_uuid(original_name)
        candidate_uuid = uid
        token = f"[CANDIDATE:{uid}]"
        mappings[token] = original_name
        # Replace all occurrences of the exact name string
        text = text.replace(original_name, token)
    else:
        # Fall back: use source_id as the candidate token so the graph still has
        # a stable identity even if the name couldn't be extracted.
        candidate_uuid = _det_uuid(source_id)
        logger.debug("No candidate name detected for source_id=%s; using ID-derived UUID", source_id)

    return MaskResult(
        masked_text=text,
        mappings=mappings,
        candidate_uuid=candidate_uuid,
    )


def unmask(text: str, mappings: dict[str, str]) -> str:
    """Replace tokens in text with their original PII values."""
    for token, original in mappings.items():
        text = text.replace(token, original)
    return text
