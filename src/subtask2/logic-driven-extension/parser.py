"""
parser.py
---------
Robust parser for syllogistic reasoning.
Uses spaCy (en_core_web_sm) with dependency-parse-aware noun extraction
and quantifier detection for identifying logical structure.

Handles the diverse phrasing patterns in the SemEval dataset:
  - "All X are Y", "Every single X is a Y", "Anything that is a X is also a Y"
  - "No X is Y", "Not a single X is a Y", "There are no X that are Y"
  - "Some X are Y", "A portion of X are Y", "At least one X is a Y"
  - "Some X are not Y", "Not all X are Y", "There exist some X which are not Y"

The parser operates on ENGLISH text only — the multilingual syllogism_t
is handled separately by the dataset module.
"""

import re
import spacy
from typing import List, Tuple, Optional


# ─── Conclusion Markers ────────────────────────────────────────────────────────
# These markers indicate the start of the conclusion sentence.
CONCLUSION_MARKERS = [
    "therefore",
    "consequently",
    "thus",
    "hence",
    "so",
    "it follows that",
    "it follows from this that",
    "the only logical conclusion is that",
    "the only conclusion is that",
    "the only conclusion that can be made is that",
    "as a result",
    "it is concluded that",
    "one must conclude that",
    "it must be the case that",
    "it is possible to conclude that",
    "it possible to conclude that",
    "we can conclude that",
    "this means that",
    "from this we can conclude that",
    "it can be concluded that",
    "it is true to say that",
    "it's the case that",
    "the conclusion is",
    "the conclusion that follows is that",
    "a valid conclusion would be that",
    "therefore, it is clear that",
    "therefore, it can be said that",
]

# ─── Quantifier Patterns ──────────────────────────────────────────────────────
# Order matters: more specific patterns first to avoid partial matches.
# Each entry: (compiled regex, relation_type)
# The regex captures (subject, predicate).

# Particular Negative: "Some A are not B"
PART_NEG_PATTERNS = [
    re.compile(
        r"(?:some|a\s+(?:portion|number|select\s+few)\s+(?:of\s+(?:the\s+)?)?|"
        r"at\s+least\s+one|there\s+(?:exist|are)\s+some|a\s+few|certain)\s+"
        r"(.*?)\s+(?:that\s+)?(?:are|is|can\s+be\s+said\s+to\s+be|can\s+be|cannot\s+be|"
        r"fail\s+to\s+be|do\s+not|does\s+not|don't|doesn't)\s+not\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:some|a\s+(?:portion|number)\s+(?:of\s+(?:the\s+)?)?|"
        r"at\s+least\s+one|there\s+(?:exist|are)\s+some)\s+"
        r"(.*?)\s+(?:are\s+not|is\s+not|cannot\s+be|can(?:'t|\s+not)\s+be|"
        r"fail\s+to\s+be)\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:some|a\s+(?:portion|number)\s+(?:of\s+(?:the\s+)?)?)\s+"
        r"(.*?)\s+(?:that\s+)?(?:are\s+not|is\s+not|cannot)\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"not\s+all\s+(.*?)\s+(?:are|is)\s+(.*)",
        re.IGNORECASE,
    ),
]

# Universal Negative: "No A is B"
UNIV_NEG_PATTERNS = [
    re.compile(
        r"(?:no|not\s+a\s+single|nothing\s+that\s+is\s+(?:a|an)?)\s+"
        r"(.*?)\s+(?:that\s+)?(?:are|is|can(?:\s+be)?)\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"there\s+(?:are|is|exists?)\s+no\s+"
        r"(.*?)\s+(?:that\s+|which\s+|who\s+)?(?:are|is|can(?:\s+be)?)\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:no)\s+(.*?)\s+(?:is|are|can\s+be)\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:no)\s+(.*?)\s+(?:whatsoever|at\s+all)",
        re.IGNORECASE,
    ),
]

# Universal Affirmative: "All A are B"
UNIV_AFF_PATTERNS = [
    re.compile(
        r"(?:all|every(?:\s+single)?|any(?:thing)?|each)\s+"
        r"(.*?)\s+(?:that\s+(?:is|are)\s+(?:a|an)\s+)?(.*?)\s+"
        r"(?:is\s+(?:also\s+)?(?:a|an)\s+|are\s+(?:also\s+)?|"
        r"is\s+(?:also\s+)?(?:a\s+type\s+of\s+)?|can\s+be\s+said\s+to\s+be\s+(?:a\s+)?)\s*(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:all|every(?:\s+single)?|any|each)\s+"
        r"(.*?)\s+(?:are|is)\s+(?:a\s+type\s+of\s+|a\s+kind\s+of\s+)?(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:all|every(?:\s+single)?|any|each)\s+"
        r"(.*?)\s+(?:are|is)\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:anything|anyone|everything|everyone)\s+(?:that|who)\s+is\s+(?:a|an)\s+"
        r"(.*?)\s+is\s+(?:also\s+)?(?:a|an)\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(.*?)\s+(?:are|is),?\s+without\s+exception,?\s+(.*)",
        re.IGNORECASE,
    ),
]

# Particular Affirmative: "Some A are B"
PART_AFF_PATTERNS = [
    re.compile(
        r"(?:some|a\s+(?:portion|number|select\s+few)\s+(?:of\s+(?:the\s+)?)?|"
        r"at\s+least\s+one|there\s+(?:exist|are)\s+(?:some|at\s+least\s+one)|"
        r"a\s+few|certain)\s+"
        r"(.*?)\s+(?:that\s+)?(?:are|is|can\s+be(?:\s+said\s+to\s+be)?)\s+(.*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"there\s+(?:exist|are)\s+some\s+(.*?)\s+(?:which|that|who)\s+(?:are|is)\s+(.*)",
        re.IGNORECASE,
    ),
]


def _load_spacy(model_name: str = "en_core_web_sm"):
    """Load spaCy model with lazy caching."""
    if not hasattr(_load_spacy, "_cache"):
        _load_spacy._cache = {}
    if model_name not in _load_spacy._cache:
        _load_spacy._cache[model_name] = spacy.load(model_name)
    return _load_spacy._cache[model_name]


def clean_sentence(sentence: str) -> str:
    """Remove conclusion markers and normalize whitespace."""
    sentence = sentence.strip()
    s_lower = sentence.lower()

    # Sort markers by length descending to match longest first
    sorted_markers = sorted(CONCLUSION_MARKERS, key=len, reverse=True)
    for marker in sorted_markers:
        if s_lower.startswith(marker):
            sentence = sentence[len(marker):].strip()
            # Remove leading comma or period after marker
            sentence = sentence.lstrip(",. ")
            break

    return sentence.strip(". ")


def split_syllogism(syllogism: str, spacy_model: str = "en_core_web_sm") -> Tuple[List[str], str]:
    """
    Split a syllogism into premises and conclusion.

    Returns:
        (premises: List[str], conclusion: str)

    Strategy:
      1. Try spaCy sentence splitting first
      2. Fall back to period-based splitting
      3. Identify conclusion by markers or take the last sentence
    """
    nlp = _load_spacy(spacy_model)
    doc = nlp(syllogism)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # If spaCy gives < 3 sentences, try period-based splitting
    if len(sentences) < 3:
        sentences = [s.strip() + "." for s in syllogism.split(".") if s.strip()]

    if len(sentences) < 2:
        # Can't split meaningfully
        return [syllogism], ""

    # Identify conclusion: check each sentence for conclusion markers
    conclusion_idx = None
    for i, sent in enumerate(sentences):
        s_lower = sent.lower().strip()
        for marker in CONCLUSION_MARKERS:
            if s_lower.startswith(marker):
                conclusion_idx = i
                break
        if conclusion_idx is not None:
            break

    # Default: last sentence is the conclusion
    if conclusion_idx is None:
        conclusion_idx = len(sentences) - 1

    premises = sentences[:conclusion_idx] + sentences[conclusion_idx + 1:]
    conclusion = sentences[conclusion_idx]

    # Ensure at least 2 premises if possible
    if len(premises) == 0 and len(sentences) >= 2:
        premises = sentences[:-1]
        conclusion = sentences[-1]

    return premises, conclusion


def extract_entities(sentences: List[str], spacy_model: str = "en_core_web_sm") -> Tuple[List[str], dict, dict]:
    """
    Extract entities (noun chunks) from sentences, filter and deduplicate.

    Returns:
        (entities: List[str], sym_map: {entity -> sym}, rev_sym_map: {sym -> entity})
    """
    nlp = _load_spacy(spacy_model)
    doc = nlp(" ".join(sentences))
    entities = []

    stopwords = {"a", "an", "the", "all", "some", "no", "every", "any", "certain",
                 "each", "not", "at", "least", "there", "this", "that", "it"}
    pronouns = {"it", "they", "them", "he", "she", "we", "us", "i", "me",
                "this", "that", "these", "those", "one"}

    for chunk in doc.noun_chunks:
        text = chunk.text.lower().strip()
        words = text.split()
        if not words:
            continue

        # Remove leading determiners/stopwords
        while words and words[0] in stopwords:
            words = words[1:]

        text = " ".join(words)
        if not text or text in pronouns:
            continue

        if text not in entities:
            entities.append(text)

    # Sort by length descending (replace longer spans first)
    entities = sorted(entities, key=len, reverse=True)

    sym_map = {}
    rev_sym_map = {}
    for i, ent in enumerate(entities):
        sym = f"sym_{i}"
        sym_map[ent] = sym
        rev_sym_map[sym] = ent

    return entities, sym_map, rev_sym_map


def encode_sentences(sentences: List[str], entities: List[str], sym_map: dict) -> List[str]:
    """Replace entities in sentences with symbolic placeholders."""
    encoded = []
    for sent in sentences:
        s = clean_sentence(sent)
        for ent in entities:
            s = s.replace(ent, sym_map[ent])
        encoded.append(s)
    return encoded


def extract_relation(sentence: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract a logical relation from a single encoded sentence.

    Returns:
        (relation_type, subject, predicate) or (None, None, None)

    Relation types: "subset", "disjoint", "intersect", "diff_intersect"
    """
    sentence = sentence.strip(". ")

    # Try particular negative first (most specific — "some A are not B")
    for pat in PART_NEG_PATTERNS:
        m = pat.search(sentence)
        if m:
            return "diff_intersect", m.group(1).strip(), m.group(2).strip()

    # Universal negative ("No A is B")
    for pat in UNIV_NEG_PATTERNS:
        m = pat.search(sentence)
        if m:
            groups = m.groups()
            if len(groups) >= 2:
                return "disjoint", groups[0].strip(), groups[1].strip()

    # Universal affirmative ("All A are B")
    for pat in UNIV_AFF_PATTERNS:
        m = pat.search(sentence)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if len(groups) >= 2:
                subj = groups[0].strip()
                pred = groups[-1].strip()
                if subj and pred:
                    return "subset", subj, pred

    # Particular affirmative ("Some A are B")
    for pat in PART_AFF_PATTERNS:
        m = pat.search(sentence)
        if m:
            return "intersect", m.group(1).strip(), m.group(2).strip()

    return None, None, None


def extract_relations_from_encoded(encoded_sentences: List[str]) -> List[dict]:
    """
    Extract logical relations from a list of encoded sentences.

    Returns:
        List of {"type": str, "args": (str, str)}
    """
    relations = []
    for s in encoded_sentences:
        rel_type, subj, pred = extract_relation(s)
        if rel_type:
            relations.append({"type": rel_type, "args": (subj, pred)})
    return relations


def parse_syllogism(syllogism: str, spacy_model: str = "en_core_web_sm") -> dict:
    """
    Full parsing pipeline for a single syllogism (English).

    Returns:
        {
            "premises": [str],
            "conclusion": str,
            "entities": [str],
            "sym_map": {entity -> sym},
            "rev_sym_map": {sym -> entity},
            "encoded_premises": [str],
            "encoded_conclusion": str,
            "relations": [{"type": str, "args": (str, str)}],
        }
    """
    premises, conclusion = split_syllogism(syllogism, spacy_model)
    all_sentences = premises + ([conclusion] if conclusion else [])

    entities, sym_map, rev_sym_map = extract_entities(all_sentences, spacy_model)
    encoded_all = encode_sentences(all_sentences, entities, sym_map)

    if conclusion:
        encoded_premises = encoded_all[:-1]
        encoded_conclusion = encoded_all[-1]
    else:
        encoded_premises = encoded_all
        encoded_conclusion = ""

    relations = extract_relations_from_encoded(encoded_premises)

    return {
        "premises": premises,
        "conclusion": conclusion,
        "entities": entities,
        "sym_map": sym_map,
        "rev_sym_map": rev_sym_map,
        "encoded_premises": encoded_premises,
        "encoded_conclusion": encoded_conclusion,
        "relations": relations,
    }
