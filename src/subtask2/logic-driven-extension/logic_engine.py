"""
logic_engine.py
---------------
Inference rules and negative sample generation for syllogistic reasoning.

Implements the logical inference engine that extends the explicitly stated
premises with implicit/derived relations, and generates negative (corrupted)
samples for contrastive learning.

Inference Rules:
  1. Subset transitivity:       A⊂B, B⊂C → A⊂C
  2. Subset + disjoint:         A⊂B, B∩C=∅ → A∩C=∅
  3. Intersect + subset:        A∩B≠∅, B⊂C → A∩C≠∅
  4. Intersect + disjoint:      A∩B≠∅, B∩C=∅ → ∃x∈A: x∉C
  5. Disjoint symmetry:         A∩B=∅ → B∩A=∅
  6. Subset contrapositive:     A⊂B → A∩¬B=∅
  7. Subset + intersect conv:   A∩B≠∅, A⊂C → C∩B≠∅
  8. Intersect symmetry:        A∩B≠∅ → B∩A≠∅
"""

import copy
import random
from typing import List


# ─── Negation Helpers ──────────────────────────────────────────────────────────

def is_negated(term: str) -> bool:
    """Check if a term is in negated form: not(X)."""
    return term.startswith("not(") and term.endswith(")")


def negate_term(term: str) -> str:
    """Toggle negation: X → not(X), not(X) → X."""
    if is_negated(term):
        return term[4:-1]
    return f"not({term})"


# ─── Inference Engine ──────────────────────────────────────────────────────────

def infer_implicit_relations(relations: List[dict], max_iterations: int = 10) -> List[dict]:
    """
    Apply inference rules to extend the explicitly stated relations.

    Uses a fixed-point iteration: keeps applying rules until no new
    relations are derived or max_iterations is reached.

    Args:
        relations: List of {"type": str, "args": (str, str)}
        max_iterations: Safety limit to prevent infinite loops

    Returns:
        Extended list of relations (includes originals)
    """
    inferred = copy.deepcopy(relations)
    seen = set()

    def _key(r):
        return (r["type"], r["args"][0], r["args"][1])

    for r in inferred:
        seen.add(_key(r))

    for _ in range(max_iterations):
        new_found = False
        snapshot = list(inferred)  # iterate over a snapshot

        for r1 in snapshot:
            # Rule 5: Disjoint symmetry — A∩B=∅ → B∩A=∅
            if r1["type"] == "disjoint":
                new_r = {"type": "disjoint", "args": (r1["args"][1], r1["args"][0])}
                k = _key(new_r)
                if k not in seen:
                    inferred.append(new_r)
                    seen.add(k)
                    new_found = True

            # Rule 6: Subset contrapositive — A⊂B → A∩¬B=∅
            if r1["type"] == "subset":
                new_r = {"type": "disjoint", "args": (r1["args"][0], negate_term(r1["args"][1]))}
                k = _key(new_r)
                if k not in seen:
                    inferred.append(new_r)
                    seen.add(k)
                    new_found = True

            # Rule 8: Intersect symmetry — A∩B≠∅ → B∩A≠∅
            if r1["type"] == "intersect":
                new_r = {"type": "intersect", "args": (r1["args"][1], r1["args"][0])}
                k = _key(new_r)
                if k not in seen:
                    inferred.append(new_r)
                    seen.add(k)
                    new_found = True

            for r2 in snapshot:
                if r1 is r2:
                    continue

                # Rule 1: Subset transitivity — A⊂B, B⊂C → A⊂C
                if r1["type"] == "subset" and r2["type"] == "subset":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "subset", "args": (r1["args"][0], r2["args"][1])}
                        k = _key(new_r)
                        if k not in seen:
                            inferred.append(new_r)
                            seen.add(k)
                            new_found = True

                # Rule 2: Subset + disjoint — A⊂B, B∩C=∅ → A∩C=∅
                if r1["type"] == "subset" and r2["type"] == "disjoint":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "disjoint", "args": (r1["args"][0], r2["args"][1])}
                        k = _key(new_r)
                        if k not in seen:
                            inferred.append(new_r)
                            seen.add(k)
                            new_found = True

                # Rule 3: Intersect + subset — A∩B≠∅, B⊂C → A∩C≠∅
                if r1["type"] == "intersect" and r2["type"] == "subset":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "intersect", "args": (r1["args"][0], r2["args"][1])}
                        k = _key(new_r)
                        if k not in seen:
                            inferred.append(new_r)
                            seen.add(k)
                            new_found = True

                # Rule 4: Intersect + disjoint — A∩B≠∅, B∩C=∅ → ∃x∈A: x∉C
                if r1["type"] == "intersect" and r2["type"] == "disjoint":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "diff_intersect", "args": (r1["args"][0], r2["args"][1])}
                        k = _key(new_r)
                        if k not in seen:
                            inferred.append(new_r)
                            seen.add(k)
                            new_found = True

                # Rule 7: Subset + intersect converse — A∩B≠∅, A⊂C → C∩B≠∅
                if r1["type"] == "intersect" and r2["type"] == "subset":
                    if r1["args"][0] == r2["args"][0]:
                        new_r = {"type": "intersect", "args": (r2["args"][1], r1["args"][1])}
                        k = _key(new_r)
                        if k not in seen:
                            inferred.append(new_r)
                            seen.add(k)
                            new_found = True

        if not new_found:
            break

    return inferred


# ─── Negative Sample Generation ───────────────────────────────────────────────

def augment_relations(relations: List[dict], seed: int = None) -> List[dict]:
    """
    Generate corrupted (negative) relations for contrastive learning.

    Two corruption strategies applied randomly per relation:
      - reverse: swap arguments (A,B) → (B,A)
      - negate:  flip relation type to its logical contradiction

    Args:
        relations: List of {"type": str, "args": (str, str)}
        seed: Optional random seed for reproducibility

    Returns:
        List of corrupted relations (same length as input)
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    augmented = []
    for r in relations:
        aug_r = copy.deepcopy(r)
        operation = rng.choice(["reverse", "negate"])

        if operation == "reverse":
            # Swap arguments — changes meaning for non-symmetric relations
            aug_r["args"] = (r["args"][1], r["args"][0])
        elif operation == "negate":
            # Flip to logically contradictory relation
            if r["type"] == "subset":
                aug_r["type"] = rng.choice(["disjoint", "diff_intersect"])
            elif r["type"] == "intersect":
                aug_r["type"] = "disjoint"
            elif r["type"] == "disjoint":
                aug_r["type"] = "intersect"
            elif r["type"] == "diff_intersect":
                aug_r["type"] = "subset"

        augmented.append(aug_r)
    return augmented


# ─── Verbalization ─────────────────────────────────────────────────────────────

def format_term(term: str, rev_sym_map: dict) -> str:
    """Convert a symbolic term (possibly negated) back to natural language."""
    if is_negated(term):
        base = term[4:-1]
        raw = rev_sym_map.get(base, base)
        return f"non-{raw}"
    return rev_sym_map.get(term, term)


def verbalize(relations: List[dict], rev_sym_map: dict) -> str:
    """
    Convert symbolic relations back to natural language sentences.

    Returns a single string of space-separated verbalized relations.
    """
    sentences = []
    for r in relations:
        A = format_term(r["args"][0], rev_sym_map)
        B = format_term(r["args"][1], rev_sym_map)

        if r["type"] == "subset":
            sentences.append(f"All {A} are {B}.")
        elif r["type"] == "disjoint":
            sentences.append(f"No {A} is {B}.")
        elif r["type"] == "intersect":
            sentences.append(f"Some {A} are {B}.")
        elif r["type"] == "diff_intersect":
            sentences.append(f"Some {A} are not {B}.")

    return " ".join(sentences)
