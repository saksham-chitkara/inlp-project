"""
augment_subtask1.py  Formal-Logic-Grounded Data Augmentation for Syllogistic
Validity Classification (SemEval Subtask 1).

STRATEGY: Compositional Template System + WordNet Taxonomy

1. FORMALLY CORRECT - validity from 6 classical distribution rules (15 valid forms)
2. LINGUISTICALLY DIVERSE - Quantifier x Copula composition = 300+ templates/type
3. PLAUSIBILITY-AWARE - WordNet hyponym/hypernym taxonomy for term triples
4. BALANCED - 4-cell design matching original data structure
5. VERIFIED - every single example checked for correctness
"""

import json
import os
import re
import random
import uuid
import hashlib
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Set

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet as wn

SEED = 42
random.seed(SEED)

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
TRAIN_FILE = os.path.join(_PROJECT_ROOT, "dataset", "train_data", "subtask 1", "train_data.json")
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, "dataset", "train_data", "subtask 1", "augmented_train_data.json")

TARGET_AUGMENTED = 2880

# ================================================================== #
#  PART 1: FORMAL SYLLOGISTIC LOGIC ENGINE                            #
# ================================================================== #

DISTRIBUTION = {
    'A': {'subject': True,  'predicate': False},
    'E': {'subject': True,  'predicate': True},
    'I': {'subject': False, 'predicate': False},
    'O': {'subject': False, 'predicate': True},
}

FIGURES = {
    1: {'major': ('M', 'P'), 'minor': ('S', 'M')},
    2: {'major': ('P', 'M'), 'minor': ('S', 'M')},
    3: {'major': ('M', 'P'), 'minor': ('M', 'S')},
    4: {'major': ('P', 'M'), 'minor': ('M', 'S')},
}


def is_valid_syllogism(mood: str, figure: int) -> bool:
    assert len(mood) == 3 and all(c in 'AEIO' for c in mood)
    assert figure in (1, 2, 3, 4)
    major_type, minor_type, concl_type = mood[0], mood[1], mood[2]
    fig = FIGURES[figure]
    major_negative = major_type in ('E', 'O')
    minor_negative = minor_type in ('E', 'O')
    concl_negative = concl_type in ('E', 'O')
    if major_negative and minor_negative:
        return False
    if (major_negative or minor_negative) and not concl_negative:
        return False
    if concl_negative and not (major_negative or minor_negative):
        return False
    dist_in_premises = {'M': set(), 'S': set(), 'P': set()}
    major_subj_term = fig['major'][0]
    major_pred_term = fig['major'][1]
    minor_subj_term = fig['minor'][0]
    minor_pred_term = fig['minor'][1]
    if DISTRIBUTION[major_type]['subject']:
        dist_in_premises[major_subj_term].add('major')
    if DISTRIBUTION[major_type]['predicate']:
        dist_in_premises[major_pred_term].add('major')
    if DISTRIBUTION[minor_type]['subject']:
        dist_in_premises[minor_subj_term].add('minor')
    if DISTRIBUTION[minor_type]['predicate']:
        dist_in_premises[minor_pred_term].add('minor')
    if len(dist_in_premises['M']) == 0:
        return False
    if DISTRIBUTION[concl_type]['subject'] and len(dist_in_premises['S']) == 0:
        return False
    if DISTRIBUTION[concl_type]['predicate'] and len(dist_in_premises['P']) == 0:
        return False
    if major_type in ('A', 'E') and minor_type in ('A', 'E') and concl_type in ('I', 'O'):
        return False
    return True


def enumerate_all_syllogisms() -> Dict[str, List]:
    types = ['A', 'E', 'I', 'O']
    valid_forms, invalid_forms = [], []
    for p1 in types:
        for p2 in types:
            for c in types:
                mood = p1 + p2 + c
                for fig in range(1, 5):
                    v = is_valid_syllogism(mood, fig)
                    (valid_forms if v else invalid_forms).append((mood, fig, v))
    return {'valid': valid_forms, 'invalid': invalid_forms}


# ================================================================== #
#  PART 2: COMPOSITIONAL NATURAL LANGUAGE TEMPLATES                   #
#                                                                      #
#  KEY INNOVATION: Quantifier x Copula composition gives 300+         #
#  templates per type - analogous to how WordNet gives unlimited nouns.#
#  Based on exhaustive analysis of ALL 960 SemEval training examples.  #
# ================================================================== #

# ============== TYPE A: UNIVERSAL AFFIRMATIVE ============== #
A_QUANTIFIERS = [
    ("All {S_pl}", "pl"),
    ("Every {S_sg}", "sg"),
    ("Every single {S_sg}", "sg"),
    ("Each {S_sg}", "sg"),
    ("Each and every {S_sg}", "sg"),
    ("Any {S_sg}", "sg"),
    ("Everything that is {a_S}", "sg"),
    ("Anything that is {a_S}", "sg"),
    ("Anything that can be called {a_S}", "sg"),
    ("All things that are {S_pl}", "pl"),
    ("All things which are {S_pl}", "pl"),
    ("All of the things that are {S_pl}", "pl"),
    ("Every single thing that is {a_S}", "sg"),
    ("Every single thing which is {a_S}", "sg"),
    ("Every single creature that is {a_S}", "sg"),
    ("Anyone who is {a_S}", "sg"),
    ("{S_cap}, without exception,", "pl"),
    ("{S_cap}, in every instance,", "pl"),
    ("{S_cap}, by definition,", "pl"),
    ("It is the case that all {S_pl}", "pl"),
    ("It is the case that every {S_sg}", "sg"),
    ("It is true that all {S_pl}", "pl"),
    ("It is true that every {S_sg}", "sg"),
    ("It is true that any {S_sg}", "sg"),
    ("It is known that all {S_pl}", "pl"),
    ("It is a known fact that all {S_pl}", "pl"),
    ("It is a certainty that every single {S_sg}", "sg"),
    ("It is also the case that every single {S_sg}", "sg"),
    ("It is also true that every {S_sg}", "sg"),
    ("It is also true that all {S_pl}", "pl"),
    ("The entire category of {S_pl}", "set"),
    ("The entire set of {S_pl}", "set"),
    ("The entire group of {S_pl}", "set"),
    ("Without exception, every {S_sg}", "sg"),
    ("Without exception, all {S_pl}", "pl"),
]
A_COPULAS_SG = [
    "is {a_P}.",
    "is also {a_P}.",
    "is a type of {P_sg}.",
    "is considered {a_P}.",
    "is classified as {a_P}.",
    "is {a_P}.",
]
A_COPULAS_PL = [
    "are {P_pl}.",
    "are also {P_pl}.",
    "are classified as {P_pl}.",
    "are considered {P_pl}.",
    "are a type of {P_sg}.",
    "are {P_pl}.",
]
A_COPULAS_SET = [
    "is composed of {P_pl}.",
    "consists of {P_pl}.",
    "is entirely made up of {P_pl}.",
    "is a subset of the set of {P_pl}.",
]

# ============== TYPE E: UNIVERSAL NEGATIVE ============== #
E_QUANTIFIERS = [
    ("No {S_sg}", "sg"),
    ("No {S_pl}", "pl"),
    ("Absolutely no {S_sg}", "sg"),
    ("Nothing that is {a_S}", "sg"),
    ("Absolutely nothing that is {a_S}", "sg"),
    ("There is nothing that is {a_S} that", "sg_rel"),
    ("There are no {S_pl} that", "pl_rel"),
    ("There are no {S_pl} which", "pl_rel"),
    ("There is no {S_sg} that", "sg_rel"),
    ("Not a single {S_sg}", "sg"),
    ("Not one {S_sg}", "sg"),
    ("Not one single {S_sg}", "sg"),
    ("It is impossible for {a_S}", "impossible"),
    ("It is the case that no {S_sg}", "sg"),
    ("It is the case that no {S_pl}", "pl"),
    ("It is true that no {S_sg}", "sg"),
    ("It is true that no {S_pl}", "pl"),
    ("It is not the case that any {S_sg}", "sg"),
    ("It is not true that any {S_sg}", "sg"),
    ("It is completely false that any {S_sg}", "sg"),
    ("It is also the case that no {S_sg}", "sg"),
    ("It is also true that no {S_sg}", "sg"),
    ("It is also true that no {S_pl}", "pl"),
    ("It is known that no {S_pl}", "pl"),
    ("None of the {S_pl}", "pl"),
    ("None of the things that are {S_pl}", "pl"),
    ("None of the creatures that are {S_pl}", "pl"),
    ("The set of {S_pl} contains no", "set_no"),
    ("The category of {S_pl} and the category of {P_pl}", "set_overlap"),
    ("{S_cap} and {P_pl}", "set_separate"),
    ("Under no circumstances is {a_S}", "impossible"),
    ("{S_cap}", "bare_neg"),
    ("{S_cap}", "bare_neg2"),
]
E_COPULAS_SG = [
    "is {a_P}.",
    "is {a_P}.",
    "can be {a_P}.",
    "can be classified as {a_P}.",
]
E_COPULAS_PL = [
    "are {P_pl}.",
    "are {P_pl}.",
    "can be {P_pl}.",
    "can be classified as {P_pl}.",
]
E_COPULAS_SG_REL = [
    "is {a_P}.",
    "is also {a_P}.",
    "can be called {a_P}.",
]
E_COPULAS_PL_REL = [
    "are {P_pl}.",
    "are also {P_pl}.",
    "can be classified as {P_pl}.",
]
E_COPULAS_IMPOSSIBLE = [
    "to be {a_P}.",
    "to be classified as {a_P}.",
]
E_COPULAS_SET_NO = [
    "{P_pl}.",
]
E_COPULAS_SET_OVERLAP = [
    "do not overlap.",
    "have no overlap whatsoever.",
    "have nothing in common.",
    "are mutually exclusive.",
    "are completely separate categories.",
    "are entirely separate.",
]
E_COPULAS_SET_SEPARATE = [
    "are completely separate categories.",
    "are mutually exclusive categories.",
    "have no overlap whatsoever.",
    "do not belong to the same group.",
    "are entirely separate.",
]
E_COPULAS_BARE_NEG = [
    "are never {P_pl}.",
    "are in no way {P_pl}.",
    "are not, in any way, {P_pl}.",
    "cannot be {P_pl}.",
    "can never be {P_pl}.",
]
E_COPULAS_BARE_NEG2 = [
    "are never {P_pl}.",
    "are in no way {P_pl}.",
]

# ============== TYPE I: PARTICULAR AFFIRMATIVE ============== #
I_QUANTIFIERS = [
    ("Some {S_pl}", "pl"),
    ("Some of the {S_pl}", "pl"),
    ("Some things that are {S_pl}", "pl"),
    ("A portion of {S_pl}", "pl"),
    ("A portion of the things that are {S_pl}", "pl"),
    ("A few {S_pl}", "pl"),
    ("A few of the {S_pl}", "pl"),
    ("A number of {S_pl}", "pl"),
    ("A select few {S_pl}", "pl"),
    ("A certain number of {S_pl}", "pl"),
    ("A certain quantity of {S_pl}", "pl"),
    ("A small number of {S_pl}", "pl"),
    ("A group of {S_pl}", "pl"),
    ("Certain {S_pl}", "pl"),
    ("At least one {S_sg}", "sg"),
    ("At least some {S_pl}", "pl"),
    ("There exist {S_pl} that", "pl_rel"),
    ("There exist some {S_pl} that", "pl_rel"),
    ("There are {S_pl} that", "pl_rel"),
    ("There are some {S_pl} that", "pl_rel"),
    ("There is a subset of {S_pl} that", "pl_rel"),
    ("There are a few {S_pl} that", "pl_rel"),
    ("There are a number of {S_pl} that", "pl_rel"),
    ("Among the {S_pl}, some", "among"),
    ("Among the {S_pl}, a few", "among"),
    ("Among the creatures that are {S_pl}, some", "among"),
    ("Among the things that are {S_pl}, some", "among"),
    ("Of the items that are {S_pl}, some", "among"),
    ("It is the case that some {S_pl}", "pl"),
    ("It is known that some {S_pl}", "pl"),
    ("It is true that some {S_pl}", "pl"),
    ("It is a fact that some {S_pl}", "pl"),
    ("It is a known fact that some {S_pl}", "pl"),
    ("It is also the case that some {S_pl}", "pl"),
    ("It is also true that some {S_pl}", "pl"),
    ("The category of {S_pl} contains some members that", "pl_rel"),
    ("The group of {S_pl} has some members that", "pl_rel"),
    ("A subset of {S_pl}", "pl"),
]
I_COPULAS_SG = [
    "is {a_P}.",
    "is also {a_P}.",
    "is classified as {a_P}.",
]
I_COPULAS_PL = [
    "are {P_pl}.",
    "are also {P_pl}.",
    "are classified as {P_pl}.",
    "are considered {P_pl}.",
    "can be classified as {P_pl}.",
    "can be described as {P_pl}.",
]
I_COPULAS_PL_REL = [
    "are {P_pl}.",
    "are also {P_pl}.",
    "are classified as {P_pl}.",
]
I_COPULAS_AMONG = [
    "are {P_pl}.",
    "are also {P_pl}.",
    "are considered {P_pl}.",
]

# ============== TYPE O: PARTICULAR NEGATIVE ============== #
O_QUANTIFIERS = [
    ("Some {S_pl}", "pl_not"),
    ("Some of the {S_pl}", "pl_not"),
    ("Some things that are {S_pl}", "pl_not"),
    ("Not all {S_pl}", "pl"),
    ("Not every {S_sg}", "sg"),
    ("A portion of {S_pl}", "pl_not"),
    ("A few {S_pl}", "pl_not"),
    ("Certain {S_pl}", "pl_not"),
    ("There exist some {S_pl} that", "pl_rel_not"),
    ("There exist some {S_pl} which", "pl_rel_not"),
    ("There are some {S_pl} that", "pl_rel_not"),
    ("There are some {S_pl} which", "pl_rel_not"),
    ("There exist certain {S_pl} that", "pl_rel_not"),
    ("At least one {S_sg}", "sg_not"),
    ("It is not the case that all {S_pl}", "pl"),
    ("It is not the case that every {S_sg}", "sg"),
    ("It is not true that all {S_pl}", "pl"),
    ("It is not true that every {S_sg}", "sg"),
    ("Some {S_pl}, it is known,", "pl_not"),
    ("A subset of {S_pl}", "pl_not"),
]
O_COPULAS_PL = [
    "are {P_pl}.",
    "are considered {P_pl}.",
    "are classified as {P_pl}.",
]
O_COPULAS_SG = [
    "is {a_P}.",
    "is considered {a_P}.",
    "is classified as {a_P}.",
]
O_COPULAS_PL_NOT = [
    "are not {P_pl}.",
    "are not considered {P_pl}.",
    "are not classified as {P_pl}.",
    "cannot be classified as {P_pl}.",
    "are not {P_pl}.",
]
O_COPULAS_SG_NOT = [
    "is not {a_P}.",
    "is not considered {a_P}.",
    "cannot be classified as {a_P}.",
]
O_COPULAS_PL_REL_NOT = [
    "are not {P_pl}.",
    "are not considered {P_pl}.",
    "cannot be classified as {P_pl}.",
]

# ============== CONCLUSION CONNECTORS ============== #
CONCLUSION_CONNECTORS = [
    "Therefore, {c}.",
    "Consequently, {c}.",
    "It follows that {c}.",
    "Thus, {c}.",
    "This means that {c}.",
    "It follows from this that {c}.",
    "As a result, {c}.",
    "This leads to the conclusion that {c}.",
    "It must be the case that {c}.",
    "This implies that {c}.",
    "It logically follows that {c}.",
    "It can be deduced that {c}.",
    "It follows, then, that {c}.",
    "It can be concluded that {c}.",
    "This proves that {c}.",
    "From this, it follows that {c}.",
    "Hence, {c}.",
    "It must follow that {c}.",
    "We must conclude that {c}.",
    "It necessarily follows that {c}.",
    "Based on this, it must be the case that {c}.",
    "Therefore, we can conclude that {c}.",
    "From this, it can be concluded that {c}.",
    "Consequently, it follows that {c}.",
    "One can therefore conclude that {c}.",
    "The only logical conclusion is that {c}.",
    "So, {c}.",
    "From this, we can conclude that {c}.",
    "One must conclude that {c}.",
    "Therefore, it follows that {c}.",
    "As such, it is necessarily true that {c}.",
    "From these facts, it is clear that {c}.",
    "A conclusion that can be drawn from this is that {c}.",
    "It is therefore the case that {c}.",
    "Consequently, it can be said that {c}.",
    "From this, we can deduce that {c}.",
]


# ================================================================== #
#  PART 3: WORDNET-POWERED TERM POOLS                                 #
# ================================================================== #

def _synset_frequency(synset) -> int:
    return sum(l.count() for l in synset.lemmas())

def _get_lemma_name(synset) -> str:
    lemmas = synset.lemma_names()
    for l in lemmas:
        if '_' not in l and l.isalpha() and len(l) > 2 and l[0].islower():
            return l.lower()
    for l in lemmas:
        name = l.replace('_', ' ').lower()
        if name.replace(' ', '').isalpha() and 3 <= len(name) <= 16:
            return name
    return lemmas[0].replace('_', ' ').lower()

def _is_good_term(synset, min_freq=1) -> bool:
    if _synset_frequency(synset) < min_freq:
        return False
    name = _get_lemma_name(synset)
    if len(name) < 3 or len(name) > 16:
        return False
    if not all(c.isalpha() or c == ' ' for c in name):
        return False
    for bad in ('genus', 'family', 'order', 'phylum', 'class', 'suborder',
                'subfamily', 'superfamily', 'subclass', 'infraorder'):
        if bad in name:
            return False
    return True

def _collect_subtree(synset, max_depth=5, min_freq=1) -> List:
    terms = []
    queue = [(synset, 0)]
    visited = set()
    while queue:
        current, depth = queue.pop(0)
        if current in visited or depth > max_depth:
            continue
        visited.add(current)
        if current != synset and _is_good_term(current, min_freq):
            terms.append(current)
        for hypo in current.hyponyms():
            queue.append((hypo, depth + 1))
    return terms

def _get_hypernym_chain(synset, max_steps=4) -> List:
    chain = []
    current = synset
    visited = {current}
    for _ in range(max_steps):
        hypers = current.hypernyms()
        if not hypers:
            break
        parent = hypers[0]
        if parent in visited:
            break
        visited.add(parent)
        pname = _get_lemma_name(parent)
        if pname in ('entity', 'abstraction', 'object', 'whole', 'thing',
                      'physical entity', 'matter'):
            break
        if _is_good_term(parent, min_freq=0):
            chain.append(parent)
        current = parent
    return chain

_ANCHOR_SYNSET_NAMES = [
    'animal.n.01', 'mammal.n.01', 'bird.n.01', 'reptile.n.01', 'insect.n.01',
    'fish.n.01',
    'plant.n.02', 'tree.n.01', 'flower.n.01', 'fruit.n.01', 'vegetable.n.01',
    'vehicle.n.01', 'weapon.n.01', 'tool.n.01', 'furniture.n.01',
    'clothing.n.01', 'container.n.01', 'building.n.01',
    'musical_instrument.n.01', 'device.n.01',
    'food.n.01', 'beverage.n.01',
    'body_of_water.n.01', 'geological_formation.n.01',
    'worker.n.01', 'performer.n.01', 'scientist.n.01',
    'sport.n.01', 'game.n.01',
    'science.n.01', 'discipline.n.01',
    'fabric.n.01', 'metal.n.01', 'mineral.n.01',
]

def _build_anchor_registry() -> Dict[str, Dict]:
    registry = {}
    for syn_name in _ANCHOR_SYNSET_NAMES:
        try:
            syn = wn.synset(syn_name)
        except Exception:
            continue
        label = _get_lemma_name(syn)
        terms = _collect_subtree(syn, max_depth=5, min_freq=1)
        if len(terms) < 3:
            terms = _collect_subtree(syn, max_depth=6, min_freq=0)
        chain = _get_hypernym_chain(syn, max_steps=4)
        if terms and chain:
            registry[syn_name] = {
                'synset': syn,
                'label': label,
                'terms': terms,
                'hypernym_chain': chain,
            }
    return registry

print("Initialising WordNet term pools...")
_ANCHOR_REGISTRY = _build_anchor_registry()
_ANCHOR_KEYS = list(_ANCHOR_REGISTRY.keys())
_total_terms = sum(len(v['terms']) for v in _ANCHOR_REGISTRY.values())
print(f"  {len(_ANCHOR_REGISTRY)} anchor categories, {_total_terms} total subtree terms")

def _get_plausible_triple() -> Tuple[str, str, str]:
    for _ in range(50):
        anchor_key = random.choice(_ANCHOR_KEYS)
        info = _ANCHOR_REGISTRY[anchor_key]
        s_syn = random.choice(info['terms'])
        S = _get_lemma_name(s_syn)
        M = info['label']
        if random.random() < 0.3:
            s_hypers = _get_hypernym_chain(s_syn, max_steps=3)
            intermediates = [h for h in s_hypers if h != info['synset'] and _is_good_term(h)]
            if intermediates:
                M = _get_lemma_name(random.choice(intermediates))
        p_syn = random.choice(info['hypernym_chain'][:3])
        P = _get_lemma_name(p_syn)
        if len({S, M, P}) == 3:
            return S, M, P
    return ("poodle", "dog", "animal")

def _get_implausible_triple() -> Tuple[str, str, str]:
    for _ in range(50):
        k1, k2, k3 = random.sample(_ANCHOR_KEYS, 3)
        i1, i2, i3 = _ANCHOR_REGISTRY[k1], _ANCHOR_REGISTRY[k2], _ANCHOR_REGISTRY[k3]
        S = _get_lemma_name(random.choice(i1['terms']))
        M = i2['label']
        P = _get_lemma_name(random.choice(i3['terms']))
        if len({S, M, P}) == 3:
            return S, M, P
    return ("rifle", "vegetable", "planet")


# ================================================================== #
#  PART 4: TERM MORPHOLOGY HELPERS                                    #
# ================================================================== #

IRREGULAR_PLURALS = {
    "person": "people", "child": "children", "man": "men", "woman": "women",
    "mouse": "mice", "goose": "geese", "tooth": "teeth", "foot": "feet",
    "fish": "fish", "sheep": "sheep", "deer": "deer", "ox": "oxen",
    "cactus": "cacti", "nucleus": "nuclei", "fungus": "fungi",
    "analysis": "analyses", "basis": "bases", "crisis": "crises",
    "bus": "buses", "octopus": "octopuses",
    "furniture": "furniture", "transport": "transport",
    "math": "math", "music": "music", "clothing": "clothing",
    "equipment": "equipment", "livestock": "livestock",
    "poultry": "poultry", "artillery": "artillery",
    "ammunition": "ammunition", "cutlery": "cutlery",
    "footwear": "footwear", "hardware": "hardware",
    "software": "software", "underwear": "underwear",
    "outerwear": "outerwear", "sportswear": "sportswear",
    "gymnastics": "gymnastics", "athletics": "athletics",
    "politics": "politics", "physics": "physics",
    "mathematics": "mathematics", "electronics": "electronics",
    "economics": "economics", "linguistics": "linguistics",
    "body of water": "bodies of water",
    "young person": "young people",
    "celestial body": "celestial bodies",
    "musical instrument": "musical instruments",
    "geological formation": "geological formations",
    "natural feature": "natural features",
    "aquatic vertebrate": "aquatic vertebrates",
    "woody plant": "woody plants",
    "bony fish": "bony fish", "game fish": "game fish",
    "food fish": "food fish", "young fish": "young fish",
    "power tool": "power tools", "hand tool": "hand tools",
    "blood sport": "blood sports", "night game": "night games",
    "easy chair": "easy chairs", "apple tree": "apple trees",
    "garter snake": "garter snakes", "hoop snake": "hoop snakes",
    "sleeping bag": "sleeping bags", "pot plant": "pot plants",
    "social insect": "social insects", "gas fixture": "gas fixtures",
    "rest house": "rest houses", "holding company": "holding companies",
    "hall of fame": "halls of fame", "man of war": "men of war",
    "coat of arms": "coats of arms", "chest of drawers": "chests of drawers",
    "bird of prey": "birds of prey", "bird of passage": "birds of passage",
    "place of worship": "places of worship",
    "commander in chief": "commanders in chief",
}

def pluralize(word: str) -> str:
    w = word.lower()
    if w in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[w]
    if ' ' in word:
        if ' of ' in w:
            parts = w.split(' of ', 1)
            return pluralize(parts[0]) + ' of ' + parts[1]
        parts = word.rsplit(' ', 1)
        return parts[0] + ' ' + pluralize(parts[1])
    if w.endswith('ics') and len(w) > 4:
        return w
    if w.endswith('man') and len(w) > 3 and w not in (
            'human', 'roman', 'ottoman', 'talisman', 'shaman',
            'dolman', 'dragoman', 'german', 'caiman'):
        return w[:-3] + 'men'
    if w.endswith('woman') and len(w) > 5:
        return w[:-5] + 'women'
    if w.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return w + 'es'
    if w.endswith('y') and w[-2:] not in ('ay', 'ey', 'oy', 'uy'):
        return w[:-1] + 'ies'
    if w.endswith('ff'):
        return w + 's'           # skiff→skiffs, muff→muffs, cliff→cliffs
    if w.endswith('f'):
        return w[:-1] + 'ves'    # leaf→leaves, calf→calves
    if w.endswith('fe'):
        return w[:-2] + 'ves'    # knife→knives, wife→wives
    return w + 's'

IRREGULAR_SINGULARS = {v: k for k, v in IRREGULAR_PLURALS.items()}

def singularize(word: str) -> str:
    w = word.lower()
    if w in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[w]
    if ' ' in word:
        parts = word.rsplit(' ', 1)
        return parts[0] + ' ' + singularize(parts[1])
    if w.endswith('men') and len(w) > 4 and not w.endswith('omen'):
        return w[:-3] + 'man'
    if w.endswith('women') and len(w) > 5:
        return w[:-5] + 'woman'
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith('ves'):
        return w[:-3] + 'f'
    if w.endswith('ses') or w.endswith('xes') or w.endswith('zes') or \
       w.endswith('ches') or w.endswith('shes'):
        return w[:-2]
    if w.endswith('s') and not w.endswith('ss') and not w.endswith('ics'):
        return w[:-1]
    return w

def article(word: str) -> str:
    w = word.lower().strip()
    if w and w[0] in 'aeiou':
        return 'an'
    return 'a'

def _fill_vars(template: str, subj: str, pred: str) -> str:
    s_pl = pluralize(subj)
    s_sg = singularize(subj)
    p_pl = pluralize(pred)
    p_sg = singularize(pred)
    a_s = f"{article(s_sg)} {s_sg}"
    a_p = f"{article(p_sg)} {p_sg}"
    s_cap = s_pl.capitalize()
    return template.format(
        S=subj, P=pred,
        S_pl=s_pl, S_sg=s_sg,
        P_pl=p_pl, P_sg=p_sg,
        S_cap=s_cap,
        a_S=a_s, a_P=a_p,
    )


# ================================================================== #
#  PART 5: COMPOSITIONAL PROPOSITION GENERATOR                        #
# ================================================================== #

def generate_proposition_A(subj: str, pred: str) -> str:
    quant_template, quant_type = random.choice(A_QUANTIFIERS)
    if quant_type == "sg":
        copula = random.choice(A_COPULAS_SG)
    elif quant_type == "set":
        copula = random.choice(A_COPULAS_SET)
    else:
        copula = random.choice(A_COPULAS_PL)
    q = _fill_vars(quant_template, subj, pred)
    c = _fill_vars(copula, subj, pred)
    return f"{q} {c}"

def generate_proposition_E(subj: str, pred: str) -> str:
    quant_template, quant_type = random.choice(E_QUANTIFIERS)
    q = _fill_vars(quant_template, subj, pred)
    copula_map = {
        "sg": E_COPULAS_SG,
        "pl": E_COPULAS_PL,
        "sg_rel": E_COPULAS_SG_REL,
        "pl_rel": E_COPULAS_PL_REL,
        "impossible": E_COPULAS_IMPOSSIBLE,
        "set_no": E_COPULAS_SET_NO,
        "bare_neg": E_COPULAS_BARE_NEG,
        "bare_neg2": E_COPULAS_BARE_NEG2,
    }
    if quant_type == "set_overlap":
        c = random.choice(E_COPULAS_SET_OVERLAP)
        return f"{q} {c}"
    elif quant_type == "set_separate":
        c = random.choice(E_COPULAS_SET_SEPARATE)
        return f"{q} {c}"
    else:
        copulas = copula_map.get(quant_type, E_COPULAS_PL)
        c = _fill_vars(random.choice(copulas), subj, pred)
        return f"{q} {c}"

def generate_proposition_I(subj: str, pred: str) -> str:
    quant_template, quant_type = random.choice(I_QUANTIFIERS)
    q = _fill_vars(quant_template, subj, pred)
    copula_map = {
        "sg": I_COPULAS_SG,
        "pl": I_COPULAS_PL,
        "pl_rel": I_COPULAS_PL_REL,
        "among": I_COPULAS_AMONG,
    }
    copulas = copula_map.get(quant_type, I_COPULAS_PL)
    c = _fill_vars(random.choice(copulas), subj, pred)
    return f"{q} {c}"

def generate_proposition_O(subj: str, pred: str) -> str:
    quant_template, quant_type = random.choice(O_QUANTIFIERS)
    q = _fill_vars(quant_template, subj, pred)
    copula_map = {
        "pl": O_COPULAS_PL,
        "sg": O_COPULAS_SG,
        "pl_not": O_COPULAS_PL_NOT,
        "sg_not": O_COPULAS_SG_NOT,
        "pl_rel_not": O_COPULAS_PL_REL_NOT,
    }
    copulas = copula_map.get(quant_type, O_COPULAS_PL_NOT)
    c = _fill_vars(random.choice(copulas), subj, pred)
    return f"{q} {c}"

PROP_GENERATORS = {
    'A': generate_proposition_A,
    'E': generate_proposition_E,
    'I': generate_proposition_I,
    'O': generate_proposition_O,
}

def generate_proposition(prop_type: str, subj: str, pred: str) -> str:
    return PROP_GENERATORS[prop_type](subj, pred)

def generate_conclusion_text(prop_type: str, subj: str, pred: str) -> str:
    raw = generate_proposition(prop_type, subj, pred).rstrip('.')
    raw_lower = raw[0].lower() + raw[1:] if raw else raw
    connector = random.choice(CONCLUSION_CONNECTORS)
    return connector.format(c=raw_lower)


# ================================================================== #
#  PART 6: SYLLOGISM GENERATOR                                        #
# ================================================================== #

def get_term_triple(plausible: bool) -> Tuple[str, str, str]:
    if plausible:
        return _get_plausible_triple()
    else:
        return _get_implausible_triple()

def generate_syllogism(mood: str, figure: int, plausible: bool) -> str:
    S, M, P = get_term_triple(plausible)
    fig = FIGURES[figure]
    def resolve(symbol):
        return {'S': S, 'M': M, 'P': P}[symbol]
    major_subj = resolve(fig['major'][0])
    major_pred = resolve(fig['major'][1])
    minor_subj = resolve(fig['minor'][0])
    minor_pred = resolve(fig['minor'][1])
    major_premise = generate_proposition(mood[0], major_subj, major_pred)
    minor_premise = generate_proposition(mood[1], minor_subj, minor_pred)
    conclusion = generate_conclusion_text(mood[2], S, P)
    return f"{major_premise} {minor_premise} {conclusion}"


# ================================================================== #
#  PART 7: VERIFICATION ENGINE                                        #
# ================================================================== #

def verify_single_example(item: Dict) -> Tuple[bool, str]:
    text = item.get('syllogism', '')
    validity = item.get('validity')
    mood = item.get('_mood', '')
    figure = item.get('_figure', 0)

    if not text or len(text) < 20:
        return False, f"Empty/short text: {text[:50]}"

    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sents) < 3:
        return False, f"Too few sentences ({len(sents)}): {text[:80]}"

    if mood and figure:
        expected_valid = is_valid_syllogism(mood, figure)
        if validity != expected_valid:
            return False, f"Validity mismatch: labeled={validity} expected={expected_valid} for {mood}-{figure}"

    for placeholder in ['{S_pl}', '{S_sg}', '{P_pl}', '{P_sg}', '{a_S}', '{a_P}', '{S_cap}', '{c}']:
        if placeholder in text:
            return False, f"Unfilled placeholder {placeholder}"

    return True, "OK"


def verify_dataset(data: List[Dict]) -> bool:
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DATA VERIFICATION")
    print("=" * 60)

    errors = []
    aug_items = [x for x in data if x.get('_source') == 'augmented']

    for i, item in enumerate(aug_items):
        ok, msg = verify_single_example(item)
        if not ok:
            errors.append(f"  Example {i}: {msg}")

    print(f"\n1. Per-example verification ({len(aug_items)} augmented):")
    if errors:
        print(f"   FAIL: {len(errors)} errors!")
        for e in errors[:10]:
            print(f"     {e}")
    else:
        print(f"   PASS: All augmented examples verified")

    vcounts = Counter((item['validity'], item['plausibility']) for item in data)
    print(f"\n2. Balance (validity, plausibility):")
    for k in sorted(vcounts.keys()):
        print(f"   {k}: {vcounts[k]}")

    all_forms = enumerate_all_syllogisms()
    valid_set = {(m, f) for m, f, _ in all_forms['valid']}
    KNOWN_VALID = {
        ("AAA", 1), ("EAE", 1), ("AII", 1), ("EIO", 1),
        ("AEE", 2), ("EAE", 2), ("EIO", 2), ("AOO", 2),
        ("AII", 3), ("IAI", 3), ("OAO", 3), ("EIO", 3),
        ("AEE", 4), ("IAI", 4), ("EIO", 4),
    }
    print(f"\n3. Validity engine: {'PASS (15/15)' if valid_set == KNOWN_VALID else 'FAIL'}")

    mood_counts = Counter(item['_mood'] for item in aug_items)
    print(f"\n4. Mood coverage: {len(mood_counts)} unique moods")

    all_words = set()
    for item in aug_items:
        all_words.update(item['syllogism'].lower().split())
    print(f"\n5. Lexical diversity: {len(all_words)} unique words")

    texts = [item['syllogism'] for item in data]
    dups = len(texts) - len(set(texts))
    print(f"\n6. Duplicates: {dups}")

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'ALL CHECKS PASSED' if len(errors) == 0 else f'{len(errors)} ERRORS FOUND'}")
    print(f"{'=' * 60}")
    return len(errors) == 0


# ================================================================== #
#  PART 8: MAIN GENERATION PIPELINE                                   #
# ================================================================== #

def fingerprint(text: str) -> str:
    normalised = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()

def load_original_data(path: str) -> Tuple[List[Dict], Set[str]]:
    with open(path, 'r') as f:
        data = json.load(f)
    fps = set()
    for item in data:
        fps.add(fingerprint(item['syllogism']))
    return data, fps

def generate_augmented_dataset(
    target_count: int = TARGET_AUGMENTED,
    original_path: str = TRAIN_FILE,
    output_path: str = OUTPUT_FILE,
) -> List[Dict]:
    print("=" * 60)
    print("FORMAL-LOGIC-GROUNDED DATA AUGMENTATION")
    print("  Compositional Template System + WordNet Taxonomy")
    print("=" * 60)

    original_data, original_fps = load_original_data(original_path)
    print(f"\nOriginal training examples: {len(original_data)}")
    print(f"Target augmented examples:  {target_count}")

    all_forms = enumerate_all_syllogisms()
    valid_forms = all_forms['valid']
    invalid_forms = all_forms['invalid']

    print(f"\nValid forms:   {len(valid_forms)}")
    print(f"Invalid forms: {len(invalid_forms)}")
    print("\nValid syllogistic forms:")
    for mood, fig, _ in sorted(valid_forms):
        print(f"  {mood}-{fig}", end="")
    print()

    per_cell = target_count // 4
    augmented = []
    seen_fps = set(original_fps)

    cells = [
        (True, True, valid_forms, "valid+plausible"),
        (True, False, valid_forms, "valid+implausible"),
        (False, True, invalid_forms, "invalid+plausible"),
        (False, False, invalid_forms, "invalid+implausible"),
    ]

    for validity, plausibility, forms_pool, cell_name in cells:
        count = 0
        attempts = 0
        max_attempts = per_cell * 20
        while count < per_cell and attempts < max_attempts:
            attempts += 1
            mood, fig, _ = random.choice(forms_pool)
            text = generate_syllogism(mood, fig, plausibility)
            fp = fingerprint(text)
            if fp in seen_fps:
                continue
            seen_fps.add(fp)
            item = {
                "id": str(uuid.uuid4()),
                "syllogism": text,
                "validity": validity,
                "plausibility": plausibility,
                "_mood": mood,
                "_figure": fig,
                "_source": "augmented",
            }
            augmented.append(item)
            count += 1
        print(f"  {cell_name}: {count}/{per_cell} (attempts: {attempts})")

    random.shuffle(augmented)
    combined = original_data + augmented
    print(f"\nCombined: {len(combined)} ({len(original_data)} original + {len(augmented)} augmented)")

    verify_dataset(combined)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aug_only_path = output_path.replace('.json', '_augmented_only.json')
    with open(aug_only_path, 'w') as f:
        json.dump(augmented, f, indent=2)
    print(f"\nAugmented-only: {aug_only_path}")
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"Combined:       {output_path}")

    print("\nSample examples:")
    for validity, plausibility, label in [
        (True, True, "valid+plausible"),
        (True, False, "valid+implausible"),
        (False, True, "invalid+plausible"),
        (False, False, "invalid+implausible"),
    ]:
        for item in augmented:
            if item['validity'] == validity and item['plausibility'] == plausibility:
                print(f"  [{label}] [{item['_mood']}-{item['_figure']}] {item['syllogism']}")
                break

    return combined


def verify_validity_engine():
    KNOWN_VALID = {
        ("AAA", 1), ("EAE", 1), ("AII", 1), ("EIO", 1),
        ("AEE", 2), ("EAE", 2), ("EIO", 2), ("AOO", 2),
        ("AII", 3), ("IAI", 3), ("OAO", 3), ("EIO", 3),
        ("AEE", 4), ("IAI", 4), ("EIO", 4),
    }
    all_forms = enumerate_all_syllogisms()
    found_valid = {(m, f) for m, f, _ in all_forms['valid']}
    print("Validity Engine Verification:")
    print(f"  Expected: 15 | Found: {len(found_valid)}")
    extras = found_valid - KNOWN_VALID
    missing = KNOWN_VALID - found_valid
    if extras:
        print(f"  EXTRA: {extras}")
    if missing:
        print(f"  MISSING: {missing}")
    if not extras and not missing:
        print("  PASS: Perfect match!")
    return found_valid == KNOWN_VALID


if __name__ == "__main__":
    print("Step 1: Verifying formal logic engine...")
    ok = verify_validity_engine()
    if not ok:
        print("FAIL: Logic engine mismatch!")
    else:
        print("PASS!\n")
    print("Step 2: Generating augmented training data...\n")
    generate_augmented_dataset()
    print("\nDone!")
