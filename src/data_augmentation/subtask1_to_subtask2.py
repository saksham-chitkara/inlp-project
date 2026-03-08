"""
subtask1_to_subtask2.py -- Augment Subtask 1 data to Subtask 2 format.

Transforms train_data.json (simple 2-3 sentence syllogisms) into the richer
format of test_data_subtask_2.json, where each sample has 5-8 statements
(original premises + thematically coherent distractors) followed by a
conclusion, with relevant_premises tracking the 0-indexed positions of real
premises among the shuffled statements.

Usage:
    python subtask1_to_subtask2.py
"""

import json
import random
import re
import os
import sys
import subprocess

# ------------------------------------------------------------------ #
#  Ensure dependencies                                                #
# ------------------------------------------------------------------ #
try:
    import spacy
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "spacy", "--quiet"]
    )
    import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download",
         "en_core_web_sm", "--quiet"]
    )
    nlp = spacy.load("en_core_web_sm")

import nltk
try:
    from nltk.corpus import wordnet as wn
    _ = wn.synsets("dog")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn

# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
INPUT_FILE = os.path.join(_PROJECT_ROOT, "train_data.json")
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, "train_data_subtask_2_format.json")
SEED = 42
random.seed(SEED)


# ================================================================== #
#  Conclusion-indicator detection                                     #
# ================================================================== #

_CONCLUSION_PHRASES = [
    r"therefore",
    r"thus",
    r"consequently",
    r"hence",
    r"it follows",
    r"this means",
    r"this implies",
    r"this proves",
    r"this leads",
    r"this demonstrates",
    r"this necessitates",
    r"from this",
    r"based on this",
    r"as such",
    r"as a result",
    r"for this reason",
    r"one must conclude",
    r"we can (?:infer|conclude|deduce|say|state)",
    r"we can thus conclude",
    r"it (?:can be|must be|is) (?:necessarily )?(?:concluded|true|the case|logically)",
    r"the (?:only )?(?:logical )?conclusion",
    r"a valid inference",
    r"it (?:logically |therefore |necessarily )?(?:follows|concluded|true|necessary)",
    r"it (?:must|can) (?:follow|be (?:concluded|inferred|deduced))",
    r"it is (?:suggested|implied|stated|asserted|necessarily true|necessarily concluded)",
    r"the result of this",
    r"a logical consequence",
    r"it logically follows",
    r"it therefore",
    r"it is necessarily true",
    r"it is necessarily concluded",
    r"it can be necessarily concluded",
    r"it must be true",
    r"it must be the case",
    r"it must follow",
    r"it is therefore (?:true|the case|a fact)",
    r"it is logically (?:concluded|necessary)",
    r"it can be (?:said|deduced|concluded|logically inferred)",
    r"it follows from this",
    r"it has been asserted",
    r"it is suggested",
    r"the conclusion .* is inescapable",
    r"it can be logically inferred",
    r"it is therefore true",
    r"it logically follows that",
    r"it follows that",
    r"it follows directly",
    r"it follows logically",
    r"it is logically concluded",
    r"it must be true that",
    r"consequently,?\s",
    r"it is the case that",
    r"it is necessarily true that",
    r"it must be the case that a portion",
    r"it follows from this that",
]

_CONCLUSION_RE = re.compile(
    r"^(?:" + "|".join(_CONCLUSION_PHRASES) + r")",
    re.IGNORECASE,
)


def _is_conclusion(sentence):
    return bool(_CONCLUSION_RE.search(sentence.strip()))


def split_syllogism(text):
    """Split a syllogism into (premises_list, conclusion_string)."""
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]

    if len(sents) < 2:
        parts = [p.strip() for p in re.split(r"\.\s+", text) if p.strip()]
        sents = [(p if p.endswith(".") else p + ".") for p in parts]

    if len(sents) < 2:
        return [text], text

    conclusion_idx = len(sents) - 1
    for i in range(1, len(sents)):
        if _is_conclusion(sents[i]):
            conclusion_idx = i
            break

    premises = sents[:conclusion_idx]
    conclusion = " ".join(sents[conclusion_idx:])

    if not premises:
        premises = sents[:-1]
        conclusion = sents[-1]

    return premises, conclusion


# ================================================================== #
#  Morphological helpers                                              #
# ================================================================== #

def _a_an(word):
    w = word.strip()
    return "an" if w and w[0].lower() in "aeiou" else "a"


_IRREG_S2P = {
    "mouse": "mice", "man": "men", "woman": "women", "child": "children",
    "person": "people", "tooth": "teeth", "foot": "feet", "goose": "geese",
    "cactus": "cacti", "fungus": "fungi", "nebula": "nebulae",
    "larva": "larvae", "vertebra": "vertebrae", "criterion": "criteria",
    "bus": "buses", "fox": "foxes", "ox": "oxen",
}
_IRREG_P2S = {v: k for k, v in _IRREG_S2P.items()}


def _singularize(w):
    w = w.strip().lower()
    if w in _IRREG_P2S:
        return _IRREG_P2S[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
    for suf in ("ses", "xes", "zes", "ches", "shes"):
        if w.endswith(suf):
            return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _pluralize(w):
    w = w.strip().lower()
    if w in _IRREG_S2P:
        return _IRREG_S2P[w]
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    if w.endswith("f"):
        return w[:-1] + "ves"
    if w.endswith("fe"):
        return w[:-2] + "ves"
    return w + "s"


# ================================================================== #
#  Domain detection                                                   #
# ================================================================== #

DOMAIN_KEYWORDS = {
    "animal": [
        "dog", "cat", "bird", "fish", "mammal", "animal", "reptile",
        "feline", "canine", "lion", "tiger", "elephant", "whale", "bear",
        "horse", "cow", "pig", "sheep", "wolf", "snake", "spider",
        "shark", "dolphin", "monkey", "rabbit", "mouse", "deer", "fox",
        "frog", "penguin", "parrot", "eagle", "sparrow", "pet",
        "creature", "poodle", "retriever", "terrier", "kitten", "puppy",
        "hound", "cheetah", "panther", "leopard", "jaguar", "cougar",
        "lynx", "gorilla", "chimpanzee", "ape", "primate", "amphibian",
        "vertebrate", "carnivore", "herbivore", "predator", "beetle",
        "ant", "bee", "wasp", "butterfly", "moth", "crocodile",
        "lizard", "turtle", "tuna", "salmon", "trout", "octopus",
        "hamster", "bat", "owl", "hawk", "robin", "collie", "spaniel",
        "bulldog", "husky", "persian",
    ],
    "plant": [
        "tree", "plant", "flower", "grass", "fern", "moss", "bush",
        "vine", "oak", "pine", "rose", "daisy", "tulip", "lily",
        "sunflower", "cactus", "herb", "shrub", "leaf", "root",
        "branch", "seed", "blossom", "photosynthetic", "maple", "birch",
        "palm", "weed",
    ],
    "food": [
        "fruit", "vegetable", "food", "apple", "orange", "banana",
        "grape", "lemon", "lime", "berry", "carrot", "potato", "tomato",
        "onion", "pepper", "broccoli", "lettuce", "cabbage", "pea",
        "bean", "spinach", "kale", "radish", "turnip", "beet", "celery",
        "cucumber", "squash", "peach", "pear", "cherry", "melon",
        "edible", "nutritious", "citrus", "salad", "mango",
    ],
    "vehicle": [
        "car", "truck", "bus", "bicycle", "motorcycle", "airplane",
        "plane", "boat", "ship", "train", "vehicle", "scooter", "van",
        "jeep", "SUV", "automobile", "helicopter", "submarine",
        "tricycle",
    ],
    "person": [
        "human", "person", "man", "woman", "child", "adult", "baby",
        "parent", "mortal", "bachelor", "husband", "wife", "boy",
        "girl", "toddler", "teenager", "grandparent", "student",
        "teacher", "citizen", "member", "individual", "roman", "greek",
    ],
    "shape": [
        "triangle", "circle", "square", "rectangle", "polygon", "shape",
        "line", "point", "hexagon", "pentagon", "octagon", "cube",
        "trapezoid", "parallelogram", "quadrilateral",
    ],
    "celestial": [
        "star", "planet", "sun", "moon", "comet", "asteroid", "galaxy",
        "nebula", "constellation", "meteor", "satellite", "celestial",
        "orbit", "earth", "mars", "venus", "jupiter", "saturn",
    ],
    "building": [
        "house", "building", "school", "church", "castle", "tower",
        "bridge", "wall", "roof", "door", "window", "floor", "room",
        "apartment", "cottage", "palace", "shed", "skyscraper",
        "structure", "cathedral", "office", "barn", "hut",
    ],
    "music": [
        "piano", "guitar", "violin", "drum", "flute", "trumpet",
        "cello", "harp", "banjo", "instrument", "musical", "song",
        "melody", "rhythm", "symphony", "orchestra", "band", "musician",
        "singer", "note", "opera", "concert",
    ],
    "profession": [
        "doctor", "lawyer", "teacher", "engineer", "nurse", "scientist",
        "professor", "dentist", "surgeon", "pharmacist", "accountant",
        "architect", "pilot", "artist", "writer", "chef", "manager",
        "professional", "consultant", "researcher", "plumber",
        "electrician", "carpenter",
    ],
    "clothing": [
        "shirt", "pants", "dress", "jacket", "coat", "hat", "shoe",
        "glove", "sock", "scarf", "belt", "tie", "garment", "clothing",
        "sweater", "blouse", "skirt",
    ],
    "tool": [
        "hammer", "saw", "drill", "screwdriver", "wrench", "tool",
        "nail", "screw", "pliers", "chisel", "axe", "shovel", "toolbox",
    ],
    "body_water": [
        "river", "lake", "ocean", "sea", "stream", "pond", "waterfall",
        "bay", "creek", "body of water", "coastline",
    ],
    "writing": [
        "pen", "pencil", "marker", "crayon", "chalk", "eraser",
        "writing", "ink", "stationery",
    ],
    "furniture": [
        "chair", "table", "desk", "sofa", "bed", "shelf", "cabinet",
        "furniture", "couch", "stool", "bench", "lamp", "rug",
        "recliner", "beanbag",
    ],
    "insect": [
        "insect", "beetle", "ant", "bee", "wasp", "grasshopper",
        "cricket", "dragonfly", "exoskeleton", "arthropod",
    ],
    "book": [
        "book", "novel", "magazine", "newspaper", "journal", "library",
        "story", "fiction", "biography", "textbook", "chapter", "page",
        "author", "literature", "publication", "encyclopedia",
    ],
    "beverage": [
        "tea", "coffee", "soda", "juice", "milk", "water", "beverage",
        "drink", "wine", "beer", "liquid",
    ],
    "geology": [
        "rock", "stone", "mineral", "crystal", "diamond", "gem",
        "pebble", "boulder", "granite", "marble", "gravel", "sand",
        "jewel", "sapphire",
    ],
    "tech": [
        "computer", "laptop", "phone", "tablet", "screen", "keyboard",
        "printer", "server", "smartphone", "monitor", "device",
        "electronic", "hardware", "software", "microchip", "cable",
    ],
    "sports": [
        "athlete", "runner", "swimmer", "player", "footballer",
        "gymnast", "tennis", "basketball", "sport", "game",
    ],
    "government": [
        "democracy", "republic", "monarchy", "government", "president",
        "king", "queen", "parliament", "dictatorship", "anarchy",
        "oligarchy", "federation",
    ],
}


def detect_domain(text):
    text_lower = text.lower()
    best, best_score = "animal", 0
    for domain, kws in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text_lower)
        if score > best_score:
            best, best_score = domain, score
    return best


# ================================================================== #
#  Domain entity / predicate pools                                    #
# ================================================================== #

DOMAIN_ENTITIES = {
    "animal": [
        "dogs", "cats", "birds", "fish", "mammals", "reptiles",
        "felines", "canines", "lions", "tigers", "elephants", "whales",
        "bears", "horses", "cows", "wolves", "snakes", "sharks",
        "dolphins", "rabbits", "foxes", "penguins", "eagles", "parrots",
        "owls", "hawks", "sparrows", "deer", "mice", "frogs", "turtles",
        "crocodiles", "lizards", "bats", "monkeys", "cheetahs",
    ],
    "plant": [
        "trees", "flowers", "ferns", "mosses", "bushes", "vines",
        "oaks", "pines", "roses", "daisies", "tulips", "lilies",
        "sunflowers", "cacti", "herbs", "shrubs", "grasses", "weeds",
        "palms", "maples", "birches",
    ],
    "food": [
        "fruits", "vegetables", "apples", "oranges", "bananas",
        "grapes", "lemons", "limes", "berries", "carrots", "potatoes",
        "tomatoes", "onions", "peppers", "beans", "peas", "radishes",
        "turnips", "beets", "peaches", "pears", "cherries", "melons",
        "mangoes",
    ],
    "vehicle": [
        "cars", "trucks", "buses", "bicycles", "motorcycles",
        "airplanes", "boats", "ships", "trains", "scooters", "vans",
        "jeeps", "SUVs", "helicopters", "submarines", "tricycles",
    ],
    "person": [
        "humans", "people", "men", "women", "children", "adults",
        "babies", "parents", "bachelors", "husbands", "wives", "boys",
        "girls", "toddlers", "teenagers", "grandparents", "students",
        "citizens",
    ],
    "shape": [
        "triangles", "circles", "squares", "rectangles", "polygons",
        "lines", "points", "hexagons", "pentagons", "octagons",
        "trapezoids", "parallelograms", "ovals", "cubes",
    ],
    "celestial": [
        "stars", "planets", "suns", "moons", "comets", "asteroids",
        "galaxies", "nebulae", "meteors", "satellites",
        "constellations",
    ],
    "building": [
        "houses", "buildings", "schools", "churches", "castles",
        "towers", "bridges", "apartments", "cottages", "palaces",
        "sheds", "skyscrapers", "cathedrals", "offices", "barns",
    ],
    "music": [
        "pianos", "guitars", "violins", "drums", "flutes", "trumpets",
        "cellos", "harps", "banjos", "instruments", "songs", "melodies",
        "symphonies",
    ],
    "profession": [
        "doctors", "lawyers", "teachers", "engineers", "nurses",
        "scientists", "professors", "dentists", "surgeons",
        "accountants", "architects", "pilots", "artists", "writers",
        "chefs", "managers", "consultants", "researchers",
    ],
    "clothing": [
        "shirts", "pants", "dresses", "jackets", "coats", "hats",
        "shoes", "gloves", "socks", "scarves", "belts", "ties",
        "sweaters",
    ],
    "tool": [
        "hammers", "saws", "drills", "screwdrivers", "wrenches",
        "nails", "screws", "pliers", "chisels", "axes", "shovels",
    ],
    "body_water": [
        "rivers", "lakes", "oceans", "seas", "streams", "ponds",
        "waterfalls", "bays", "creeks",
    ],
    "writing": [
        "pens", "pencils", "markers", "crayons", "chalks", "erasers",
    ],
    "furniture": [
        "chairs", "tables", "desks", "sofas", "beds", "shelves",
        "cabinets", "couches", "stools", "benches", "lamps", "rugs",
        "recliners",
    ],
    "insect": [
        "insects", "beetles", "ants", "bees", "wasps", "flies",
        "butterflies", "moths", "grasshoppers", "crickets",
        "dragonflies", "spiders",
    ],
    "book": [
        "books", "novels", "magazines", "newspapers", "journals",
        "stories", "biographies", "textbooks", "encyclopedias",
        "comics",
    ],
    "beverage": [
        "teas", "coffees", "sodas", "juices", "beverages", "drinks",
        "wines", "beers", "liquids", "waters",
    ],
    "geology": [
        "rocks", "stones", "minerals", "crystals", "diamonds", "gems",
        "pebbles", "boulders", "jewels", "sapphires",
    ],
    "tech": [
        "computers", "laptops", "phones", "tablets", "screens",
        "keyboards", "printers", "servers", "smartphones", "monitors",
        "devices", "microchips", "cables",
    ],
    "sports": [
        "athletes", "runners", "swimmers", "players", "footballers",
        "gymnasts", "teams",
    ],
    "government": [
        "democracies", "republics", "monarchies", "governments",
        "dictatorships", "federations",
    ],
}

PLAUSIBLE_PREDICATES = {
    "animal": [
        "a mammal", "a vertebrate", "a creature", "a living thing",
        "an organism", "an animal", "a warm-blooded creature",
        "a four-legged animal", "a predator", "a herbivore",
        "a cold-blooded creature", "a wild animal", "a pet",
        "a carnivore",
    ],
    "plant": [
        "a living thing", "an organism", "a green thing",
        "a photosynthetic organism", "a plant", "a perennial",
        "a flowering plant",
    ],
    "food": [
        "edible", "a consumable", "a food item", "nutritious",
        "a food", "a healthy food", "a plant",
    ],
    "vehicle": [
        "a mode of transport", "a machine", "a thing with wheels",
        "an object", "a vehicle", "a motorized vehicle",
    ],
    "person": [
        "a human", "a mortal", "a mammal", "a person",
        "a living thing", "a member of society", "a citizen",
    ],
    "shape": [
        "a polygon", "a geometric figure", "a shape", "a figure",
        "a two-dimensional object", "a closed figure",
    ],
    "celestial": [
        "a celestial body", "an astronomical object",
        "a heavenly body", "an object in space",
    ],
    "building": [
        "a structure", "an edifice", "a construction",
        "a dwelling", "a building",
    ],
    "music": [
        "a musical instrument", "a string instrument",
        "an instrument", "a thing that produces sound",
    ],
    "profession": [
        "a professional", "a person", "a worker",
        "a member of the workforce", "a trained individual",
    ],
    "clothing": [
        "a garment", "a piece of clothing",
        "an article of apparel", "a wearable",
    ],
    "tool": [
        "a utensil", "an implement", "a device", "an instrument",
    ],
    "body_water": [
        "a body of water", "a geographic feature",
        "a natural formation", "a water source",
    ],
    "writing": [
        "a writing utensil", "an implement for writing",
        "a tool", "an instrument",
    ],
    "furniture": [
        "a piece of furniture", "a household item",
        "an object", "a furnishing",
    ],
    "insect": [
        "an arthropod", "a creature", "an invertebrate",
        "an organism", "an animal with six legs",
    ],
    "book": [
        "a written work", "a publication", "a text",
        "a piece of literature", "a book",
    ],
    "beverage": [
        "a liquid", "a drink", "a beverage", "a fluid",
    ],
    "geology": [
        "a mineral", "a natural substance", "a solid",
        "a hard substance",
    ],
    "tech": [
        "a device", "an electronic device", "a machine",
        "a piece of hardware",
    ],
    "sports": [
        "an athlete", "a person", "a competitor",
    ],
    "government": [
        "a governing system", "a political system",
        "a form of governance",
    ],
}

_ABSURD_PREDICATES = [
    "a cloud", "a rock", "a river", "made of wood", "made of glass",
    "made of metal", "made of paper", "made of ice", "made of cheese",
    "a vegetable", "an insect", "a fish", "a bird", "a reptile",
    "a robot", "a ghost", "made of fire", "made of candy",
    "made of solid gold", "a tiny insect", "a cloud in the sky",
    "a piece of wood", "a sheet of ice", "frozen", "invisible",
    "a type of fruit", "a living organism", "a banana",
    "underground", "able to fly", "made of cotton candy",
    "smaller than an ant", "heavier than the sun",
    "a musical instrument", "made of plastic", "made of concrete",
    "a type of car", "a liquid", "made of jelly",
]


def _get_predicates(domain, plausible):
    if plausible:
        return PLAUSIBLE_PREDICATES.get(
            domain, PLAUSIBLE_PREDICATES["animal"]
        )
    pool = list(_ABSURD_PREDICATES)
    other = [d for d in PLAUSIBLE_PREDICATES if d != domain]
    for od in random.sample(other, min(3, len(other))):
        pool.extend(PLAUSIBLE_PREDICATES[od])
    return pool


# ================================================================== #
#  Distractor sentence templates                                      #
# ================================================================== #

_T_A = [
    "Every {s} is {p}.",
    "Every single {s} is {p}.",
    "Any {s} is {p}.",
    "Anything that is {a_s} is {p}.",
    "Anything that is {a_s} is also {p}.",
    "All {sp} are {pp}.",
    "It is a fact that all {sp} are {pp}.",
    "It is true that all {sp} are {pp}.",
    "It is a known fact that all {sp} are {pp}.",
    "It is undeniable that all {sp} are {pp}.",
    "It is also true that all {sp} are {pp}.",
    "All things that are {sp} are {pp}.",
    "Any and all {sp} are also {pp}.",
    "{sp_cap} are, without exception, {pp}.",
    "Every single thing which is {a_s} is {p}.",
    "Any creature that is {a_s} is {p}.",
    "Every single creature that is {a_s} is {p}.",
    "All of the {sp} are {pp}.",
    "The entire set of {sp} is contained within the set of {pp}.",
]

_T_E = [
    "No {s} is {p}.",
    "There are no {sp} that are {pp}.",
    "There is not a single {s} that is {p}.",
    "There are no {sp} which are {pp}.",
    "{a_s_cap} is never {p}.",
    "There is no overlap between {sp} and {pp}.",
    "The set of {sp} contains no {pp}.",
    "Not a single {s} is {p}.",
    "There exist no {sp} that are {pp}.",
    "{sp_cap} and {pp} have no members in common.",
    "{sp_cap} and {pp} are mutually exclusive.",
    "There are no {sp} that are not {pp}.",
    "There is no {s} which is {p}.",
    "No single thing that is {a_s} is {p}.",
]

_T_I = [
    "Some {sp} are {pp}.",
    "A few {sp} are {pp}.",
    "There are some {sp} that are {pp}.",
    "There are many {sp} that are {pp}.",
    "A certain number of {sp} are {pp}.",
    "A portion of {sp} are {pp}.",
    "There is at least {a_s} that is {p}.",
    "Among the {sp}, some are {pp}.",
    "Among the items that are {sp}, some are {pp}.",
    "It is known that some {sp} are {pp}.",
    "Of the items that are {sp}, some are {pp}.",
    "There exist {sp} that are {pp}.",
    "A select few {sp} are {pp}.",
    "Some of the {sp} are {pp}.",
    "There are a few {sp} that are {pp}.",
    "There are {sp} that are {pp}.",
]

_T_O = [
    "Some {sp} are not {pp}.",
    "Not all {sp} are {pp}.",
    "A portion of {sp} are not {pp}.",
    "There are some {sp} which are not {pp}.",
    "Not every {s} is {p}.",
    "Some {sp} fail to be {pp}.",
    "There are {sp} that are not {pp}.",
    "It is not the case that every {s} is {p}.",
]

_TEMPLATE_MAP = {"A": _T_A, "E": _T_E, "I": _T_I, "O": _T_O}


# ================================================================== #
#  Noun extraction                                                    #
# ================================================================== #

def _extract_nouns(text):
    doc = nlp(text)
    nouns = set()
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and len(tok.text) > 2:
            nouns.add(tok.lemma_.lower())
    for chunk in doc.noun_chunks:
        nouns.add(chunk.root.lemma_.lower())
    return nouns


# ================================================================== #
#  Entity selection helpers                                           #
# ================================================================== #

def _get_distractor_entities(domain, exclude, n=12):
    pool = DOMAIN_ENTITIES.get(domain, DOMAIN_ENTITIES["animal"])
    excl = {x.lower() for x in exclude}
    excl |= {_singularize(x).lower() for x in exclude}
    avail = [e for e in pool
             if e.lower() not in excl
             and _singularize(e).lower() not in excl]
    if len(avail) < n:
        for w in list(exclude)[:5]:
            for syn in wn.synsets(w, pos=wn.NOUN)[:2]:
                for rel in syn.hyponyms()[:3] + syn.hypernyms()[:2]:
                    for lem in rel.lemmas()[:2]:
                        name = lem.name().replace("_", " ")
                        if " " not in name and name.lower() not in excl:
                            avail.append(_pluralize(name))
    random.shuffle(avail)
    return avail[:n]


# ================================================================== #
#  Distractor generation                                              #
# ================================================================== #

def _pred_forms(pred):
    """Return (singular_display, plural_display) for a predicate."""
    if pred.startswith("a ") or pred.startswith("an "):
        core = pred.split(" ", 1)[1]
        if " " in core:
            return pred, core
        return pred, _pluralize(core)
    return pred, pred


def generate_distractor(domain, plausible, entities, predicates, used):
    qtype = random.choices(
        ["A", "E", "I", "O"], weights=[0.40, 0.20, 0.25, 0.15]
    )[0]
    template = random.choice(_TEMPLATE_MAP[qtype])

    avail = [e for e in entities if e.lower() not in used]
    if not avail:
        avail = entities[:]
    subj_pl = random.choice(avail)
    used.add(subj_pl.lower())
    subj = _singularize(subj_pl)

    pred_raw = random.choice(predicates)
    p_sing, p_plur = _pred_forms(pred_raw)

    a_s = "{} {}".format(_a_an(subj), subj)

    try:
        sent = template.format(
            s=subj,
            sp=subj_pl,
            sp_cap=subj_pl[0].upper() + subj_pl[1:],
            p=p_sing,
            pp=p_plur,
            a_s=a_s,
            a_s_cap=a_s[0].upper() + a_s[1:],
        )
    except (KeyError, IndexError):
        sent = "Every {} is {}.".format(subj, p_sing)

    return sent[0].upper() + sent[1:]


# ================================================================== #
#  Main augmentation logic                                            #
# ================================================================== #

def augment_entry(entry):
    text = entry["syllogism"]
    validity = entry["validity"]
    plausibility = entry["plausibility"]

    premises, conclusion = split_syllogism(text)
    if not premises:
        premises = [text]

    domain = detect_domain(text)
    orig_nouns = _extract_nouns(text)
    entities = _get_distractor_entities(domain, orig_nouns, n=14)
    predicates = _get_predicates(domain, plausible=plausibility)

    n_prem = len(premises)
    target_total = random.randint(5, 8)
    n_dist = max(3, target_total - n_prem)
    n_dist = min(n_dist, 6)

    used_subjs = set()
    distractors = []
    for _ in range(n_dist):
        d = generate_distractor(
            domain, plausibility, entities, predicates, used_subjs
        )
        distractors.append(d)

    items = (
        [{"text": p.strip(), "is_premise": True} for p in premises]
        + [{"text": d.strip(), "is_premise": False} for d in distractors]
    )
    random.shuffle(items)

    final_stmts = []
    rel_idxs = []
    for i, it in enumerate(items):
        final_stmts.append(it["text"])
        if it["is_premise"]:
            rel_idxs.append(i)

    final_stmts.append(conclusion.strip())

    if not validity:
        rel_idxs = []
    else:
        rel_idxs = sorted(rel_idxs)

    return {
        "id": entry["id"],
        "syllogism": " ".join(final_stmts),
        "validity": validity,
        "plausibility": plausibility,
        "relevant_premises": rel_idxs,
    }


# ================================================================== #
#  Entry point                                                        #
# ================================================================== #

def main():
    if not os.path.exists(INPUT_FILE):
        print("Error: {} not found.".format(INPUT_FILE))
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loaded {} training entries from {}".format(len(data), INPUT_FILE))
    print("Augmenting to subtask-2 format ...")

    augmented = []
    for idx, entry in enumerate(data):
        try:
            augmented.append(augment_entry(entry))
        except Exception as exc:
            print("  Warning: entry {} ({}): {}".format(
                idx, entry.get("id", "?"), exc))
            augmented.append({
                "id": entry["id"],
                "syllogism": entry["syllogism"],
                "validity": entry["validity"],
                "plausibility": entry["plausibility"],
                "relevant_premises": [],
            })
        if (idx + 1) % 100 == 0:
            print("  {:>5} / {}".format(idx + 1, len(data)))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=4, ensure_ascii=False)

    print("\nDone! {} entries written to {}".format(len(augmented), OUTPUT_FILE))

    # Quick quality report
    valid_with_prem = sum(
        1 for e in augmented if e["validity"] and e["relevant_premises"]
    )
    invalid_empty = sum(
        1 for e in augmented
        if not e["validity"] and e["relevant_premises"] == []
    )
    avg_sents = sum(
        len(e["syllogism"].split(". ")) for e in augmented
    ) / max(len(augmented), 1)

    print("  Valid entries with relevant_premises filled : {}".format(
        valid_with_prem))
    print("  Invalid entries with empty relevant_premises: {}".format(
        invalid_empty))
    print("  Average sentence count per sample           : {:.1f}".format(
        avg_sents))

    # Show 5 samples
    print("\n-- Sample outputs --")
    for s in augmented[:5]:
        print("\n  ID          : {}".format(s["id"]))
        print("  Validity    : {}".format(s["validity"]))
        print("  Plausibility: {}".format(s["plausibility"]))
        print("  Rel. prem.  : {}".format(s["relevant_premises"]))
        syl = s["syllogism"]
        if len(syl) > 300:
            syl = syl[:300] + "..."
        print("  Syllogism   : {}".format(syl))
        print("  " + "-" * 58)


if __name__ == "__main__":
    main()
