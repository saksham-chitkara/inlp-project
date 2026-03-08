import re
import copy
import random
import spacy

# Initialize Spacy
nlp = spacy.load("en_core_web_sm")

def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    markers = [
        "therefore, ", "consequently, ", "thus, ", "hence, ", "so, ", 
        "it follows that ", "the only logical conclusion is that ", "as a result, ",
        "it is concluded that ", "one must conclude that "
    ]
    for marker in markers:
        if sentence.startswith(marker):
            sentence = sentence[len(marker):]
    return sentence.strip('. ')

def extract_and_encode_entities(sentences):
    """
    Extracts entities (noun chunks), limits them, and maps them to symbols.
    Returns the symbiotically encoded sentences and the mapping to decode them.
    """
    doc = nlp(" ".join(sentences))
    entities = []
    
    for chunk in doc.noun_chunks:
        text = chunk.text.lower()
        words = text.split()
        if not words:
            continue
            
        # Remove determiners at the beginning
        stopwords = {'a', 'an', 'the', 'all', 'some', 'no', 'every', 'any', 'certain'}
        if words[0] in stopwords:
            text = ' '.join(words[1:])
            
        if text and text not in entities and text not in ['it', 'they', 'them', 'he', 'she']:
            entities.append(text)
            
    # Sort entities by length descending to replace larger spans first
    entities = sorted(entities, key=len, reverse=True)
    
    sym_map = {}
    rev_sym_map = {}
    for i, ent in enumerate(entities):
        sym = f"sym_{i}"
        sym_map[ent] = sym
        rev_sym_map[sym] = ent
        
    encoded_sentences = []
    for sent in sentences:
        s = clean_sentence(sent)
        for ent in entities:
             s = s.replace(ent, sym_map[ent])
        encoded_sentences.append(s)
        
    return encoded_sentences, rev_sym_map

PATTERNS = [
    (r"(?:all|every|any)\s+(.*?)\s+(?:that\s+)?(?:are|is)\s+(.*)", "subset"),
    (r"(?:no|nothing that is a|there are no)\s+(.*?)\s+(?:that\s+)?(?:are|is)\s+(.*)", "disjoint"),
    (r"some\s+(.*?)\s+(?:that\s+)?(?:are|is)\s+not\s+(.*)", "diff_intersect"),
    (r"some\s+(.*?)\s+(?:that\s+)?(?:are|is)\s+(.*)", "intersect"),
]

def extract_logic_from_sentence(sentence):
    sentence = sentence.strip('. ')
    for pattern, rel_type in PATTERNS:
        match = re.search(pattern, sentence)
        if match:
            return rel_type, match.group(1).strip(), match.group(2).strip()
    return None, None, None

def extract_relations_from_encoded(encoded_sentences):
    relations = []
    for s in encoded_sentences:
        rel, A, B = extract_logic_from_sentence(s)
        if rel:
            relations.append({"type": rel, "args": (A, B)})
    return relations

def is_negated(term):
    return term.startswith("not(") and term.endswith(")")

def negate_term(term):
    if is_negated(term):
        return term[4:-1]
    return f"not({term})"

def infer_implicit_relations(relations):
    """Apply inference rules to extend context, covering rule set operations."""
    inferred = copy.deepcopy(relations)
    new_relations = True
    
    while new_relations:
        new_relations = False
        current_inferred = copy.deepcopy(inferred)
        
        for r1 in current_inferred:
            if r1["type"] == "subset":
                new_r = {"type": "disjoint", "args": (r1['args'][0], negate_term(r1['args'][1]))}
                if new_r not in inferred:
                    inferred.append(new_r); new_relations = True
            
            # Disjoint is symmetric
            if r1["type"] == "disjoint":
                new_r = {"type": "disjoint", "args": (r1['args'][1], r1['args'][0])}
                if new_r not in inferred:
                    inferred.append(new_r); new_relations = True

            for r2 in current_inferred:
                if r1 == r2: continue
                
                # Rule 1
                if r1["type"] == "subset" and r2["type"] == "subset":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "subset", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True
                            
                # Rule 2
                elif r1["type"] == "subset" and r2["type"] == "disjoint":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "disjoint", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True
                            
                # Rule 3
                elif r1["type"] == "intersect" and r2["type"] == "subset":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "intersect", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True
                            
                # Rule 4
                elif r1["type"] == "intersect" and r2["type"] == "disjoint":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "diff_intersect", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True
                            
    return inferred

def augment_relations(relations):
    """Logically negate a relation for contrastive learning."""
    augmented = []
    for r in relations:
        aug_r = copy.deepcopy(r)
        operation = random.choice(["reverse", "negate"])
        
        if operation == "reverse":
            aug_r["args"] = (r["args"][1], r["args"][0])
        elif operation == "negate":
            if r["type"] == "subset":
                aug_r["type"] = random.choice(["disjoint", "diff_intersect"])
            elif r["type"] == "intersect":
                aug_r["type"] = "disjoint"
            elif r["type"] == "disjoint":
                aug_r["type"] = "intersect"
            elif r["type"] == "diff_intersect":
                aug_r["type"] = "subset"
                
        augmented.append(aug_r)
    return augmented

def format_term(term, rev_sym_map):
    if is_negated(term):
        base = term[4:-1]
        raw = rev_sym_map.get(base, base)
        return f"non-{raw}"
    return rev_sym_map.get(term, term)

def verbalize(relations, rev_sym_map):
    sentences = []
    for r in relations:
        A = format_term(r['args'][0], rev_sym_map)
        B = format_term(r['args'][1], rev_sym_map)
        if r['type'] == 'subset':
            sentences.append(f"All {A} are {B}.")
        elif r['type'] == 'disjoint':
            sentences.append(f"No {A} is {B}.")
        elif r['type'] == 'intersect':
            sentences.append(f"Some {A} are {B}.")
        elif r['type'] == 'diff_intersect':
            sentences.append(f"Some {A} are not {B}.")
    return " ".join(sentences)
