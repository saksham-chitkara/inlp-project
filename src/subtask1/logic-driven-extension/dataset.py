import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from logic_utils import nlp, extract_and_encode_entities, extract_relations_from_encoded, infer_implicit_relations, verbalize, augment_relations

label_map = {True: 1, False: 0}

class SyllogismDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=256):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._process_data()
        
    def _process_data(self):
        processed = []
        for item in tqdm(self.data, desc="Preprocessing Logic/Entities"):
            syllogism = item['syllogism']
            label = label_map[item['validity']]
            
            # Sentence extraction
            doc = nlp(syllogism)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            # Enforce 3 sentences (2 premises, 1 conclusion) heuristically
            if len(sentences) >= 3:
                premises = sentences[:2]
                conclusion = sentences[-1]
            else:
                sentences = [s.strip() + "." for s in syllogism.split('.') if s.strip()]
                if len(sentences) >= 3:
                    premises = sentences[:2]
                    conclusion = sentences[-1]
                else:
                    premises = [syllogism]
                    conclusion = ""
                    
            # Entity extraction & Symbiolic assignment
            encoded_sentences, rev_sym_map = extract_and_encode_entities(premises + [conclusion])
            encoded_premises = encoded_sentences[:-1]
            
            # Logical Inference
            relations = extract_relations_from_encoded(encoded_premises)
            extended_relations = infer_implicit_relations(relations)
            extended_context_str = verbalize(extended_relations, rev_sym_map)
            
            c_plus = f"{' '.join(premises)} {extended_context_str}"
            
            # Data Augmentation (for Contrastive Loss)
            neg_relations = augment_relations(extended_relations)
            c_minus_str = verbalize(neg_relations, rev_sym_map)
            c_minus = f"{' '.join(premises)} {c_minus_str}"
            
            processed.append({
                'id': item['id'],
                'c_plus': c_plus,
                'c_minus': c_minus,
                'conclusion': conclusion,
                'label': label
            })
        return processed

    def __len__(self):
        return len(self.processed_data)
        
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        enc_plus = self.tokenizer(
            item['c_plus'], item['conclusion'],
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        
        enc_minus = self.tokenizer(
            item['c_minus'], item['conclusion'],
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        
        return {
            'id': item['id'],
            'input_ids_plus': enc_plus['input_ids'].squeeze(),
            'attention_mask_plus': enc_plus['attention_mask'].squeeze(),
            'input_ids_minus': enc_minus['input_ids'].squeeze(),
            'attention_mask_minus': enc_minus['attention_mask'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }
