import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class LReasonerModel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", alpha=0.5):
        super(LReasonerModel, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, 2)
        )
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, input_ids_plus, attention_mask_plus, input_ids_minus=None, attention_mask_minus=None, labels=None):
        # Forward pass true pairs
        outputs_plus = self.encoder(input_ids=input_ids_plus, attention_mask=attention_mask_plus)
        pooled_output_plus = outputs_plus.pooler_output
        logits = self.classifier(pooled_output_plus)
        
        loss = None
        if labels is not None:
            loss_ce = self.cross_entropy(logits, labels)
            loss_cl = 0
            
            if input_ids_minus is not None:
                # Forward pass corrupted pairs
                outputs_minus = self.encoder(input_ids=input_ids_minus, attention_mask=attention_mask_minus)
                pooled_output_minus = outputs_minus.pooler_output
                
                # Contrastive: maximize distance between true and false logic embeddings
                cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
                target = torch.full((pooled_output_plus.size(0),), -1).to(pooled_output_plus.device)
                loss_cl = cosine_loss(pooled_output_plus, pooled_output_minus, target)
                
            loss = loss_ce + self.alpha * loss_cl
            
        return logits, loss
