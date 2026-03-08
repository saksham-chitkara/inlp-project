import torch
from tqdm import tqdm
import json

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=10, patience=3):
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids_plus = batch['input_ids_plus'].to(device)
            attention_mask_plus = batch['attention_mask_plus'].to(device)
            input_ids_minus = batch['input_ids_minus'].to(device)
            attention_mask_minus = batch['attention_mask_minus'].to(device)
            labels = batch['label'].to(device)
            
            logits, loss = model(input_ids_plus, attention_mask_plus, input_ids_minus, attention_mask_minus, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids_plus = batch['input_ids_plus'].to(device)
                attention_mask_plus = batch['attention_mask_plus'].to(device)
                labels = batch['label'].to(device)
                
                # We skip contrastive loss during validation (no minus inputs provided)
                logits, loss = model(input_ids_plus, attention_mask_plus, labels=labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_lreasoner_model.pt')
            print("Saved new best model.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break

def evaluate(model, test_dataset, test_dataloader, device, output_pred_path="predictions.json"):
    model.eval()
    predictions = []
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids_plus = batch['input_ids_plus'].to(device)
            attention_mask_plus = batch['attention_mask_plus'].to(device)
            
            logits, _ = model(input_ids_plus, attention_mask_plus)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Map batch predictions back to dataset ids
            for i, p in enumerate(preds):
                predictions.append({
                    "id": batch["id"][i],
                    "validity": bool(p)
                })
                
    with open(output_pred_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_pred_path}.")
