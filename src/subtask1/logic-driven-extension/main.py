import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import XLMRobertaTokenizer, get_linear_schedule_with_warmup

from dataset import SyllogismDataset
from model import LReasonerModel
from trainer import train, evaluate

# Import evaluation function
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from evaluation_script import run_full_scoring
except ImportError:
    print("Warning: evaluation_script.py not found. Evaluation will fail to execute properly.")
    run_full_scoring = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../DataAugmentation/train_data.json")
    parser.add_argument("--val_data", type=str, default="../DataAugmentation/train_data.json")
    parser.add_argument("--test_data", type=str, default="test_data_subtask_1.json") 
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    print("Initializing model...")
    model = LReasonerModel().to(device)
    
    if args.do_train:
        print("Loading datasets for Training...")
        train_dataset = SyllogismDataset(args.train_data, tokenizer)
        val_dataset = SyllogismDataset(args.val_data, tokenizer)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        
        print("Starting training framework...")
        train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=args.epochs)
    
    if args.do_eval:
        if os.path.exists('best_lreasoner_model.pt'):
            model.load_state_dict(torch.load('best_lreasoner_model.pt'))
            print("Loaded trained model for evaluation.")
            
        print("Loading test dataset...")
        test_dataset = SyllogismDataset(args.test_data, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        pred_path = "predictions_subtask_1.json"
        
        evaluate(model, test_dataset, test_dataloader, device, output_pred_path=pred_path)
        
        if run_full_scoring is not None:
            output_metrics = "metrics_subtask_1.json"
            print("Evaluating via evaluation_script...")
            run_full_scoring(args.test_data, pred_path, output_metrics)
