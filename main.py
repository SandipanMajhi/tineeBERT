import torch
import torch.nn as nn
from Models.bert import TineeBERT
from CustomBPE.Model.Tokenizer import AutoTokenizer, MLMTokenizer

from utils.loader import MaskedData
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd

from tqdm import tqdm

torch.cuda.empty_cache()

def lr_lambda(current_step, num_warmup_steps = 10000, num_training_steps = 1000000):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        return max(0.0, float(num_training_steps - current_step) /
                        float(max(1, num_training_steps - num_warmup_steps)))

if __name__ == "__main__":
    checkpoint_path = "Checkpoints"
    optimizer_path = "Optimizer_States"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_steps = 1000000
    warmup_steps = 10000

    textloader, vocab = MaskedData.create_dataloader(corpus_path="BookCorpus/books.pkl", batch_size=256, mask_rate=0.15, max_tokens=512)
    bert_model = TineeBERT(num_repeats=10, vocab_size=len(vocab), embed_size=768, seqlen=512, num_heads=16)
    optimizer = torch.optim.Adam(bert_model.parameters(), lr = 1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    bert_model = bert_model.to(device)

    training_history = {
        "loss" : []
    }

    for step in tqdm(range(training_steps)):

        steploss = 0

        bert_model.train()
        for batch in textloader:
            optimizer.zero_grad()
            tokenized_texts, attention_masks, targets = batch 

            tokenized_texts = tokenized_texts.to(device)
            attention_masks = attention_masks.to(device)
            targets = targets.to(device)

            output = bert_model(tokenized_texts, attention_masks)
            loss = loss_fn(output.view(-1, len(vocab)), targets.view(-1))

            loss.backward()
            optimizer.step()
            scheduler.step() 

            steploss += loss.item()
        
        training_history["loss"].append(steploss / len(textloader))

        if step  % 100 == 0:
            torch.save(bert_model.state_dict(), f"{checkpoint_path}/model_checkpoint.pt")
            torch.save(optimizer.state_dict(), f"{optimizer_path}/optimizer_checkpoint.pt")
            torch.save(scheduler.state_dict(), f"{optimizer_path}/scheduler_checkpoint.pt")
            torch.save(step, f"{checkpoint_path}/epoch.pt")
            torch.save(training_history, f"{checkpoint_path}/history.pt")

            print(f"Epoch - {step} --- Loss = {training_history['loss'][-1]}")




    
