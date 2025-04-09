import torch
import torch.nn as nn
from Models.bert import TineeBERT
from CustomBPE.Model.Tokenizer import AutoTokenizer, MLMTokenizer

from utils.loader import MaskedData
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd

from tqdm import tqdm
import gc

torch.cuda.empty_cache()



if __name__ == "__main__":
    checkpoint_path = "Checkpoints"
    optimizer_path = "Optimizer_States"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_steps = 1000000
    warmup_steps = 10000

    textloader, vocab = MaskedData.create_dataloader(corpus_path="BookCorpus/books.pkl", batch_size=64, mask_rate=0.20, max_tokens=256)
    bert_model = TineeBERT(num_repeats=10, vocab_size=len(vocab), embed_size=1024, seqlen=256, num_heads=8)
    optimizer = torch.optim.Adam(bert_model.parameters(), lr = 1e-4)



    def lr_lambda(current_step, num_warmup_steps = 10000, num_training_steps = len(textloader)):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return max(0.0, float(num_training_steps - current_step) /
                            float(max(1, num_training_steps - num_warmup_steps)))


    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    bert_model = bert_model.to(device)

    training_history = {
        "loss" : []
    }

    update_step = 0

    bert_model.train()
    for batch in tqdm(textloader, total=len(textloader)):
        update_step += 1
        
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

        steploss = loss.item()
    
        training_history["loss"].append(steploss)

        if update_step  % 100 == 0:
            torch.save(bert_model.state_dict(), f"{checkpoint_path}/model_checkpoint_v1.pt")
            torch.save(optimizer.state_dict(), f"{optimizer_path}/optimizer_checkpoint_v1.pt")
            torch.save(scheduler.state_dict(), f"{optimizer_path}/scheduler_checkpoint_v1.pt")
            torch.save(update_step, f"{checkpoint_path}/epoch_v1.pt")
            torch.save(training_history, f"{checkpoint_path}/history_v1.pt")

            print(f"Step- {update_step} --- Loss = {training_history['loss'][-1]}")

        




    
