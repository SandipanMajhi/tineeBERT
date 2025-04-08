import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from Models.bert import TineeBERT
from CustomBPE.Model.Tokenizer import AutoTokenizer, MLMTokenizer

from utils.loader import MaskedData
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd

import os
from tqdm import tqdm


torch.cuda.empty_cache()

def setup_ddp():
    # Initialize the process group. This example assumes environment variables are set.
    # For example, WORLD_SIZE, RANK, and LOCAL_RANK can be set by torchrun.
    dist.init_process_group(backend="nccl")  # Use NCCL for multi-GPU training

def cleanup_ddp():
    dist.destroy_process_group()



if __name__ == "__main__":
    setup_ddp()

    checkpoint_path = "Checkpoints"
    optimizer_path = "Optimizer_States"
    
    
    # Retrieve local GPU id from environment variable
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)


    training_steps = 1000000
    warmup_steps = 10000
    
    #### Distributed loader object ####
    textloader, vocab = MaskedData.create_dataloader(corpus_path="BookCorpus/books.pkl", batch_size=64, mask_rate=0.15, max_tokens=512)
    bert_model = TineeBERT(num_repeats=10, vocab_size=len(vocab), embed_size=512, seqlen=512, num_heads=8)
    ddp_bert_model = DDP(bert_model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(ddp_bert_model.parameters(), lr = 1e-4)



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

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if update_step  % 100 == 0 and local_rank == 0:
            torch.save(bert_model.state_dict(), f"{checkpoint_path}/model_checkpoint_ddp.pt")
            torch.save(optimizer.state_dict(), f"{optimizer_path}/optimizer_checkpoint_ddp.pt")
            torch.save(scheduler.state_dict(), f"{optimizer_path}/scheduler_checkpoint_ddp.pt")
            torch.save(update_step, f"{checkpoint_path}/epoch_ddp.pt")
            torch.save(training_history, f"{checkpoint_path}/history_ddp.pt")

            print(f"Step- {update_step} --- Loss = {training_history['loss'][-1]}")

        if update_step >= 50000:
            break

    cleanup_ddp()

        




    
