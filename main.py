import torch
import torch.nn as nn
from Models.bert import TineeBERT
from CustomBPE.Model.Tokenizer import AutoTokenizer, MLMTokenizer

from utils.loader import MaskedData
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd



def lr_lambda(current_step, num_warmup_steps = 10000, num_training_steps = 1000000):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        return max(0.0, float(num_training_steps - current_step) /
                        float(max(1, num_training_steps - num_warmup_steps)))

if __name__ == "__main__":
    checkpoint_path = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_steps = 1000000
    warmup_steps = 10000
    
    textloader, vocab = MaskedData.create_dataloader(corpus_path="BookCorpus/books.pkl", batch_size=256, mask_rate=0.15, max_tokens=512)
    bert_model = TineeBERT(num_repeats=10, vocab_size=len(vocab), embed_size=768, seqlen=512, num_heads=16)
    optimizer = torch.optim.Adam(bert_model.parameters(), lr = 1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in textloader:
            tokenized_texts, attention_masks, targets = batch 
            output = bert_model(tokenized_texts, attention_masks)
            loss = loss_fn(output, targets)
            print(output.shape)
            print(loss)
            break

    
