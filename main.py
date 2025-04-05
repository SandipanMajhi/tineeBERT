from Models.bert import TineeBERT
from CustomBPE.Model.Tokenizer import AutoTokenizer, MLMTokenizer

from utils.loader import MaskedData

import pandas as pd

if __name__ == "__main__":
    # masked_data = MaskedData()
    # tokenized_texts, attention_masks, targets = masked_data.create_dataloaders(corpus_path="BookCorpus/books.pkl", save_path="BookCorpus/", batch_size=1024)

    model = TineeBERT(num_repeats=3)
    textloader = MaskedData.create_dataloader(corpus_path="BookCorpus/books.pkl", batch_size=256, mask_rate=0.15, max_tokens=512)

    for batch in textloader:
        tokenized_texts, attention_masks, targets = batch 
        print(tokenized_texts.shape)
        print(attention_masks.shape)
        print(targets.shape)
        break

    
