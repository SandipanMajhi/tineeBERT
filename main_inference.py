import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.bert import TineeBERT
from CustomBPE.Model.Tokenizer import AutoTokenizer, MLMTokenizer

from utils.loader import MaskedData
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd

from tqdm import tqdm

torch.cuda.empty_cache()

if __name__ == "__main__":
    checkpoint_path = "Checkpoints"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    mlm_tokenizer = MLMTokenizer(truncation_side="right", from_pretrained=True, max_tokens=512)
    textloader, vocab = MaskedData.create_dataloader(corpus_path="BookCorpus/books.pkl", batch_size=1, mask_rate=0.15, max_tokens=512)
    bert_model = TineeBERT(num_repeats=10, vocab_size=len(vocab), embed_size=512, seqlen=512, num_heads=8)
    bert_model.load_state_dict(torch.load("Checkpoints/model_checkpoint_v2.pt", weights_only=True))

    bert_model = bert_model.to(device)

    bert_model.eval()
    with torch.no_grad():
        for batch in textloader:
            tokenized_texts, attention_masks, targets = batch 

            tokenized_texts = tokenized_texts.to(device)
            attention_masks = attention_masks.to(device)
            targets = targets.to(device)

            logits = bert_model(tokenized_texts, attention_masks)
            predicted_seq = F.softmax(logits, dim = -1)
            predicted_seq = torch.argmax(predicted_seq, dim = -1)
            print("Before BERT:")
            print(tokenized_texts)
            print(mlm_tokenizer.batch_decode(tokenized_texts))

            print("After BERT:")
            print(predicted_seq)
            print(mlm_tokenizer.batch_decode(predicted_seq))
            break