import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from CustomBPE.Model.Tokenizer import AutoTokenizer, MLMTokenizer
import pickle as pkl



class MaskedDataset(Dataset):
    def __init__(self, texts):
        super().__init__()
        self.texts = texts

    def __getitem__(self, index):
        return self.texts[index]
    
    def __len__(self):
        return len(self.texts)
    
class Collator:
    def __init__(self, max_tokens, mask_rate):
        self.mlm_tokenizer = MLMTokenizer(truncation_side="right", from_pretrained=True, return_tensors=True, max_tokens=max_tokens, mask_rate=mask_rate)
    
    def __call__(self, batch):
        tokenized_batch, attention_batch, target_batch = self.mlm_tokenizer(batch)
        return tokenized_batch, attention_batch, target_batch

        

class MaskedData:
    def __init__(self, src = "CustomBPE/Data/BookCorpus3.csv", save_path = "BookCorpus/books.pkl"):
        self.data = pd.read_csv(src)
        self.save_path = save_path

    def save(self):
        texts = []
        for i in tqdm(range(self.data.shape[0])):
            texts.append(self.data.loc[i][0])
        
        with open(f"{self.save_path}", "wb") as fp: 
            pkl.dump(texts, fp)

    @classmethod
    def data_to_masked_tensors(cls, corpus_path, save_path, batch_size = 128, mask_rate = 0.15, max_tokens = 512):
        mlm_tokenizer = MLMTokenizer(truncation_side="right", from_pretrained=True, return_tensors=True, max_tokens=max_tokens, mask_rate=mask_rate)
        
        with open(corpus_path, "rb") as fp:
            texts = pkl.load(fp)

        tokenized_texts = []
        attention_masks = []
        targets = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            tokenized_batch, attention_batch, target_batch = mlm_tokenizer(batch_texts)
            tokenized_texts.append(tokenized_batch)
            attention_masks.append(attention_masks)
            targets.append(target_batch)

        tokenized_texts = torch.cat(tokenized_texts, dim = 0)
        attention_masks = torch.cat(attention_masks, dim = 0)
        targets = torch.cat(targets, dim = 0)

        torch.save(tokenized_texts, f"{save_path}tokenized_texts.pt")
        torch.save(attention_masks, f"{save_path}attention_masks.pt")
        torch.save(targets, f"{save_path}targets.pt")

        return tokenized_texts, attention_masks, targets
    
    @classmethod
    def create_dataloader(cls, corpus_path, batch_size = 128, mask_rate = 0.15, max_tokens = 512):
        with open(corpus_path, "rb") as fp:
            texts = pkl.load(fp)

        text_dataset = MaskedDataset(texts=texts)
        collate_ = Collator(max_tokens=max_tokens, mask_rate=mask_rate)

        text_dataloader = DataLoader(dataset=text_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_)
        unique_vocab = { v[0] : k for k, v in collate_.mlm_tokenizer.bpe_model.inverse_vocab.items()}

        return text_dataloader, unique_vocab








        

    


