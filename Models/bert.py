import torch
import torch.nn as nn

from Models.blocks import EmbeddingLayer, SinusoidPE, MultiheadAttention, FeedForward, EncoderLayer


class TineeBERT(nn.Module):
    def __init__(self, num_repeats, vocab_size, embed_size, seqlen, num_heads, device = "cpu"):
        super().__init__()

        self.device = device
        self.embed_size = embed_size
        self.num_repeats = num_repeats
        self.vocab_size = vocab_size
        self.seqlen = seqlen
        self.num_heads = num_heads

        self.embedding = EmbeddingLayer(vocab_size, embed_size)
        self.positional_encoding = SinusoidPE(embed_size, seqlen, device = device)
        
        self.encoders = nn.ModuleList([
            EncoderLayer(embed_size = embed_size, num_heads = num_heads, hidden_size = embed_size, device = device) for _ in range(num_repeats)
        ])

        self.decoder = nn.Linear(embed_size, vocab_size) 


    def forward(self, x, mask = None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for encoder in self.encoders:
            x = encoder(x, mask = mask)

        logits = self.decoder(x)
        return logits



