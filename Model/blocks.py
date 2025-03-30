import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.embed_size)
    

class AbsolutePE(nn.Module):
    def __init__(self, embed_size, seq_len, dropout = 0.2, device = "cpu"):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.positional_encoding = torch.zeros(seq_len, embed_size).to(device)
        self.positional_encoding.requires_grad = False

        positions = torch.arange(0, seq_len).unsqueeze(1).to(device)
        positions = positions.to(torch.float16)

        _2i = torch.arange(0, embed_size, 2).to(device)

        self.positional_encoding[:, 0::2] = torch.sin(positions / (10000 ** (_2i / embed_size)))
        self.positional_encoding[:, 1::2] = torch.cos(positions / (10000 ** (_2i / embed_size)))
        self.positional_encoding = self.positional_encoding.unsqueeze(0).to(device)

    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_size)
        """
        x = x + self.positional_encoding
        x = self.dropout(x)
        return x
    


class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout = 0.2, device = "cpu"):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.device = device
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.lin_q = nn.Linear(embed_size, embed_size)
        self.lin_k = nn.Linear(embed_size, embed_size)
        self.lin_v = nn.Linear(embed_size, embed_size)

    
    def forward(self, query, key, value, mask = None):
        """
            query : (batch_size, seq_len, embed_size)
            key : (batch_size, seq_len, embed_size) 
            value : (batch_size, seq_len, embed_size)
        """

        Q = self.lin_q(query)
        K = self.lin_k(key)
        V = self.lin_v(value)

        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim)

        attention_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attention_mask = torch.zeros_like(attention_scores).to(self.device)
            


        
        
        