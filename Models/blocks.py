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
    

class SinusoidPE(nn.Module):
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
    def __init__(self, embed_size, num_heads, device = "cpu"):
        super().__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.device = device
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.lin_q = nn.Linear(embed_size, embed_size)
        self.lin_k = nn.Linear(embed_size, embed_size)
        self.lin_v = nn.Linear(embed_size, embed_size)
        self.lin_out = nn.Linear(embed_size, embed_size)

    
    def forward(self, query, key, value, mask = None):
        """
            query : (batch_size, seq_len, embed_size)
            key : (batch_size, seq_len, embed_size) 
            value : (batch_size, seq_len, embed_size)
        """

        Q = self.lin_q(query)
        K = self.lin_k(key)
        V = self.lin_v(value)

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim)

        attention_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attention_mask = torch.zeros_like(attention_scores).to(self.device)
            un_attention_indices = torch.argwhere(mask == 0)
            attention_mask[un_attention_indices[:,0], :, un_attention_indices[:,1], :] = -1e10
            attention_mask[un_attention_indices[:,0], :, :, un_attention_indices[:,1]] = -1e10

        attention_scores = attention_scores + attention_mask
        attention_weights = F.softmax(attention_scores, dim = -1)
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
        attention_output = attention_output.view(attention_output.shape[0], attention_output.shape[1], -1)

        attention_output = self.lin_out(attention_output)
        return attention_output


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout = 0.2, device = "cpu"):
        super().__init__()

        self.device = device
        self.attention = MultiheadAttention(embed_size, num_heads, device)
        self.feed_forward = FeedForward(embed_size, hidden_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
       