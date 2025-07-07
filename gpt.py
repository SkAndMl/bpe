import torch
import torch.nn.functional as F
from torch import nn, Tensor
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int
    ctx_size: int
    n_heads: int
    head_dim: int
    n_blocks: int
    device: torch.device = "cpu"
    ffn_multiplier: int = 4

class Embedding(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.tok_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embedding_table = nn.Embedding(cfg.ctx_size, cfg.n_embd)
    
    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, seq_len
        _, seq_len = x.shape
        tok_embeddings = self.tok_embedding_table(x) # bsz, seq_len, n_embd
        pos_embeddings = self.pos_embedding_table(
            torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        ) # 1, seq_len, n_embd

        return tok_embeddings + pos_embeddings # bsz, seq_len, n_embd

class FFN(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd*4),
            nn.ReLU(),
            nn.Linear(cfg.n_embd*4, cfg.n_embd)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)

class MHA(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.head_dim * cfg.n_heads == cfg.n_embd
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, cfg.n_embd*3)
        self.o = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.register_buffer(
            "mask",
            torch.triu(torch.full((1, 1, cfg.ctx_size, cfg.ctx_size), float('-inf')), diagonal=1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, seq_len, n_embd
        bsz, seq_len, n_embd = x.shape
        qkv: Tensor = self.qkv(x) # bsz, seq_len, n_embd*3
        q, k, v = qkv.split(self.cfg.n_embd, -1)
        q = q.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2) # bsz, n_heads, seq_len, head_dim
        k = k.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)

        attn_scores = q @ k.transpose(-1, -2) / (self.cfg.head_dim ** 0.5)  # bsz, n_heads, seq_len, seq_len
        attn_scores += self.mask[:, :, :seq_len, :seq_len] # masks tokens
        attn_scores = F.softmax(attn_scores, dim=-1)
        y = (attn_scores @ v).transpose(1, 2).contiguous().reshape(bsz, seq_len, -1) # bsz, seq_len, n_embd
        return self.o(y)

class Block(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mha = MHA(cfg)
        self.ffn = FFN(cfg)
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = Embedding(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_blocks)])
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(x)[:, -1]
            token = torch.argmax(logits, -1).unsqueeze(1)
            x = torch.cat([x, token], dim=1)
        return x