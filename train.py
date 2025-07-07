import torch
import random
import os
import time

from torch import nn
from torch.nn import functional as F
from gpt import GPT, ModelConfig
from bpe_py.bpe import BPE
from typing import Tuple
from torch.amp import autocast
from argparse import ArgumentParser



class DataLoader:

    def __init__(self, filepath, tokenizer, batch_size, ctx_size):
    
        with open(filepath, "r") as f:
            self.data = f.read().split("<|endoftext|>")
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.ctx_size = ctx_size
        self.i = 0

    def _convert_data_to_tensors(self, data) -> None:
        tokens = []
        for i in range(len(data)):
            _tokens = self.tokenizer.encode(data[i])
            _tokens += [self.tokenizer.sp_token_to_id["<|endoftext|>"]] * (self.ctx_size + 1 - len(_tokens))
            tokens.append(torch.tensor(_tokens[:self.ctx_size+1]))
        return tokens

    def __len__(self) -> int:
        return len(self.data) // self.batch_size

    def __iter__(self):
        self.i = 0
        random.shuffle(self.data)
        return self
            
    def __next__(self) -> Tuple[torch.Tensor]:
        if self.i + self.batch_size > len(self.data):
            raise StopIteration
        
        tokens = torch.stack(
            self._convert_data_to_tensors(self.data[self.i:self.i+self.batch_size]),
            dim=0
        )
        x, y = tokens[:, :-1], tokens[:, 1:]
        self.i += self.batch_size
        return x, y
    


def train(cfg: ModelConfig, tokenizer: BPE, bsz: int):
    gpt = GPT(cfg).to(cfg.device)
    train_dl = DataLoader("data/tinystories/train.txt", tokenizer, bsz, cfg.ctx_size)
    val_dl = DataLoader("data/tinystories/test.txt", tokenizer, bsz, cfg.ctx_size)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.sp_token_to_id["<|endoftext|>"])
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=3e-4)

    use_amp = cfg.device.type == "mps"

    @torch.inference_mode()
    def evaluate() -> float:
        val_loss = 0
        for x, y in val_dl:
            x, y = x.to(cfg.device), y.to(cfg.device)
            with autocast(device_type=cfg.device.type, dtype=torch.float16 if use_amp else torch.float32):
                logits: torch.Tensor = gpt(x) # bsz, seq_len, vocab_size
                logits = logits.view(-1, logits.size(-1))
                loss: torch.Tensor = loss_fn(logits, y.view(-1))
            val_loss += loss.item()
        
        return val_loss / len(val_dl)


    log_file = open(f"log_{cfg.vocab_size}.txt", "w")
    log_step = 50
    generate_step = 500
    evaluate_step = 100
    stop_step = 2000
    
    start_time = time.time()
    for step, (x, y) in enumerate(train_dl):
        if step >= stop_step:
            to_log = f"training done for vocab size: {cfg.vocab_size}; took: {time.time() - start_time:.4f} seconds"
            print(to_log)
            break
        x, y = x.to(cfg.device), y.to(cfg.device) # bsz, seq_len
        with autocast(device_type=cfg.device.type, dtype=torch.float16 if use_amp else torch.float32):
            logits: torch.Tensor = gpt(x) # bsz, seq_len, vocab_size
            logits = logits.view(-1, logits.size(-1))
            loss: torch.Tensor = loss_fn(logits, y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_step == 0:
            to_log = f"Step {step+1} | loss = {loss.item():.4f}"
            log_file.write(to_log + "\n")
            print(to_log)
        if (step + 1) % generate_step == 0 or step == len(train_dl) - 1:
            with torch.no_grad():
                x = torch.tensor([tokenizer.encode("I am going to")]).to(cfg.device)
                out = gpt.generate(x, max_new_tokens=30)
                decoded_str = tokenizer.decode(out.detach().tolist()[0])
                to_log = f"Generation at step {step+1}: {decoded_str}"
                log_file.write(to_log + "\n")
                print(to_log)
        if (step + 1) % evaluate_step == 0:
            val_loss = evaluate()
            to_log = f"Step {step+1} | val loss = {val_loss:.4f}"
            log_file.write(to_log + "\n")
            print(to_log)

    log_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str)
    
    args = parser.parse_args()
    tokenizer_dir = args.tokenizer_dir
    assert os.path.exists(tokenizer_dir)

    bsz = 16
    tokenizer = BPE(tokenizer_dir, ["<|endoftext|>"])
    vocab_size = len(tokenizer.vocab) + len(tokenizer.special_tokens)
    cfg = ModelConfig(
        vocab_size=vocab_size,
        n_embd=256,
        ctx_size=256,
        n_heads=4,
        head_dim=64,
        n_blocks=4
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg.device = device
    print(f"running on {device}")

    train(cfg, tokenizer, bsz)    