import regex as re
import os
import json

from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from tqdm import tqdm

class BPE:

    CHUNK_PAT = re.compile(r"""'s|'ve|'ll|'d|'t|'re|'m| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(self, tokenizer_dir: str, special_tokens: Optional[List[str]]=None):
        
        assert os.path.exists(tokenizer_dir)
        tokenizer_path = Path(tokenizer_dir)
        with open(tokenizer_path / "merges.json", "r") as f:
            self.merges = {}
            for k, v in json.loads(f.read()).items():
                tok0, tok1 = k.split(",")
                self.merges[(int(tok0), int(tok1))] = int(v)
        
        with open(tokenizer_path / "vocab.json", "r") as f:
            self.vocab = {}
            for k, v in json.loads(f.read()).items():
                self.vocab[int(k)] = bytes(v)
        
        self.special_tokens = special_tokens
        if special_tokens is not None:
            self._init_special_tokens()
    
    def _init_special_tokens(self):
        self.SPECIAL_PAT = r"(" + \
                           r"|".join(re.escape(sp_token) for sp_token in self.special_tokens) + \
                           r")" 
        self.sp_token_to_id = {self.special_tokens[i]: len(self.vocab)+i for i in range(len(self.special_tokens))}
        self.id_to_sp_token = {v: k for k, v in self.sp_token_to_id.items()}

    def encode_ordinary(self, text: str) -> List[int]:
        byte_chunks = re.findall(self.CHUNK_PAT, text)
        byte_chunks = [list(chunk.encode("utf-8")) for chunk in byte_chunks]
        while True:
            freq = BPE.get_freq(byte_chunks)
            if len(freq) == 0:
                break
            pair = min(freq, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            byte_chunks = BPE.merge(byte_chunks, pair, self.merges[pair])
        
        return [token for chunk in byte_chunks for token in chunk]

    def encode(self, text: str) -> List[int]:
        if self.special_tokens is not None:
            chunks = re.split(self.SPECIAL_PAT, text)
        else:
            chunks = [text]
        
        tokens = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                tokens.append(self.sp_token_to_id[chunk])
            else:
                tokens.extend(self.encode_ordinary(chunk))
        return tokens


    def decode(self, tokens: List[int]) -> str:
        decoded_str = ""
        _bytes = b""
        for token in tokens:
            if token in self.id_to_sp_token:
                decoded_str += _bytes.decode("utf-8", errors="ignore")
                decoded_str += self.id_to_sp_token[token]
                _bytes = b""
            else:
                _bytes += self.vocab[token]
        
        if _bytes:
            decoded_str += _bytes.decode("utf-8", errors="ignore")

        return decoded_str


    @staticmethod
    def get_freq(byte_chunks: List[List[int]]) -> dict:
        freq = defaultdict(int)
        for byte_chunk in byte_chunks:
            for b0, b1 in zip(byte_chunk[:-1], byte_chunk[1:]):
                freq[(b0, b1)] += 1
        return freq

    @staticmethod
    def merge(byte_chunks: List[List[int]], pair: Tuple[int], new_id: int):
        new_byte_chunks = []
        for byte_chunk in byte_chunks:
            new_byte_chunk = []
            i = 0
            while i < len(byte_chunk)-1:
                if byte_chunk[i] == pair[0] and byte_chunk[i+1] == pair[1]:
                    new_byte_chunk.append(new_id)
                    i += 2
                else:
                    new_byte_chunk.append(byte_chunk[i])
                    i += 1
            
            if i == len(byte_chunk)-1:
                new_byte_chunk.append(byte_chunk[-1])

            new_byte_chunks.append(new_byte_chunk)
        return new_byte_chunks

    @classmethod
    def build_vocab(cls, merges: Dict[Tuple[int], int]) -> Dict:
        vocab = {_id: bytes([_id]) for _id in range(256)}
        for (tok0, tok1), _id in sorted(merges.items(), key=lambda m: m[1]):
            vocab[_id] = vocab[tok0] + vocab[tok1]
        return vocab

    @classmethod
    def train(cls, filepath: str, num_merges: int, save_dir: str) -> None:

        with open(filepath, "r") as f:
            text = f.read()
        
        chunks = re.findall(cls.CHUNK_PAT, text)
        byte_chunks = [list(bytes(chunk.encode("utf-8"))) for chunk in chunks]
        
        merges = {} # (tok_id1, tok_id2) -> new_tok_id
        for i in tqdm(range(num_merges), desc="Training..."):
            freq = cls.get_freq(byte_chunks)
            if len(freq) == 0:
                break
            pair = max(freq, key=freq.get)
            byte_chunks = cls.merge(byte_chunks, pair, 256 + i)
            merges[pair] = 256 + i
        
        vocab = cls.build_vocab(merges)
        
        os.makedirs(save_dir, exist_ok=True)

        merges_path = os.path.join(save_dir, "merges.json")
        vocab_path = os.path.join(save_dir, "vocab.json")
        

        with open(merges_path, "w") as f:
            merges = {",".join([str(tok) for tok in k]): v for k, v in merges.items()}
            f.write(json.dumps(merges))

        with open(vocab_path, "w") as f:
            vocab = {str(k): list(v) for k, v in vocab.items()}
            f.write(json.dumps(vocab))     


if __name__ == "__main__":
    if not os.path.exists("tokenizer"):
        BPE.train("data/tinystories/train.txt", num_merges=3000, save_dir="tokenizer")
    
    special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    tokenizer = BPE("tokenizer", special_tokens)
    
    test_txt = [
        "hi how are you?",
        "i am good",
        "my number is 123456789",
        "1+1 is 2"
    ]

    for txt in test_txt:
        _txt = special_tokens[0] + txt + special_tokens[1] + special_tokens[2]
        assert tokenizer.decode(tokenizer.encode(_txt)) == _txt
    
    print("passed all tests!")