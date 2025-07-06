import time, pathlib

from bpe_rs import Tokenizer as RsTokenizer
from bpe_py.bpe import BPE as PyTokenizer                         

DATA_FILE   = "data/tinystories/train.txt"  
RS_DIR      = "tokenizer_rs"
PY_DIR      = "tokenizer_py"
NUM_MERGES  = 3_000
SPECIALS    = ["<pad>", "<s>", "</s>"]

def elapsed(fn, *args, **kw):
    """Return result, seconds"""
    start = time.perf_counter()
    print(f"starting {fn}")
    res   = fn(*args, **kw)
    return res, time.perf_counter() - start


if not pathlib.Path(RS_DIR).exists():
    _, rs_train_sec = elapsed(RsTokenizer.train, DATA_FILE, NUM_MERGES, RS_DIR)
else:
    rs_train_sec = float('nan')

if not pathlib.Path(PY_DIR).exists():
    _, py_train_sec = elapsed(
        PyTokenizer.train, DATA_FILE, NUM_MERGES, PY_DIR
    )
else:
    py_train_sec = float('nan')


print(f"Training with rust implementation: {rs_train_sec: .2f}")
print(f"Training with python implementation: {py_train_sec: .2f}")