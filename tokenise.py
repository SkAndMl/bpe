from bpe_rs import Tokenizer
from argparse import ArgumentParser
from time import time

def tokenise(num_merges: int) -> None:
    start_time = time()
    Tokenizer.train("data/tinystories/train_tokenizer.txt", num_merges, f"tokenizer_{num_merges}")
    print(f"Time taken for {num_merges} merges: {time() - start_time:.4f} seconds")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_merges", "-n", type=int, default=3000)
    args = parser.parse_args()
    tokenise(args.num_merges)