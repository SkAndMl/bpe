use bpe_rs::{train, Tokenizer};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    if !std::path::Path::new("tokenizer/merges.json").exists() {
        train("../data/tinystories/train.txt", 3000, "tokenizer")?;
    }
    println!("training took {:.2?}", start.elapsed());
    // Load
    let tok = Tokenizer::new("tokenizer", Some(vec![
        "<|im_start|>", "<|im_end|>", "<|endoftext|>"
    ]))?;

    // Try it
    let text = "<|im_start|>Hi, how are you?<|im_end|><|endoftext|>";
    let ids  = tok.encode(text);
    let back = tok.decode(&ids);

    println!("encoded ids: {:?}", ids);
    println!("decoded    : {}", back);

    Ok(())
}
