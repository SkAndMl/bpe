use std::collections::HashMap;

fn get_freq(byte_chunks: &[Vec<u32>]) -> HashMap<(u32, u32), usize>{
    let mut freq = HashMap::new();
    for chunk in byte_chunks{
        for pair in chunk.windows(2){
            let key = (pair[0], pair[1]);
            *freq.entry(key).or_insert(0) += 1;
        }
    }
    return freq;
}

fn merge(byte_chunks: &[Vec<u32>], pair: (u32, u32), new_id: u32) -> Vec<Vec<u32>>{
    let mut new_byte_chunks = Vec::new();
    for byte_chunk in byte_chunks{
        let mut new_byte_chunk = Vec::new();
        let mut i: usize= 0;
        while i < byte_chunk.len() - 1{
            if byte_chunk[i] == pair.0 && byte_chunk[i+1] == pair.1{
                new_byte_chunk.push(new_id);
                i += 2;
            }
            else{
                new_byte_chunk.push(byte_chunk[i]);
                i += 1;
            }
        }
        if i == byte_chunk.len() - 1{
            new_byte_chunk.push(byte_chunk[i]);
        }
        new_byte_chunks.push(new_byte_chunk);
    }
    return new_byte_chunks;
}

fn build_vocab(merges: &HashMap<(u32, u32), u32>) -> HashMap<u32, Vec<u8>> {
    let mut vocab = HashMap::new();

    for i in 0u32..=255 {
        vocab.insert(i, vec![i as u8]);
    }

    for (&(tok0, tok1), &id) in merges {
        let mut merged = vocab.get(&tok0).unwrap().clone();
        merged.extend(vocab.get(&tok1).unwrap());
        vocab.insert(id, merged);
    }

    vocab
}
