use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, Context};
use regex::Regex;
use std::{fs::File, io::BufReader};

pub struct Tokenizer {
    merges: HashMap<(u32, u32), u32>,
    vocab: HashMap<u32, Vec<u8>>,

    chunk_pat: Regex,
    special_pat: Option<Regex>,
    special_to_id: HashMap<String, u32>,
    id_to_special: HashMap<u32, String>,
}

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
    let mut vocab: HashMap<u32, Vec<u8>> = (0u32..=255)
        .map(|b| (b, vec![b as u8]))
        .collect();

    // 1. collect (pair, id) into a Vec so we can sort
    let mut sorted: Vec<(&(u32, u32), &u32)> = merges.iter().collect();
    // 2. sort by id (same order Python used)
    sorted.sort_by_key(|&(_, id)| *id);

    // 3. now build higher tokens in ascending order
    for (&(tok0, tok1), &id) in sorted {
        let mut merged = vocab.get(&tok0)
                              .expect("tok0 missing")
                              .clone();
        merged.extend(
            vocab.get(&tok1)
                 .expect("tok1 missing")
        );
        vocab.insert(id, merged);
    }
    vocab
}



pub fn train(filepath: &str, num_merges: u32, save_dir: &str)  -> Result<()> {
    let text = std::fs::read_to_string(filepath)?;
    let chunk_pat = Regex::new(r"'s|'re|'ll|'ve|'d|'t| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
    let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
    let mut byte_chunks: Vec<Vec<u32>>= chunk_pat
                                .find_iter(&text)
                                .map(|m| m.as_str().as_bytes().iter().map(|&b| b as u32).collect())
                                .collect();

    for i in 0..num_merges {
        let freq = get_freq(&byte_chunks);
        if freq.is_empty() {
            break;
        }
        
        let (&pair, _) = freq.iter().max_by_key(|(_, count)| *count).unwrap();
        byte_chunks = merge(&byte_chunks, pair, 256+i);
        merges.insert(pair, 256+i);
    }

    let vocab = build_vocab(&merges);

    // safe files
    std::fs::create_dir_all(save_dir)?;
    let merge_path = Path::new(save_dir).join("merges.json");
    let vocab_path = Path::new(save_dir).join("vocab.json");

    let merge_json: HashMap<String, u32>= merges
                                               .iter()
                                               .map(|(&(a, b), &id)| (format!("{},{}", a, b), id))
                                               .collect();
    let vocab_json: HashMap<String, Vec<u8>> = vocab
                                            .iter()
                                            .map(|(&id, bytes)| (id.to_string(), bytes.clone()))
                                            .collect();
    serde_json::to_writer_pretty(std::fs::File::create(merge_path)?, &merge_json)?;
    serde_json::to_writer_pretty(std::fs::File::create(vocab_path)?, &vocab_json)?;
    Ok(())

}


impl Tokenizer {
    pub fn new<P: AsRef<std::path::Path>>(tokenizer_dir: P, special_tokens: Option<Vec<&str>>) -> Result<Self> {
        let dir = tokenizer_dir.as_ref();

        // Load merges.json
        let merge_path = dir.join("merges.json");
        let merge_file = File::open(&merge_path).context("Failed to open merges.json")?;
        let merge_json: HashMap<String, u32> = serde_json::from_reader(BufReader::new(merge_file))?;

        let mut merges = HashMap::new();
        for (pair_str, id) in merge_json {
            let parts: Vec<u32> = pair_str
                .split(',')
                .map(|s| s.parse::<u32>().unwrap())
                .collect();
            merges.insert((parts[0], parts[1]), id);
        }

        // Load vocab.json
        let vocab_path = dir.join("vocab.json");
        let vocab_file = File::open(&vocab_path).context("Failed to open vocab.json")?;
        let vocab_json: HashMap<String, Vec<u8>> = serde_json::from_reader(BufReader::new(vocab_file))?;
        let vocab: HashMap<u32, Vec<u8>> = vocab_json
            .into_iter()
            .map(|(k, v)| (k.parse::<u32>().unwrap(), v))
            .collect();

        // Compile main regex
        let chunk_pat = Regex::new(
            r"'s|'re|'ll|'ve|'d|'t| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"
        ).unwrap();

        // Handle special tokens
        let mut special_pat = None;
        let mut special_to_id = HashMap::new();
        let mut id_to_special = HashMap::new();

        if let Some(tokens) = special_tokens {
            let base_id = vocab.len() as u32;
            for (i, token) in tokens.iter().enumerate() {
                let id = base_id + i as u32;
                special_to_id.insert(token.to_string(), id);
                id_to_special.insert(id, token.to_string());
            }

            let pattern = format!("({})", special_to_id.keys()
                .map(|tok| regex::escape(tok))
                .collect::<Vec<_>>()
                .join("|"));
            special_pat = Some(Regex::new(&pattern).unwrap());
        }

        Ok(Self {
            merges,
            vocab,
            chunk_pat,
            special_pat,
            special_to_id,
            id_to_special,
        })
    }
}


impl Tokenizer {
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut chunks: Vec<Vec<u32>> = self.chunk_pat
            .find_iter(text)
            .map(|m| m.as_str().as_bytes().iter().map(|&b| b as u32).collect())
            .collect();

        loop {
            let freq = get_freq(&chunks);
            if freq.is_empty() {
                break;
            }

            let pair = freq
                .iter()
                .filter(|(p, _)| self.merges.contains_key(p))
                .min_by_key(|(p, _)| self.merges.get(p).unwrap())
                .map(|(p, _)| *p);

            if let Some(pair) = pair {
                let new_id = self.merges[&pair];
                chunks = merge(&chunks, pair, new_id);
            } else {
                break;
            }
        }

        chunks.into_iter().flatten().collect()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // No special tokens → ordinary encode
        if self.special_pat.is_none() {
            return self.encode_ordinary(text);
        }
        let special_pat = self.special_pat.as_ref().unwrap();

        let mut tokens = Vec::new();
        let mut last = 0;

        // Walk through every match (= special token)
        for m in special_pat.find_iter(text) {
            // 1) Encode text *before* the special token
            if m.start() > last {
                tokens.extend(self.encode_ordinary(&text[last..m.start()]));
            }
            // 2) Push the special-token ID itself
            if let Some(&id) = self.special_to_id.get(m.as_str()) {
                tokens.push(id);
            }
            last = m.end();
        }
        // 3) Encode any trailing text after the last special token
        if last < text.len() {
            tokens.extend(self.encode_ordinary(&text[last..]));
        }
        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        let mut buffer: Vec<u8> = Vec::new();

        for &token in tokens {
            if let Some(special) = self.id_to_special.get(&token) {
                if let Ok(decoded) = String::from_utf8(buffer.clone()) {
                    result.push_str(&decoded);
                }
                buffer.clear();
                result.push_str(special);
            } else if let Some(bytes) = self.vocab.get(&token) {
                buffer.extend_from_slice(bytes);
            }
        }

        if let Ok(decoded) = String::from_utf8(buffer) {
            result.push_str(&decoded);
        }
        result
    }
}

// ──────────────────────────────────────────────────────────────
//             Python bindings (thin wrapper)
// ──────────────────────────────────────────────────────────────
use pyo3::prelude::*;
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Create from a tokenizer directory.
    #[new]
    fn new(tokenizer_dir: &str, special_tokens: Option<Vec<String>>) -> PyResult<Self> {
        // convert Vec<String> → Vec<&str>
        let specials = special_tokens
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let tok = Tokenizer::new(tokenizer_dir, specials)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: tok })
    }

    /// Encode text into a list[int].
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode list[int] back to text.
    fn decode(&self, tokens: Vec<u32>) -> String {
        self.inner.decode(&tokens)
    }

    #[staticmethod]
    fn train(filepath: &str, num_merges: u32, save_dir: &str) -> PyResult<()> {
        crate::train(filepath, num_merges, save_dir)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// train_bpe(filepath, num_merges, save_dir)
#[pyfunction]
fn train_bpe(filepath: &str, num_merges: u32, save_dir: &str) -> PyResult<()> {
    crate::train(filepath, num_merges, save_dir)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Python module definition: `import bpe_rs`
#[pymodule]
fn bpe_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}