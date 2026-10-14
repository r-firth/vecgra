//! EmbeddingGemma 2 running in-process through ONNX Runtime.
//!
//! The text-only ONNX export and its tokenizer are downloaded once from
//! Hugging Face into a local cache, pinned to a repository revision and
//! verified by SHA-256.

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use tokenizers::{Encoding, Tokenizer, TruncationParams};

pub const GEMMA_MODEL: &str = "embeddinggemma-2";
/// Matryoshka sizes EmbeddingGemma 2 is trained to truncate to.
pub const GEMMA_DIMENSIONS: [usize; 4] = [768, 512, 256, 128];

const QUERY_PROMPT: &str = "task: search result | query: ";
const DOCUMENT_PROMPT: &str = "title: none | text: ";
const MAX_TOKENS: usize = 8192;
/// Maximum texts per ONNX Runtime call.
const INFERENCE_BATCH: usize = 8;
/// Bound padded tokens as well as rows; a longer single document runs alone.
const BATCH_TOKEN_BUDGET: usize = 2048;
/// Tokenize bounded windows rather than retaining encodings for an entire import.
const TOKENIZATION_WINDOW: usize = 128;
/// Width of the empty image, video and audio feature inputs.
const TEXT_HIDDEN_SIZE: usize = 512;

const REPOSITORY: &str = "onnx-community/embeddinggemma-2-ONNX";
const REVISION: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";
const MODEL_FILE: &str = "onnx/model_quantized.onnx";

struct ModelFile {
    path: &'static str,
    sha256: &'static str,
    bytes: u64,
}

const FILES: [ModelFile; 3] = [
    ModelFile {
        path: "tokenizer.json",
        sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
        bytes: 32_170_510,
    },
    ModelFile {
        path: MODEL_FILE,
        sha256: "d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953",
        bytes: 495_165,
    },
    ModelFile {
        path: "onnx/model_quantized.onnx_data",
        sha256: "278a7ff1248c3618e4bd11a607fc54f7bdc7778854230f3956d3f86bd9db4f3b",
        bytes: 313_724_928,
    },
];

static MODEL: Mutex<Option<GemmaModel>> = Mutex::new(None);

/// Embeds documents with the retrieval-document prompt.
pub fn gemma_documents(texts: &[String], dimension: usize) -> Result<Vec<Vec<f32>>, String> {
    gemma_documents_with_progress(texts, dimension, INFERENCE_BATCH, |_| {})
}

/// Groups documents by length and limits padded tokens per batch, preserving input order.
/// Calls `progress` with the cumulative completed text count after each inference.
/// `max_batch_size` can lower the default ceiling of eight texts per inference.
pub fn gemma_documents_with_progress(
    texts: &[String],
    dimension: usize,
    max_batch_size: usize,
    progress: impl FnMut(usize),
) -> Result<Vec<Vec<f32>>, String> {
    embed(texts, DOCUMENT_PROMPT, dimension, max_batch_size, progress)
}

/// Embeds one retrieval query with the search-query prompt.
pub fn gemma_query(text: &str, dimension: usize) -> Result<Vec<f32>, String> {
    embed(&[text.to_owned()], QUERY_PROMPT, dimension, 1, |_| {})?
        .pop()
        .ok_or_else(|| "EmbeddingGemma 2 returned no query embedding".into())
}

pub fn validate_gemma_dimension(dimension: usize) -> Result<(), String> {
    if GEMMA_DIMENSIONS.contains(&dimension) {
        Ok(())
    } else {
        Err(format!(
            "EmbeddingGemma 2 dimension must be one of {GEMMA_DIMENSIONS:?}, got {dimension}"
        ))
    }
}

fn embed(
    input: &[String],
    prompt: &str,
    dimension: usize,
    max_batch_size: usize,
    mut progress: impl FnMut(usize),
) -> Result<Vec<Vec<f32>>, String> {
    validate_gemma_dimension(dimension)?;
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mut model = loaded_model()?;
    let model = model.as_mut().expect("model was loaded");
    let mut vectors = vec![Vec::new(); input.len()];
    let mut indexes: Vec<_> = (0..input.len()).collect();
    // Group by byte length without tokenizing the whole corpus into memory.
    // Exact token lengths determine the inference batches within each window.
    indexes.sort_unstable_by_key(|&index| (input[index].len(), index));
    let mut completed = 0;
    for window in indexes.chunks(TOKENIZATION_WINDOW) {
        let encodings = model
            .tokenizer
            .encode_batch(
                window
                    .iter()
                    .map(|&index| format!("{prompt}{}", input[index]))
                    .collect(),
                true,
            )
            .map_err(|error| format!("could not tokenize text: {error}"))?;
        let mut encoded: Vec<_> = window.iter().copied().zip(encodings).collect();
        encoded.sort_unstable_by_key(|(index, encoding)| (encoding.len(), *index));
        let lengths: Vec<_> = encoded.iter().map(|(_, encoding)| encoding.len()).collect();
        for range in inference_batches(&lengths, max_batch_size) {
            let batch = &encoded[range];
            for ((index, _), vector) in batch.iter().zip(model.embed(batch)?) {
                vectors[*index] = truncate(vector, dimension)?;
            }
            completed += batch.len();
            progress(completed);
        }
    }
    Ok(vectors)
}

fn inference_batches(lengths: &[usize], max_texts: usize) -> Vec<Range<usize>> {
    let max_texts = max_texts.clamp(1, INFERENCE_BATCH);
    let mut batches = Vec::new();
    let mut start = 0;
    while start < lengths.len() {
        let mut end = start + 1;
        let mut longest = lengths[start];
        while end < lengths.len() && end - start < max_texts {
            longest = longest.max(lengths[end]);
            if longest * (end - start + 1) > BATCH_TOKEN_BUDGET {
                break;
            }
            end += 1;
        }
        batches.push(start..end);
        start = end;
    }
    batches
}

/// Loads the model on first use and keeps it for the life of the process.
fn loaded_model() -> Result<MutexGuard<'static, Option<GemmaModel>>, String> {
    let mut model = MODEL
        .lock()
        .map_err(|_| "EmbeddingGemma 2 model lock was poisoned".to_string())?;
    if model.is_none() {
        *model = Some(GemmaModel::load(&model_directory()?)?);
    }
    Ok(model)
}

struct GemmaModel {
    session: Session,
    tokenizer: Tokenizer,
}

impl GemmaModel {
    fn load(directory: &Path) -> Result<Self, String> {
        for file in &FILES {
            ensure_file(directory, file)?;
        }
        let mut tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| format!("could not load EmbeddingGemma 2 tokenizer: {error}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..TruncationParams::default()
            }))
            .map_err(|error| format!("could not configure tokenizer truncation: {error}"))?;
        let session = open_session(&directory.join(MODEL_FILE))
            .map_err(|error| format!("could not load EmbeddingGemma 2: {error}"))?;
        Ok(Self { session, tokenizer })
    }

    fn embed(&mut self, encodings: &[(usize, Encoding)]) -> Result<Vec<Vec<f32>>, String> {
        let rows = encodings.len();
        let columns = encodings
            .iter()
            .map(|(_, encoding)| encoding.len())
            .max()
            .unwrap_or(0);
        let mut input_ids = Vec::with_capacity(rows * columns);
        let mut attention_mask = Vec::with_capacity(rows * columns);
        for (_, encoding) in encodings {
            input_ids.extend(encoding.get_ids().iter().map(|&id| i64::from(id)));
            attention_mask.extend(encoding.get_attention_mask().iter().map(|&m| i64::from(m)));
            let padding = columns - encoding.len();
            input_ids.resize(input_ids.len() + padding, 0);
            attention_mask.resize(attention_mask.len() + padding, 0);
        }
        let shape = [rows as i64, columns as i64];
        let tensor_error = |error: ort::Error| format!("could not build model input: {error}");
        let no_features =
            || Tensor::from_array(([0_i64, TEXT_HIDDEN_SIZE as i64], Vec::<f32>::new()));
        let outputs = self
            .session
            .run(ort::inputs! {
                "input_ids" => Tensor::from_array((shape, input_ids)).map_err(tensor_error)?,
                "attention_mask" => Tensor::from_array((shape, attention_mask)).map_err(tensor_error)?,
                "image_features" => no_features().map_err(tensor_error)?,
                "video_features" => no_features().map_err(tensor_error)?,
                "audio_features" => no_features().map_err(tensor_error)?,
            })
            .map_err(|error| format!("EmbeddingGemma 2 inference failed: {error}"))?;
        let (shape, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("could not read sentence embedding: {error}"))?;
        let width = shape.last().copied().unwrap_or(0) as usize;
        if shape.len() != 2 || shape[0] as usize != rows || width == 0 {
            return Err(format!("unexpected sentence embedding shape {shape:?}"));
        }
        Ok(data.chunks(width).map(<[f32]>::to_vec).collect())
    }
}

fn open_session(model: &Path) -> ort::Result<Session> {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?
        .commit_from_file(model)
}

/// `VECGRA_MODEL_DIR`, or the platform cache directory.
fn model_directory() -> Result<PathBuf, String> {
    let root = match env::var_os("VECGRA_MODEL_DIR") {
        Some(directory) if !directory.is_empty() => PathBuf::from(directory),
        _ => dirs::cache_dir()
            .ok_or("no cache directory found; set VECGRA_MODEL_DIR")?
            .join("vecgra")
            .join("models"),
    };
    Ok(root.join(GEMMA_MODEL).join(REVISION))
}

/// Downloads `file` unless a copy of the expected size is already cached.
fn ensure_file(directory: &Path, file: &ModelFile) -> Result<(), String> {
    let destination = directory.join(file.path);
    if fs::metadata(&destination).is_ok_and(|metadata| metadata.len() == file.bytes) {
        return Ok(());
    }
    let parent = destination.parent().unwrap_or(directory);
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let url = format!(
        "https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{}",
        file.path
    );
    eprintln!(
        "downloading EmbeddingGemma 2 {} ({:.1} MB) to {}",
        file.path,
        file.bytes as f64 / 1_000_000.0,
        destination.display()
    );
    let partial = destination.with_extension("partial");
    let result = download(&url, &partial, file).and_then(|()| {
        fs::rename(&partial, &destination)
            .map_err(|error| format!("could not move {}: {error}", partial.display()))
    });
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

fn download(url: &str, destination: &Path, file: &ModelFile) -> Result<(), String> {
    let response = ureq::get(url)
        .call()
        .map_err(|error| format!("could not download {url}: {error}"))?;
    let mut reader = response.into_body().into_reader();
    let mut writer = BufWriter::new(
        File::create(destination)
            .map_err(|error| format!("could not create {}: {error}", destination.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    let mut written = 0_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("download of {url} failed: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        writer
            .write_all(&buffer[..read])
            .map_err(|error| format!("could not write {}: {error}", destination.display()))?;
        written += read as u64;
    }
    writer
        .into_inner()
        .map_err(io::IntoInnerError::into_error)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("could not write {}: {error}", destination.display()))?;
    let digest = hex(&hasher.finalize());
    if written != file.bytes || digest != file.sha256 {
        return Err(format!(
            "{} failed verification: got {written} bytes with SHA-256 {digest}",
            file.path
        ));
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Keeps the leading Matryoshka prefix and restores unit length.
fn truncate(mut vector: Vec<f32>, dimension: usize) -> Result<Vec<f32>, String> {
    if vector.len() < dimension {
        return Err(format!(
            "EmbeddingGemma 2 returned a {}-dimensional embedding; {dimension} requested",
            vector.len()
        ));
    }
    vector.truncate(dimension);
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("embedding contains a non-finite value".into());
    }
    let magnitude = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if magnitude <= f32::EPSILON {
        return Err("EmbeddingGemma 2 returned a zero embedding".into());
    }
    for value in &mut vector {
        *value /= magnitude;
    }
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_batches_bound_padded_tokens() {
        assert_eq!(
            inference_batches(&[800; 8], 8),
            vec![0..2, 2..4, 4..6, 6..8]
        );
        assert_eq!(inference_batches(&[32; 9], 8), vec![0..8, 8..9]);
        // The sum fits, but padding every row to 1,000 tokens does not.
        assert_eq!(
            inference_batches(&[100, 100, 100, 1000], 8),
            vec![0..3, 3..4]
        );
    }

    #[test]
    fn long_inputs_run_alone_without_dropping_texts() {
        assert_eq!(inference_batches(&[3000, 8192], 8), vec![0..1, 1..2]);
    }

    #[test]
    fn inference_batches_honor_smaller_batch_limits_and_empty_input() {
        assert_eq!(inference_batches(&[32, 64, 128], 1), vec![0..1, 1..2, 2..3]);
        assert_eq!(inference_batches(&[32, 64], 0), vec![0..1, 1..2]);
        assert!(inference_batches(&[], 8).is_empty());
    }

    #[test]
    fn rejects_non_matryoshka_dimensions() {
        assert!(validate_gemma_dimension(256).is_ok());
        assert!(validate_gemma_dimension(300).is_err());
        assert!(gemma_documents(&["x".into()], 1024).is_err());
    }

    #[test]
    fn truncation_renormalizes() {
        let vector = truncate(vec![3.0, 4.0, 12.0], 2).unwrap();
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn empty_documents_do_not_report_completed_work() {
        let vectors = gemma_documents_with_progress(&[], 256, 8, |_| {
            panic!("empty input must not report progress")
        })
        .unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    #[ignore = "loads or downloads EmbeddingGemma 2"]
    fn reordered_batches_preserve_vectors_and_report_completion_across_windows() {
        let examples = [
            "Ownership and borrowing make Rust memory safe.".to_owned(),
            "Banana bread needs ripe fruit and flour.".repeat(12),
            "Cats sleep.".to_owned(),
            "An async function returns a future. Rust polls it to make progress.".repeat(4),
        ];
        let expected: Vec<_> = examples
            .iter()
            .map(|text| {
                gemma_documents(std::slice::from_ref(text), 256)
                    .unwrap()
                    .remove(0)
            })
            .collect();
        let texts: Vec<_> = (0..131).map(|index| examples[index % 4].clone()).collect();
        let mut completed = Vec::new();
        let vectors =
            gemma_documents_with_progress(&texts, 256, 8, |count| completed.push(count)).unwrap();
        assert_eq!(vectors.len(), 131);
        for (index, vector) in vectors.iter().enumerate() {
            let similarity: f32 = vector
                .iter()
                .zip(&expected[index % 4])
                .map(|(a, b)| a * b)
                .sum();
            assert!(similarity > 0.999, "input {index}: cosine {similarity}");
        }
        assert_eq!(completed.last(), Some(&131));
        let mut previous = 0;
        for count in completed {
            assert!(count > previous && count - previous <= 8);
            previous = count;
        }
    }

    /// Downloads the real model (~350 MB) on first run.
    #[test]
    #[ignore = "downloads EmbeddingGemma 2"]
    fn real_model_ranks_related_text_first() {
        let query = gemma_query("how does Rust prevent memory bugs", 256).unwrap();
        let documents = gemma_documents(
            &[
                "Ownership and borrowing make Rust memory safe.".into(),
                "Banana bread needs ripe fruit and flour.".into(),
            ],
            256,
        )
        .unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
        let related = dot(&query, &documents[0]);
        let unrelated = dot(&query, &documents[1]);
        eprintln!("related {related:.4} unrelated {unrelated:.4}");
        assert_eq!(query.len(), 256);
        assert!(related > unrelated + 0.1);
    }
}
