//! Query embedding adapters shared by Vecgra tools.
//!
//! A database deliberately records only its vector dimension, not model
//! metadata. Callers choose the one embedding model used for their database.

use serde::{Deserialize, Serialize};
use std::env;
use std::thread;
use std::time::Duration;

const OPENROUTER_ENDPOINT: &str = "https://openrouter.ai/api/v1/embeddings";
pub const QWEN_MODEL: &str = "qwen/qwen3-embedding-8b";

/// Embeds one retrieval query with the model selected for a database.
pub fn embed_query(model: &str, dimension: usize, text: &str) -> Result<Vec<f32>, String> {
    if dimension == 0 {
        return Err("embedding dimension must be greater than zero".into());
    }
    match model {
        "hash" => Ok(feature_vector(text, dimension)),
        "qwen" | QWEN_MODEL => openrouter_qwen_query(dimension, text),
        other => Err(format!(
            "unknown embedder {other:?}; expected hash, qwen, or {QWEN_MODEL}"
        )),
    }
}

/// The deterministic lexical embedding used by local fixtures and offline use.
pub fn feature_vector(text: &str, dimension: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; dimension];
    if dimension == 0 {
        return vector;
    }
    let lowercase = text.to_ascii_lowercase();
    let mut populated = false;
    for token in lowercase
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter(|token| !token.is_empty())
    {
        add_feature(&mut vector, token.as_bytes(), 1.0);
        populated = true;
        if token.len() >= 3 {
            for trigram in token.as_bytes().windows(3) {
                add_feature(&mut vector, trigram, 0.2);
            }
        }
    }
    if !populated {
        add_feature(&mut vector, lowercase.as_bytes(), 1.0);
    }
    let magnitude = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if magnitude > f32::EPSILON {
        for value in &mut vector {
            *value /= magnitude;
        }
    } else {
        vector[0] = 1.0;
    }
    vector
}

fn openrouter_qwen_query(dimension: usize, text: &str) -> Result<Vec<f32>, String> {
    if dimension > 4096 {
        return Err("Qwen embedding dimension must be between 1 and 4096".into());
    }
    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| "OPENROUTER_API_KEY must be set for Qwen semantic search".to_string())?;
    if api_key.trim().is_empty() {
        return Err("OPENROUTER_API_KEY is empty".into());
    }
    let input = vec![format!(
        "Instruct: Retrieve graph elements relevant to the query\nQuery: {text}"
    )];
    let request = EmbeddingRequest {
        model: QWEN_MODEL,
        input: &input,
        dimensions: dimension,
    };
    let agent = ureq::Agent::new_with_defaults();
    let mut last_error = None;
    for attempt in 0..3 {
        let response = agent
            .post(OPENROUTER_ENDPOINT)
            .header("Authorization", &format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .send_json(&request);
        match response {
            Ok(mut response) => {
                let response: EmbeddingResponse = response
                    .body_mut()
                    .read_json()
                    .map_err(|error| format!("could not decode OpenRouter response: {error}"))?;
                let datum = response
                    .data
                    .into_iter()
                    .find(|datum| datum.index == 0)
                    .ok_or_else(|| "OpenRouter returned no query embedding".to_string())?;
                validate_vector(&datum.embedding, dimension)?;
                return Ok(datum.embedding);
            }
            Err(error) => {
                last_error = Some(error.to_string());
                if attempt < 2 {
                    thread::sleep(Duration::from_millis(350 * (1 << attempt)));
                }
            }
        }
    }
    Err(format!(
        "OpenRouter embedding request failed: {}",
        last_error.unwrap_or_else(|| "unknown error".into())
    ))
}

fn validate_vector(vector: &[f32], dimension: usize) -> Result<(), String> {
    if vector.len() != dimension {
        return Err(format!(
            "embedding dimension {} does not match database dimension {dimension}",
            vector.len()
        ));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("embedding contains a non-finite value".into());
    }
    Ok(())
}

fn add_feature(vector: &mut [f32], bytes: &[u8], weight: f32) {
    let hash = fnv1a(bytes);
    let index = (hash as usize) % vector.len();
    let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
    vector[index] += sign * weight;
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    dimensions: usize,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_vectors_are_normalized_and_related_tokens_overlap() {
        let left = feature_vector("rust function parser", 128);
        let related = feature_vector("parser function declaration", 128);
        let unrelated = feature_vector("database transaction checksum", 128);
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
        assert!((dot(&left, &left) - 1.0).abs() < 1e-5);
        assert!(dot(&left, &related) > dot(&left, &unrelated));
    }
}
