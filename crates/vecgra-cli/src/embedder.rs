use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const OPENROUTER_ENDPOINT: &str = "https://openrouter.ai/api/v1/embeddings";
const QWEN_MODEL: &str = "qwen/qwen3-embedding-8b";

pub(crate) trait Embedder {
    fn dimension(&self) -> usize;
    fn name(&self) -> &str;
    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn Error>>;
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, Box<dyn Error>>;
}

pub(crate) fn create_embedder(
    name: &str,
    dimension: usize,
    request_batch_size: usize,
) -> Result<Box<dyn Embedder>, Box<dyn Error>> {
    match name {
        "hash" => Ok(Box::new(HashEmbedder::new(dimension)?)),
        "qwen" | QWEN_MODEL => Ok(Box::new(OpenRouterEmbedder::qwen(
            dimension,
            request_batch_size,
        )?)),
        other => {
            Err(format!("unknown embedder {other:?}; expected hash, qwen, or {QWEN_MODEL}").into())
        }
    }
}

pub(crate) struct EmbeddingCache {
    embedder: Box<dyn Embedder>,
    vectors: HashMap<String, Arc<[f32]>>,
    embedded_texts: usize,
}

impl EmbeddingCache {
    pub(crate) fn new(embedder: Box<dyn Embedder>) -> Self {
        Self {
            embedder,
            vectors: HashMap::new(),
            embedded_texts: 0,
        }
    }

    pub(crate) fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    pub(crate) fn name(&self) -> &str {
        self.embedder.name()
    }

    pub(crate) fn embedded_texts(&self) -> usize {
        self.embedded_texts
    }

    pub(crate) fn ensure<'a>(
        &mut self,
        texts: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), Box<dyn Error>> {
        let mut seen = HashSet::new();
        let missing: Vec<String> = texts
            .into_iter()
            .filter(|text| !self.vectors.contains_key(*text))
            .filter(|text| seen.insert((*text).to_owned()))
            .map(str::to_owned)
            .collect();

        let vectors = self.embedder.embed_documents(&missing)?;
        if vectors.len() != missing.len() {
            return Err(format!(
                "embedder returned {} vectors for {} inputs",
                vectors.len(),
                missing.len()
            )
            .into());
        }
        for (text, vector) in missing.into_iter().zip(vectors) {
            validate_dimension(&vector, self.dimension())?;
            self.vectors.insert(text, vector.into());
            self.embedded_texts += 1;
        }
        Ok(())
    }

    pub(crate) fn vector(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        self.vectors
            .get(text)
            .map(|vector| vector.to_vec())
            .ok_or_else(|| format!("embedding was not prepared for {text:?}").into())
    }
}

pub(crate) struct HashEmbedder {
    dimension: usize,
}

impl HashEmbedder {
    fn new(dimension: usize) -> Result<Self, Box<dyn Error>> {
        if dimension == 0 {
            return Err("embedding dimension must be greater than zero".into());
        }
        Ok(Self { dimension })
    }
}

impl Embedder for HashEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "hash"
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        Ok(texts
            .iter()
            .map(|text| feature_vector(text, self.dimension))
            .collect())
    }

    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(feature_vector(text, self.dimension))
    }
}

#[derive(Clone)]
pub(crate) struct OpenRouterEmbedder {
    agent: ureq::Agent,
    api_key: String,
    model: String,
    dimension: usize,
    request_batch_size: usize,
    concurrency: usize,
}

impl OpenRouterEmbedder {
    fn qwen(dimension: usize, request_batch_size: usize) -> Result<Self, Box<dyn Error>> {
        if dimension == 0 || dimension > 4096 {
            return Err("Qwen embedding dimension must be between 1 and 4096".into());
        }
        let api_key = env::var("OPENROUTER_API_KEY")
            .map_err(|_| "OPENROUTER_API_KEY must be set when using the qwen embedder")?;
        if api_key.trim().is_empty() {
            return Err("OPENROUTER_API_KEY is empty".into());
        }
        let concurrency = env::var("VECGRA_EMBEDDING_CONCURRENCY")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|error| format!("invalid VECGRA_EMBEDDING_CONCURRENCY: {error}"))?
            .unwrap_or(4)
            .clamp(1, 32);
        Ok(Self {
            agent: ureq::Agent::new_with_defaults(),
            api_key,
            model: QWEN_MODEL.into(),
            dimension,
            request_batch_size: request_batch_size.max(1),
            concurrency,
        })
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let request = EmbeddingRequest {
            model: &self.model,
            input: texts,
            dimensions: self.dimension,
        };
        let mut last_error = None;
        for attempt in 0..5 {
            let response = self
                .agent
                .post(OPENROUTER_ENDPOINT)
                .header("Authorization", &format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .send_json(&request);
            match response {
                Ok(mut response) => {
                    let response: EmbeddingResponse = response.body_mut().read_json()?;
                    return ordered_embeddings(response, texts.len(), self.dimension);
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                    if attempt < 4 {
                        thread::sleep(Duration::from_millis(500 * (1 << attempt)));
                    }
                }
            }
        }
        Err(format!(
            "OpenRouter embedding request failed after retries: {}",
            last_error.unwrap_or_else(|| "unknown error".into())
        )
        .into())
    }

    fn embed_documents_parallel(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch_count = texts.len().div_ceil(self.request_batch_size);
        if batch_count == 1 {
            return self.embed(texts);
        }

        let worker_count = self.concurrency.min(batch_count);
        let completed = AtomicUsize::new(0);
        let (sender, receiver) = mpsc::channel();
        thread::scope(|scope| {
            for worker in 0..worker_count {
                let sender = sender.clone();
                let embedder = self.clone();
                let completed = &completed;
                scope.spawn(move || {
                    for batch_index in (worker..batch_count).step_by(worker_count) {
                        let start = batch_index * embedder.request_batch_size;
                        let end = (start + embedder.request_batch_size).min(texts.len());
                        let result = embedder.embed(&texts[start..end]).map_err(|error| {
                            format!("embedding batch {batch_index} failed: {error}")
                        });
                        if sender.send((batch_index, result)).is_err() {
                            return;
                        }
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == batch_count || done.is_multiple_of(10) {
                            eprintln!("embedded {done}/{batch_count} batches");
                        }
                    }
                });
            }
            drop(sender);

            let mut batches: Vec<Option<Vec<Vec<f32>>>> = vec![None; batch_count];
            let mut first_error = None;
            for (batch_index, result) in receiver {
                match result {
                    Ok(vectors) => batches[batch_index] = Some(vectors),
                    Err(error) if first_error.is_none() => first_error = Some(error),
                    Err(_) => {}
                }
            }
            if let Some(error) = first_error {
                return Err(error.into());
            }
            let mut vectors = Vec::with_capacity(texts.len());
            for (index, batch) in batches.into_iter().enumerate() {
                vectors.extend(batch.ok_or_else(|| format!("embedding batch {index} was lost"))?);
            }
            Ok(vectors)
        })
    }
}

impl Embedder for OpenRouterEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        QWEN_MODEL
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        self.embed_documents_parallel(texts)
    }

    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let input = vec![format!(
            "Instruct: Retrieve graph elements relevant to the query\nQuery: {text}"
        )];
        let mut vectors = self.embed(&input)?;
        vectors
            .pop()
            .ok_or_else(|| "embedder returned no query vector".into())
    }
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

fn ordered_embeddings(
    response: EmbeddingResponse,
    expected: usize,
    dimension: usize,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    if response.data.len() != expected {
        return Err(format!(
            "OpenRouter returned {} embeddings for {expected} inputs",
            response.data.len()
        )
        .into());
    }
    let mut ordered = vec![None; expected];
    for datum in response.data {
        if datum.index >= expected {
            return Err(format!("OpenRouter returned invalid input index {}", datum.index).into());
        }
        validate_dimension(&datum.embedding, dimension)?;
        if ordered[datum.index].replace(datum.embedding).is_some() {
            return Err(
                format!("OpenRouter returned duplicate input index {}", datum.index).into(),
            );
        }
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, vector)| {
            vector.ok_or_else(|| format!("OpenRouter omitted input index {index}").into())
        })
        .collect()
}

fn validate_dimension(vector: &[f32], dimension: usize) -> Result<(), Box<dyn Error>> {
    if vector.len() != dimension {
        return Err(format!(
            "embedding dimension {} does not match requested dimension {dimension}",
            vector.len()
        )
        .into());
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("embedding contains a non-finite value".into());
    }
    Ok(())
}

pub(crate) fn feature_vector(text: &str, dimension: usize) -> Vec<f32> {
    vecgra_embedding::feature_vector(text, dimension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_vectors_are_normalized_and_related_tokens_overlap() {
        let left = feature_vector("rust function parser", 128);
        let right = feature_vector("parser function declaration", 128);
        let unrelated = feature_vector("database transaction checksum", 128);
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
        assert!((dot(&left, &left) - 1.0).abs() < 1e-5);
        assert!(dot(&left, &right) > dot(&left, &unrelated));
    }

    #[test]
    fn cache_deduplicates_texts() {
        let embedder = create_embedder("hash", 8, 2).unwrap();
        let mut cache = EmbeddingCache::new(embedder);
        cache.ensure(["same", "same", "different"]).unwrap();
        assert_eq!(cache.embedded_texts(), 2);
        assert_eq!(cache.vector("same").unwrap().len(), 8);
    }
}
