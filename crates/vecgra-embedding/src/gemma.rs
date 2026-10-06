//! EmbeddingGemma 2 served by a local Ollama instance.

use serde::{Deserialize, Serialize};
use std::env;

pub const GEMMA_MODEL: &str = "embeddinggemma-2";
/// Matryoshka sizes EmbeddingGemma 2 is trained to truncate to.
pub const GEMMA_DIMENSIONS: [usize; 4] = [768, 512, 256, 128];

const DEFAULT_OLLAMA_HOST: &str = "http://127.0.0.1:11434";
const QUERY_PROMPT: &str = "task: search result | query: ";
const DOCUMENT_PROMPT: &str = "title: none | text: ";

/// Embeds documents with the retrieval-document prompt.
pub fn gemma_documents(texts: &[String], dimension: usize) -> Result<Vec<Vec<f32>>, String> {
    let input: Vec<String> = texts
        .iter()
        .map(|text| format!("{DOCUMENT_PROMPT}{text}"))
        .collect();
    embed(&ollama_host(), &input, dimension)
}

/// Embeds one retrieval query with the search-query prompt.
pub fn gemma_query(text: &str, dimension: usize) -> Result<Vec<f32>, String> {
    let input = vec![format!("{QUERY_PROMPT}{text}")];
    embed(&ollama_host(), &input, dimension)?
        .pop()
        .ok_or_else(|| "Ollama returned no query embedding".into())
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

fn ollama_host() -> String {
    let host = env::var("OLLAMA_HOST").unwrap_or_default();
    let host = host.trim().trim_end_matches('/');
    if host.is_empty() {
        DEFAULT_OLLAMA_HOST.into()
    } else if host.contains("://") {
        host.into()
    } else {
        format!("http://{host}")
    }
}

fn embed(host: &str, input: &[String], dimension: usize) -> Result<Vec<Vec<f32>>, String> {
    validate_gemma_dimension(dimension)?;
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let request = EmbedRequest {
        model: GEMMA_MODEL,
        input,
        truncate: true,
    };
    let response: EmbedResponse = ureq::Agent::new_with_defaults()
        .post(&format!("{host}/api/embed"))
        .send_json(&request)
        .map_err(|error| {
            format!(
                "Ollama embedding request to {host} failed: {error}. Install Ollama, run \
                 `ollama pull {GEMMA_MODEL}`, or set OLLAMA_HOST"
            )
        })?
        .body_mut()
        .read_json()
        .map_err(|error| format!("could not decode Ollama response: {error}"))?;
    if response.embeddings.len() != input.len() {
        return Err(format!(
            "Ollama returned {} embeddings for {} inputs",
            response.embeddings.len(),
            input.len()
        ));
    }
    response
        .embeddings
        .into_iter()
        .map(|vector| truncate(vector, dimension))
        .collect()
}

/// Keeps the leading Matryoshka prefix and restores unit length.
fn truncate(mut vector: Vec<f32>, dimension: usize) -> Result<Vec<f32>, String> {
    if vector.len() < dimension {
        return Err(format!(
            "Ollama returned a {}-dimensional embedding; {dimension} requested",
            vector.len()
        ));
    }
    vector.truncate(dimension);
    if vector.iter().any(|value| !value.is_finite()) {
        return Err("embedding contains a non-finite value".into());
    }
    let magnitude = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if magnitude <= f32::EPSILON {
        return Err("Ollama returned a zero embedding".into());
    }
    for value in &mut vector {
        *value /= magnitude;
    }
    Ok(vector)
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    truncate: bool,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::thread;

    /// Answers one `/api/embed` request and returns the request body.
    fn serve_once(response: String) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host = format!("http://{}", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            assert!(request_line.starts_with("POST /api/embed "));
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            write!(
                reader.get_mut(),
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{response}",
                response.len()
            )
            .unwrap();
            String::from_utf8(body).unwrap()
        });
        (host, handle)
    }

    #[test]
    fn embeds_through_ollama_and_truncates_to_unit_prefix() {
        let mut full = vec![0.0_f32; 768];
        full[0] = 0.6;
        full[200] = 0.8;
        full[700] = 1.0;
        let response = format!(r#"{{"model":"{GEMMA_MODEL}","embeddings":[{full:?}]}}"#);
        let (host, handle) = serve_once(response);
        let input = vec![format!("{QUERY_PROMPT}ownership")];
        let vectors = embed(&host, &input, 256).unwrap();
        let body: serde_json::Value = serde_json::from_str(&handle.join().unwrap()).unwrap();
        assert_eq!(body["model"], GEMMA_MODEL);
        assert_eq!(body["input"][0], "task: search result | query: ownership");
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].len(), 256);
        assert!((vectors[0][0] - 0.6).abs() < 1e-6);
        assert!((vectors[0][200] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rejects_non_matryoshka_dimensions() {
        assert!(validate_gemma_dimension(256).is_ok());
        assert!(validate_gemma_dimension(300).is_err());
        assert!(embed(DEFAULT_OLLAMA_HOST, &["x".into()], 1024).is_err());
    }

    #[test]
    fn truncation_renormalizes() {
        let vector = truncate(vec![3.0, 4.0, 12.0], 2).unwrap();
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }
}
