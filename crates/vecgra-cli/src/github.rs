use crate::embedder::{Embedder, EmbeddingCache};
use chrono::DateTime;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value as JsonValue, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;
use vecgra::{BulkLoader, DatabaseOptions, Similarity, Value, VectorEncoding};

mod model;

use model::*;

const GITHUB_GRAPHQL_ENDPOINT: &str = "https://api.github.com/graphql";
const PAGE_SIZE: usize = 20;
const PULL_REQUEST_PAGE_SIZE: usize = 5;

const REPOSITORY_QUERY: &str = r#"
query Repository($owner:String!,$name:String!){
  repository(owner:$owner,name:$name){
    id name nameWithOwner description url createdAt updatedAt
    stargazerCount forkCount isArchived isFork
    primaryLanguage{name}
    repositoryTopics(first:20){nodes{topic{name}}}
    issues{totalCount}
    pullRequests{totalCount}
    discussions{totalCount}
    releases{totalCount}
  }
}
"#;

const ISSUES_QUERY: &str = r#"
query Issues($owner:String!,$name:String!,$first:Int!,$after:String){
  repository(owner:$owner,name:$name){
    issues(first:$first,after:$after,orderBy:{field:UPDATED_AT,direction:DESC}){
      totalCount pageInfo{hasNextPage endCursor}
      nodes{
        id number title bodyText url state createdAt updatedAt closedAt
        author{login avatarUrl url}
        assignees(first:10){nodes{login avatarUrl url}}
        labels(first:20){nodes{id name color description url}}
        milestone{id number title description state dueOn createdAt updatedAt url}
        comments(first:20){
          totalCount
          nodes{id bodyText url createdAt updatedAt author{login avatarUrl url}}
        }
        closedByPullRequestsReferences(first:20){
          nodes{
            id number title bodyText url state createdAt updatedAt closedAt merged mergedAt
            author{login avatarUrl url}
          }
        }
      }
    }
  }
}
"#;

const PULL_REQUESTS_QUERY: &str = r#"
query PullRequests($owner:String!,$name:String!,$first:Int!,$after:String){
  repository(owner:$owner,name:$name){
    pullRequests(first:$first,after:$after,orderBy:{field:UPDATED_AT,direction:DESC}){
      totalCount pageInfo{hasNextPage endCursor}
      nodes{
        id number title bodyText url state isDraft merged
        createdAt updatedAt closedAt mergedAt
        additions deletions changedFiles baseRefName headRefName
        author{login avatarUrl url}
        assignees(first:10){nodes{login avatarUrl url}}
        labels(first:20){nodes{id name color description url}}
        milestone{id number title description state dueOn createdAt updatedAt url}
        comments(first:20){
          totalCount
          nodes{id bodyText url createdAt updatedAt author{login avatarUrl url}}
        }
        reviews(first:20){
          totalCount
          nodes{id bodyText url state submittedAt createdAt updatedAt author{login avatarUrl url}}
        }
        commits(first:40){
          totalCount
          nodes{commit{
            id oid messageHeadline messageBody url authoredDate committedDate
            author{name user{login avatarUrl url}}
          }}
        }
        files(first:50){totalCount nodes{path additions deletions changeType}}
        closingIssuesReferences(first:20){
          nodes{id number title bodyText url state createdAt updatedAt closedAt author{login avatarUrl url}}
        }
      }
    }
  }
}
"#;

const PULL_REQUESTS_LITE_QUERY: &str = r#"
query PullRequestsLite($owner:String!,$name:String!,$first:Int!,$after:String){
  repository(owner:$owner,name:$name){
    pullRequests(first:$first,after:$after,orderBy:{field:UPDATED_AT,direction:DESC}){
      totalCount pageInfo{hasNextPage endCursor}
      nodes{
        id number title bodyText url state isDraft merged
        createdAt updatedAt closedAt mergedAt
        additions deletions changedFiles baseRefName headRefName
        author{login avatarUrl url}
        comments{totalCount}
        reviews{totalCount}
        commits{totalCount}
        files(first:1){totalCount}
      }
    }
  }
}
"#;

const DISCUSSIONS_QUERY: &str = r#"
query Discussions($owner:String!,$name:String!,$first:Int!,$after:String){
  repository(owner:$owner,name:$name){
    discussions(first:$first,after:$after,orderBy:{field:UPDATED_AT,direction:DESC}){
      totalCount pageInfo{hasNextPage endCursor}
      nodes{
        id number title bodyText url createdAt updatedAt closed locked
        author{login avatarUrl url}
        category{id name description emoji}
        answer{id}
        comments(first:40){
          totalCount
          nodes{
            id bodyText url createdAt updatedAt author{login avatarUrl url}
            replies(first:20){
              totalCount
              nodes{id bodyText url createdAt updatedAt author{login avatarUrl url}}
            }
          }
        }
      }
    }
  }
}
"#;

const RELEASES_QUERY: &str = r#"
query Releases($owner:String!,$name:String!,$first:Int!,$after:String){
  repository(owner:$owner,name:$name){
    releases(first:$first,after:$after,orderBy:{field:CREATED_AT,direction:DESC}){
      totalCount pageInfo{hasNextPage endCursor}
      nodes{
        id name tagName description url createdAt publishedAt updatedAt isDraft isPrerelease
        author{login avatarUrl url}
        tagCommit{oid url}
      }
    }
  }
}
"#;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GithubImportLimits {
    pub(crate) issues: usize,
    pub(crate) pull_requests: usize,
    pub(crate) discussions: usize,
    pub(crate) releases: usize,
}

pub(crate) fn import_github_repository(
    repository: &str,
    database_path: &Path,
    limits: GithubImportLimits,
    embedder: Box<dyn Embedder>,
) -> Result<(), Box<dyn Error>> {
    if database_path.try_exists()? {
        return Err(format!(
            "bulk-load destination already exists: {}",
            database_path.display()
        )
        .into());
    }
    let (owner, name) = parse_repository(repository)?;
    let cache_path = cache_path(database_path);
    let client = GithubClient::new(cache_path.clone())?;
    let started = Instant::now();
    let mut graph = EngineeringGraph::default();

    let repository_data: RepositoryResponse = client.request(
        "repository",
        REPOSITORY_QUERY,
        &json!({"owner":owner,"name":name}),
    )?;
    let repository = repository_data
        .repository
        .ok_or_else(|| format!("GitHub repository {owner}/{name} was not found"))?;
    let repository_key = repository_key(&repository.id);
    graph.upsert_node(repository_key.clone(), repository_node(&repository), 3);
    for topic in &repository.repository_topics.nodes {
        let key = format!("topic:{}", topic.topic.name.to_lowercase());
        graph.upsert_node(
            key.clone(),
            GraphNode::new(
                "Topic",
                vec![("name", string_value(&topic.topic.name))],
                vec![format!("GitHub repository topic {}", topic.topic.name)],
                topic.topic.name.clone(),
            ),
            2,
        );
        graph.add_edge(
            &repository_key,
            &key,
            "HAS_TOPIC",
            Vec::new(),
            format!(
                "{} has repository topic {}",
                repository.name_with_owner, topic.topic.name
            ),
        );
    }

    let issue_count = crawl_issues(
        &client,
        &owner,
        &name,
        limits.issues,
        &repository_key,
        &mut graph,
    )?;
    let pull_request_count = crawl_pull_requests(
        &client,
        &owner,
        &name,
        limits.pull_requests,
        &repository_key,
        &mut graph,
    )?;
    let discussion_count = crawl_discussions(
        &client,
        &owner,
        &name,
        limits.discussions,
        &repository_key,
        &mut graph,
    )?;
    let release_count = crawl_releases(
        &client,
        &owner,
        &name,
        limits.releases,
        &repository_key,
        &mut graph,
    )?;

    graph.resolve_stubs();
    let mut embeddings = EmbeddingCache::new(embedder);
    let embedding_text_count = graph.embedding_text_count();
    eprintln!(
        "embedding {embedding_text_count} node/relationship payloads for {} nodes and {} edges with {}",
        graph.nodes.len(),
        graph.edges.len(),
        embeddings.name()
    );
    embeddings.ensure(graph.embedding_texts())?;

    let mut database = BulkLoader::new(
        database_path,
        DatabaseOptions {
            vector_dimension: embeddings.dimension(),
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F16,
            sync_on_commit: true,
        },
    )?;
    let mut node_ids = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let vectors = node
            .embedding_texts
            .iter()
            .map(|text| embeddings.vector(text))
            .collect::<Result<Vec<_>, _>>()?;
        node_ids.push(database.create_node(node.label, node.properties.clone(), &vectors)?);
    }
    for edge in &graph.edges {
        let source = node_ids[*graph
            .node_indexes
            .get(&edge.source)
            .ok_or_else(|| format!("missing source node {}", edge.source))?];
        let target = node_ids[*graph
            .node_indexes
            .get(&edge.target)
            .ok_or_else(|| format!("missing target node {}", edge.target))?];
        let vector = embeddings.vector(&edge.embedding_text)?;
        database.create_edge(
            source,
            target,
            edge.label,
            edge.properties.clone(),
            &[vector],
        )?;
    }
    let stats = database.finish()?;

    println!("database\t{}", database_path.display());
    println!("source\thttps://github.com/{owner}/{name}");
    println!("cache\t{}", cache_path.display());
    println!("issues\t{issue_count}");
    println!("pull_requests\t{pull_request_count}");
    println!("discussions\t{discussion_count}");
    println!("releases\t{release_count}");
    println!("nodes\t{}", stats.nodes);
    println!("edges\t{}", stats.edges);
    println!("vectors\t{}", stats.indexed_vectors);
    println!("embedder\t{}", embeddings.name());
    println!("unique_embedding_texts\t{}", embeddings.embedded_texts());
    println!("elapsed_ms\t{}", started.elapsed().as_millis());
    for (label, count) in graph.node_label_counts() {
        println!("node_label.{label}\t{count}");
    }
    for (label, count) in graph.edge_label_counts() {
        println!("edge_label.{label}\t{count}");
    }
    Ok(())
}

fn crawl_issues(
    client: &GithubClient,
    owner: &str,
    name: &str,
    limit: usize,
    repository_key: &str,
    graph: &mut EngineeringGraph,
) -> Result<usize, Box<dyn Error>> {
    let mut cursor = None;
    let mut imported = 0;
    let mut page = 0;
    while imported < limit {
        let first = PAGE_SIZE.min(limit - imported);
        let response: IssuesResponse = client.request(
            &format!("issues-{page:05}"),
            ISSUES_QUERY,
            &json!({"owner":owner,"name":name,"first":first,"after":cursor}),
        )?;
        let connection = response
            .repository
            .ok_or("repository disappeared while crawling issues")?
            .issues;
        if connection.nodes.is_empty() {
            break;
        }
        for issue in &connection.nodes {
            add_issue(graph, repository_key, issue, true);
        }
        imported += connection.nodes.len();
        page += 1;
        eprintln!(
            "crawled {imported}/{} issues (repository total {})",
            limit.min(connection.total_count),
            connection.total_count
        );
        if !connection.page_info.has_next_page {
            break;
        }
        cursor = connection.page_info.end_cursor;
    }
    Ok(imported)
}

fn crawl_pull_requests(
    client: &GithubClient,
    owner: &str,
    name: &str,
    limit: usize,
    repository_key: &str,
    graph: &mut EngineeringGraph,
) -> Result<usize, Box<dyn Error>> {
    let mut cursor = None;
    let mut imported = 0;
    let mut page = 0;
    while imported < limit {
        let first = PULL_REQUEST_PAGE_SIZE.min(limit - imported);
        let variables = json!({"owner":owner,"name":name,"first":first,"after":cursor});
        let rich_response: Result<PullRequestsResponse, Box<dyn Error>> = client.request(
            &format!("pull-requests-{page:05}"),
            PULL_REQUESTS_QUERY,
            &variables,
        );
        let (response, detail_complete) = match rich_response {
            Ok(response) => (response, true),
            Err(error) => {
                eprintln!(
                    "rich pull-request page {page} failed ({error}); continuing with its lite shape"
                );
                (
                    client.request(
                        &format!("pull-requests-lite-{page:05}"),
                        PULL_REQUESTS_LITE_QUERY,
                        &variables,
                    )?,
                    false,
                )
            }
        };
        let connection = response
            .repository
            .ok_or("repository disappeared while crawling pull requests")?
            .pull_requests;
        if connection.nodes.is_empty() {
            break;
        }
        for pull_request in &connection.nodes {
            add_pull_request(graph, repository_key, pull_request, detail_complete);
        }
        imported += connection.nodes.len();
        page += 1;
        eprintln!(
            "crawled {imported}/{} pull requests (repository total {})",
            limit.min(connection.total_count),
            connection.total_count
        );
        if !connection.page_info.has_next_page {
            break;
        }
        cursor = connection.page_info.end_cursor;
    }
    Ok(imported)
}

fn crawl_discussions(
    client: &GithubClient,
    owner: &str,
    name: &str,
    limit: usize,
    repository_key: &str,
    graph: &mut EngineeringGraph,
) -> Result<usize, Box<dyn Error>> {
    let mut cursor = None;
    let mut imported = 0;
    let mut page = 0;
    while imported < limit {
        let first = PAGE_SIZE.min(limit - imported);
        let response: DiscussionsResponse = client.request(
            &format!("discussions-{page:05}"),
            DISCUSSIONS_QUERY,
            &json!({"owner":owner,"name":name,"first":first,"after":cursor}),
        )?;
        let connection = response
            .repository
            .ok_or("repository disappeared while crawling discussions")?
            .discussions;
        if connection.nodes.is_empty() {
            break;
        }
        for discussion in &connection.nodes {
            add_discussion(graph, repository_key, discussion);
        }
        imported += connection.nodes.len();
        page += 1;
        eprintln!(
            "crawled {imported}/{} discussions (repository total {})",
            limit.min(connection.total_count),
            connection.total_count
        );
        if !connection.page_info.has_next_page {
            break;
        }
        cursor = connection.page_info.end_cursor;
    }
    Ok(imported)
}

fn crawl_releases(
    client: &GithubClient,
    owner: &str,
    name: &str,
    limit: usize,
    repository_key: &str,
    graph: &mut EngineeringGraph,
) -> Result<usize, Box<dyn Error>> {
    let mut cursor = None;
    let mut imported = 0;
    let mut page = 0;
    while imported < limit {
        let first = PAGE_SIZE.min(limit - imported);
        let response: ReleasesResponse = client.request(
            &format!("releases-{page:05}"),
            RELEASES_QUERY,
            &json!({"owner":owner,"name":name,"first":first,"after":cursor}),
        )?;
        let connection = response
            .repository
            .ok_or("repository disappeared while crawling releases")?
            .releases;
        if connection.nodes.is_empty() {
            break;
        }
        for release in &connection.nodes {
            add_release(graph, repository_key, release);
        }
        imported += connection.nodes.len();
        page += 1;
        eprintln!(
            "crawled {imported}/{} releases (repository total {})",
            limit.min(connection.total_count),
            connection.total_count
        );
        if !connection.page_info.has_next_page {
            break;
        }
        cursor = connection.page_info.end_cursor;
    }
    Ok(imported)
}

fn parse_repository(value: &str) -> Result<(String, String), Box<dyn Error>> {
    let trimmed = value
        .trim()
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .strip_prefix("https://github.com/")
        .or_else(|| {
            value
                .trim()
                .trim_end_matches('/')
                .trim_end_matches(".git")
                .strip_prefix("http://github.com/")
        })
        .unwrap_or_else(|| value.trim().trim_end_matches('/').trim_end_matches(".git"));
    let mut parts = trimmed.split('/');
    let Some(owner) = parts.next().filter(|value| !value.is_empty()) else {
        return Err(format!("expected GitHub repository as owner/name, got {value:?}").into());
    };
    let Some(name) = parts.next().filter(|value| !value.is_empty()) else {
        return Err(format!("expected GitHub repository as owner/name, got {value:?}").into());
    };
    if parts.next().is_some() {
        return Err(format!("expected GitHub repository as owner/name, got {value:?}").into());
    }
    Ok((owner.to_owned(), name.to_owned()))
}

fn cache_path(database_path: &Path) -> PathBuf {
    if let Some(path) = env::var_os("VECGRA_GITHUB_CACHE") {
        return PathBuf::from(path);
    }
    if let Some(path) = dirs::cache_dir() {
        return path.join("vecgra").join("github");
    }
    let file_name = database_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("vecgra")
        .to_owned()
        + ".github-cache";
    database_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(file_name)
}

struct GithubClient {
    agent: ureq::Agent,
    token: String,
    cache_path: PathBuf,
}

impl GithubClient {
    fn new(cache_path: PathBuf) -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all(&cache_path)?;
        Ok(Self {
            agent: ureq::Agent::new_with_defaults(),
            token: github_token()?,
            cache_path,
        })
    }

    fn request<T: DeserializeOwned>(
        &self,
        cache_name: &str,
        query: &str,
        variables: &JsonValue,
    ) -> Result<T, Box<dyn Error>> {
        let request_fingerprint = request_fingerprint(query, variables);
        let cache_file = self
            .cache_path
            .join(format!("{cache_name}-{request_fingerprint:016x}.json"));
        let body = if cache_file.try_exists()? {
            fs::read_to_string(&cache_file)?
        } else {
            let payload = json!({"query":query,"variables":variables});
            let mut attempt = 0_u32;
            let mut response = loop {
                match self
                    .agent
                    .post(GITHUB_GRAPHQL_ENDPOINT)
                    .header("Authorization", &format!("Bearer {}", self.token))
                    .header("Content-Type", "application/json")
                    .header("User-Agent", "Vecgra-GitHub-Importer")
                    .send_json(payload.clone())
                {
                    Ok(response) => break response,
                    Err(error)
                        if attempt < 4
                            && matches!(error, ureq::Error::StatusCode(429 | 500..=599)) =>
                    {
                        let delay = std::time::Duration::from_millis(250 * (1 << attempt));
                        eprintln!(
                            "GitHub request {cache_name} failed ({error}); retrying in {} ms",
                            delay.as_millis()
                        );
                        std::thread::sleep(delay);
                        attempt += 1;
                    }
                    Err(error) => return Err(error.into()),
                }
            };
            let body = response.body_mut().read_to_string()?;
            let temporary = self
                .cache_path
                .join(format!(".{cache_name}.{}.tmp", std::process::id()));
            fs::write(&temporary, &body)?;
            fs::rename(temporary, &cache_file)?;
            body
        };
        let envelope: GraphQlEnvelope<T> = match serde_json::from_str(&body) {
            Ok(envelope) => envelope,
            Err(error) => {
                let _ = fs::remove_file(&cache_file);
                return Err(format!(
                    "invalid GitHub response {}: {error}; discarded it so the crawl can retry",
                    cache_file.display()
                )
                .into());
            }
        };
        if !envelope.errors.is_empty() {
            let messages = envelope
                .errors
                .into_iter()
                .map(|error| error.message)
                .collect::<Vec<_>>()
                .join("; ");
            let _ = fs::remove_file(&cache_file);
            return Err(format!("GitHub GraphQL request failed: {messages}").into());
        }
        envelope
            .data
            .ok_or_else(|| "GitHub GraphQL response did not contain data".into())
    }
}

fn request_fingerprint(query: &str, variables: &JsonValue) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in query
        .as_bytes()
        .iter()
        .copied()
        .chain(variables.to_string().bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn github_token() -> Result<String, Box<dyn Error>> {
    for variable in ["GITHUB_TOKEN", "GH_TOKEN"] {
        if let Ok(token) = env::var(variable)
            && !token.trim().is_empty()
        {
            return Ok(token);
        }
    }
    let output = Command::new("gh")
        .args(["auth", "token"])
        .output()
        .map_err(|_| "set GITHUB_TOKEN/GH_TOKEN or install and authenticate GitHub CLI")?;
    if !output.status.success() {
        return Err("set GITHUB_TOKEN/GH_TOKEN or run `gh auth login`".into());
    }
    let token = String::from_utf8(output.stdout)?.trim().to_owned();
    if token.is_empty() {
        return Err("GitHub authentication returned an empty token".into());
    }
    Ok(token)
}

#[derive(Deserialize)]
struct GraphQlEnvelope<T> {
    data: Option<T>,
    #[serde(default)]
    errors: Vec<GraphQlError>,
}

#[derive(Deserialize)]
struct GraphQlError {
    message: String,
}

#[derive(Deserialize)]
struct RepositoryResponse {
    repository: Option<Repository>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Repository {
    id: String,
    name: String,
    name_with_owner: String,
    description: Option<String>,
    url: String,
    created_at: String,
    updated_at: String,
    stargazer_count: i64,
    fork_count: i64,
    is_archived: bool,
    is_fork: bool,
    primary_language: Option<NamedValue>,
    repository_topics: Connection<RepositoryTopic>,
    issues: CountConnection,
    pull_requests: CountConnection,
    discussions: CountConnection,
    releases: CountConnection,
}

#[derive(Deserialize)]
struct NamedValue {
    name: String,
}

#[derive(Deserialize)]
struct RepositoryTopic {
    topic: NamedValue,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CountConnection {
    total_count: usize,
}

#[derive(Deserialize)]
struct IssuesResponse {
    repository: Option<IssuesRepository>,
}

#[derive(Deserialize)]
struct IssuesRepository {
    issues: Connection<Issue>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PullRequestsResponse {
    repository: Option<PullRequestsRepository>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PullRequestsRepository {
    pull_requests: Connection<PullRequest>,
}

#[derive(Deserialize)]
struct DiscussionsResponse {
    repository: Option<DiscussionsRepository>,
}

#[derive(Deserialize)]
struct DiscussionsRepository {
    discussions: Connection<Discussion>,
}

#[derive(Deserialize)]
struct ReleasesResponse {
    repository: Option<ReleasesRepository>,
}

#[derive(Deserialize)]
struct ReleasesRepository {
    releases: Connection<Release>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
struct Connection<T> {
    #[serde(default)]
    total_count: usize,
    #[serde(default, deserialize_with = "deserialize_connection_nodes")]
    nodes: Vec<T>,
    #[serde(default)]
    page_info: PageInfo,
}

fn deserialize_connection_nodes<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(Option::<Vec<Option<T>>>::deserialize(deserializer)?
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .collect())
}

fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

impl<T> Default for Connection<T> {
    fn default() -> Self {
        Self {
            total_count: 0,
            nodes: Vec::new(),
            page_info: PageInfo::default(),
        }
    }
}

#[derive(Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PageInfo {
    has_next_page: bool,
    end_cursor: Option<String>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Actor {
    login: String,
    avatar_url: String,
    url: String,
}

#[derive(Clone, Deserialize)]
struct Label {
    id: String,
    name: String,
    color: String,
    description: Option<String>,
    url: String,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Milestone {
    id: String,
    number: i64,
    title: String,
    description: Option<String>,
    state: String,
    due_on: Option<String>,
    created_at: String,
    updated_at: String,
    url: String,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Comment {
    id: String,
    body_text: String,
    url: String,
    created_at: String,
    updated_at: String,
    author: Option<Actor>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    replies: Connection<Comment>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Issue {
    id: String,
    number: i64,
    title: String,
    body_text: String,
    url: String,
    state: String,
    created_at: String,
    updated_at: String,
    closed_at: Option<String>,
    author: Option<Actor>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    assignees: Connection<Actor>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    labels: Connection<Label>,
    milestone: Option<Milestone>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    comments: Connection<Comment>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    closed_by_pull_requests_references: Connection<PullRequest>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PullRequest {
    id: String,
    number: i64,
    title: String,
    body_text: String,
    url: String,
    state: String,
    #[serde(default)]
    is_draft: bool,
    #[serde(default)]
    merged: bool,
    created_at: String,
    updated_at: String,
    closed_at: Option<String>,
    merged_at: Option<String>,
    #[serde(default)]
    additions: i64,
    #[serde(default)]
    deletions: i64,
    #[serde(default)]
    changed_files: i64,
    #[serde(default)]
    base_ref_name: String,
    #[serde(default)]
    head_ref_name: String,
    author: Option<Actor>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    assignees: Connection<Actor>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    labels: Connection<Label>,
    milestone: Option<Milestone>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    comments: Connection<Comment>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    reviews: Connection<Review>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    commits: Connection<PullRequestCommit>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    files: Connection<ChangedFile>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    closing_issues_references: Connection<Issue>,
}

#[derive(Clone, Deserialize)]
struct PullRequestCommit {
    commit: Commit,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Review {
    id: String,
    body_text: String,
    url: String,
    state: String,
    submitted_at: Option<String>,
    created_at: String,
    updated_at: String,
    author: Option<Actor>,
}

#[derive(Clone, Deserialize)]
struct CommitAuthor {
    name: Option<String>,
    user: Option<Actor>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Commit {
    id: String,
    oid: String,
    message_headline: String,
    message_body: String,
    url: String,
    authored_date: String,
    committed_date: String,
    author: Option<CommitAuthor>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChangedFile {
    path: String,
    additions: i64,
    deletions: i64,
    change_type: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Discussion {
    id: String,
    number: i64,
    title: String,
    body_text: String,
    url: String,
    created_at: String,
    updated_at: String,
    closed: bool,
    locked: bool,
    author: Option<Actor>,
    category: DiscussionCategory,
    answer: Option<Answer>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    comments: Connection<Comment>,
}

#[derive(Deserialize)]
struct DiscussionCategory {
    id: String,
    name: String,
    description: Option<String>,
    emoji: String,
}

#[derive(Deserialize)]
struct Answer {
    id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Release {
    id: String,
    name: Option<String>,
    tag_name: String,
    description: Option<String>,
    url: String,
    created_at: String,
    published_at: Option<String>,
    updated_at: String,
    is_draft: bool,
    is_prerelease: bool,
    author: Option<Actor>,
    tag_commit: Option<TagCommit>,
}

#[derive(Deserialize)]
struct TagCommit {
    oid: String,
    url: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_parser_accepts_slug_and_url() {
        assert_eq!(
            parse_repository("zed-industries/zed").unwrap(),
            ("zed-industries".into(), "zed".into())
        );
        assert_eq!(
            parse_repository("https://github.com/zed-industries/zed.git").unwrap(),
            ("zed-industries".into(), "zed".into())
        );
        assert!(parse_repository("not-enough").is_err());
    }

    #[test]
    fn graph_upgrades_stubs_and_deduplicates_relationships() {
        let mut graph = EngineeringGraph::default();
        graph.upsert_node(
            "issue:1".into(),
            GraphNode::new("Issue", Vec::new(), vec!["stub".into()], "stub".into()),
            1,
        );
        graph.upsert_node(
            "issue:1".into(),
            GraphNode::new("Issue", Vec::new(), vec!["full".into()], "full".into()),
            3,
        );
        graph.upsert_node(
            "user:a".into(),
            GraphNode::new("User", Vec::new(), vec!["a".into()], "a".into()),
            2,
        );
        graph.add_edge(
            "user:a",
            "issue:1",
            "AUTHORED",
            Vec::new(),
            "a authored issue".into(),
        );
        graph.add_edge(
            "user:a",
            "issue:1",
            "AUTHORED",
            Vec::new(),
            "duplicate".into(),
        );

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].summary, "full");
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn utf8_truncation_preserves_boundaries() {
        let value = "hello 🦀 graph";
        let truncated = truncate_utf8(value, 8);
        assert!(truncated.is_char_boundary(truncated.len()));
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn github_connections_tolerate_deleted_null_nodes() {
        let connection: Connection<NamedValue> = serde_json::from_str(
            r#"{"totalCount":3,"nodes":[{"name":"one"},null,{"name":"two"}]}"#,
        )
        .unwrap();
        assert_eq!(connection.total_count, 3);
        assert_eq!(connection.nodes.len(), 2);
    }

    #[test]
    fn cache_fingerprint_includes_query_and_variables() {
        let first = request_fingerprint("query", &json!({"first": 5, "after": null}));
        let different_page = request_fingerprint("query", &json!({"first": 5, "after": "cursor"}));
        let different_shape = request_fingerprint("lite query", &json!({"first": 5}));
        assert_ne!(first, different_page);
        assert_ne!(first, different_shape);
    }
}
