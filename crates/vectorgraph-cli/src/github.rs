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
use vectorgraph::{BulkLoader, DatabaseOptions, Similarity, Value, VectorEncoding};

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

#[derive(Default)]
struct EngineeringGraph {
    nodes: Vec<GraphNode>,
    node_indexes: HashMap<String, usize>,
    node_ranks: Vec<u8>,
    edges: Vec<GraphEdge>,
    edge_keys: HashSet<(String, &'static str, String)>,
}

impl EngineeringGraph {
    fn upsert_node(&mut self, key: String, node: GraphNode, rank: u8) {
        if let Some(&index) = self.node_indexes.get(&key) {
            if rank > self.node_ranks[index] {
                self.nodes[index] = node;
                self.node_ranks[index] = rank;
            }
            return;
        }
        self.node_indexes.insert(key, self.nodes.len());
        self.nodes.push(node);
        self.node_ranks.push(rank);
    }

    fn add_edge(
        &mut self,
        source: &str,
        target: &str,
        label: &'static str,
        properties: Vec<(&'static str, Value)>,
        embedding_text: String,
    ) {
        let key = (source.to_owned(), label, target.to_owned());
        if self.edge_keys.insert(key) {
            self.edges.push(GraphEdge {
                source: source.to_owned(),
                target: target.to_owned(),
                label,
                properties,
                embedding_text: embedding_payload(&embedding_text),
            });
        }
    }

    fn resolve_stubs(&self) {
        debug_assert!(self.edges.iter().all(|edge| {
            self.node_indexes.contains_key(&edge.source)
                && self.node_indexes.contains_key(&edge.target)
        }));
        let _summary_bytes = self
            .nodes
            .iter()
            .map(|node| node.summary.len())
            .sum::<usize>();
    }

    fn embedding_texts(&self) -> impl Iterator<Item = &str> {
        self.nodes
            .iter()
            .flat_map(|node| node.embedding_texts.iter().map(String::as_str))
            .chain(self.edges.iter().map(|edge| edge.embedding_text.as_str()))
    }

    fn embedding_text_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.embedding_texts.len())
            .sum::<usize>()
            + self.edges.len()
    }

    fn node_label_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for node in &self.nodes {
            *counts.entry(node.label).or_default() += 1;
        }
        counts
    }

    fn edge_label_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for edge in &self.edges {
            *counts.entry(edge.label).or_default() += 1;
        }
        counts
    }
}

struct GraphNode {
    label: &'static str,
    properties: Vec<(&'static str, Value)>,
    embedding_texts: Vec<String>,
    summary: String,
}

impl GraphNode {
    fn new(
        label: &'static str,
        properties: Vec<(&'static str, Value)>,
        embedding_texts: Vec<String>,
        summary: String,
    ) -> Self {
        Self {
            label,
            properties,
            embedding_texts: embedding_texts
                .into_iter()
                .filter(|text| !text.trim().is_empty())
                .map(|text| embedding_payload(&text))
                .collect(),
            summary,
        }
    }
}

struct GraphEdge {
    source: String,
    target: String,
    label: &'static str,
    properties: Vec<(&'static str, Value)>,
    embedding_text: String,
}

fn repository_node(repository: &Repository) -> GraphNode {
    let mut properties = vec![
        ("external_id", string_value(&repository.id)),
        ("name", string_value(&repository.name)),
        ("name_with_owner", string_value(&repository.name_with_owner)),
        ("url", string_value(&repository.url)),
        ("stars", Value::Int(repository.stargazer_count)),
        ("forks", Value::Int(repository.fork_count)),
        ("archived", Value::Bool(repository.is_archived)),
        ("fork", Value::Bool(repository.is_fork)),
        (
            "issue_total",
            Value::Int(repository.issues.total_count as i64),
        ),
        (
            "pull_request_total",
            Value::Int(repository.pull_requests.total_count as i64),
        ),
        (
            "discussion_total",
            Value::Int(repository.discussions.total_count as i64),
        ),
        (
            "release_total",
            Value::Int(repository.releases.total_count as i64),
        ),
    ];
    push_optional_string(
        &mut properties,
        "description",
        repository.description.as_deref(),
    );
    push_optional_string(
        &mut properties,
        "primary_language",
        repository
            .primary_language
            .as_ref()
            .map(|value| value.name.as_str()),
    );
    push_datetime(
        &mut properties,
        "created_at",
        "created_at_ms",
        Some(&repository.created_at),
    );
    push_datetime(
        &mut properties,
        "updated_at",
        "updated_at_ms",
        Some(&repository.updated_at),
    );
    let text = format!(
        "GitHub repository {}. {} Primary language {}.",
        repository.name_with_owner,
        repository.description.as_deref().unwrap_or(""),
        repository
            .primary_language
            .as_ref()
            .map_or("unknown", |language| language.name.as_str())
    );
    GraphNode::new(
        "Repository",
        properties,
        vec![text],
        repository.name_with_owner.clone(),
    )
}

fn add_issue(graph: &mut EngineeringGraph, repository_key: &str, issue: &Issue, complete: bool) {
    let key = issue_key(&issue.id);
    let mut properties = content_properties(ContentFields {
        id: &issue.id,
        number: issue.number,
        title: &issue.title,
        body: &issue.body_text,
        url: &issue.url,
        state: &issue.state,
        created_at: &issue.created_at,
        updated_at: &issue.updated_at,
        closed_at: issue.closed_at.as_deref(),
    });
    properties.push((
        "comment_total",
        Value::Int(issue.comments.total_count as i64),
    ));
    properties.push(("detail_complete", Value::Bool(complete)));
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "Issue",
            properties,
            title_body_embeddings("GitHub issue", &issue.title, &issue.body_text),
            format!("issue #{} {}", issue.number, issue.title),
        ),
        if complete { 3 } else { 1 },
    );
    graph.add_edge(
        repository_key,
        &key,
        "HAS_ISSUE",
        Vec::new(),
        format!(
            "Repository contains GitHub issue #{}: {}",
            issue.number, issue.title
        ),
    );
    add_author(
        graph,
        issue.author.as_ref(),
        &key,
        &format!("issue #{}", issue.number),
    );
    add_taxonomy(
        graph,
        repository_key,
        &key,
        &issue.labels.nodes,
        issue.milestone.as_ref(),
    );
    add_assignees(
        graph,
        &key,
        &issue.assignees.nodes,
        &format!("issue #{}", issue.number),
    );
    for comment in &issue.comments.nodes {
        add_comment(graph, &key, comment, "IssueComment", "COMMENTS_ON");
    }
    for pull_request in &issue.closed_by_pull_requests_references.nodes {
        add_pull_request(graph, repository_key, pull_request, false);
        let pull_key = pull_request_key(&pull_request.id);
        graph.add_edge(
            &pull_key,
            &key,
            "CLOSES",
            Vec::new(),
            format!(
                "Pull request #{} closes issue #{}: {}",
                pull_request.number, issue.number, issue.title
            ),
        );
    }
}

fn add_pull_request(
    graph: &mut EngineeringGraph,
    repository_key: &str,
    pull_request: &PullRequest,
    complete: bool,
) {
    let key = pull_request_key(&pull_request.id);
    let mut properties = content_properties(ContentFields {
        id: &pull_request.id,
        number: pull_request.number,
        title: &pull_request.title,
        body: &pull_request.body_text,
        url: &pull_request.url,
        state: &pull_request.state,
        created_at: &pull_request.created_at,
        updated_at: &pull_request.updated_at,
        closed_at: pull_request.closed_at.as_deref(),
    });
    properties.extend([
        ("detail_complete", Value::Bool(complete)),
        ("draft", Value::Bool(pull_request.is_draft)),
        ("merged", Value::Bool(pull_request.merged)),
        ("additions", Value::Int(pull_request.additions)),
        ("deletions", Value::Int(pull_request.deletions)),
        ("changed_files", Value::Int(pull_request.changed_files)),
        ("base_ref", string_value(&pull_request.base_ref_name)),
        ("head_ref", string_value(&pull_request.head_ref_name)),
        (
            "comment_total",
            Value::Int(pull_request.comments.total_count as i64),
        ),
        (
            "review_total",
            Value::Int(pull_request.reviews.total_count as i64),
        ),
        (
            "commit_total",
            Value::Int(pull_request.commits.total_count as i64),
        ),
        (
            "file_total",
            Value::Int(pull_request.files.total_count as i64),
        ),
    ]);
    push_datetime(
        &mut properties,
        "merged_at",
        "merged_at_ms",
        pull_request.merged_at.as_deref(),
    );
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "PullRequest",
            properties,
            title_body_embeddings(
                "GitHub pull request",
                &pull_request.title,
                &pull_request.body_text,
            ),
            format!(
                "pull request #{} {}",
                pull_request.number, pull_request.title
            ),
        ),
        if complete { 3 } else { 1 },
    );
    graph.add_edge(
        repository_key,
        &key,
        "HAS_PULL_REQUEST",
        Vec::new(),
        format!(
            "Repository contains GitHub pull request #{}: {}",
            pull_request.number, pull_request.title
        ),
    );
    add_author(
        graph,
        pull_request.author.as_ref(),
        &key,
        &format!("pull request #{}", pull_request.number),
    );
    add_taxonomy(
        graph,
        repository_key,
        &key,
        &pull_request.labels.nodes,
        pull_request.milestone.as_ref(),
    );
    add_assignees(
        graph,
        &key,
        &pull_request.assignees.nodes,
        &format!("pull request #{}", pull_request.number),
    );
    for comment in &pull_request.comments.nodes {
        add_comment(graph, &key, comment, "PullRequestComment", "COMMENTS_ON");
    }
    for review in &pull_request.reviews.nodes {
        add_review(graph, &key, pull_request.number, review);
    }
    for commit_node in &pull_request.commits.nodes {
        add_commit(graph, &key, pull_request.number, &commit_node.commit, true);
    }
    for file in &pull_request.files.nodes {
        add_file(graph, repository_key, &key, pull_request.number, file);
    }
    for issue in &pull_request.closing_issues_references.nodes {
        add_issue(graph, repository_key, issue, false);
        graph.add_edge(
            &key,
            &issue_key(&issue.id),
            "CLOSES",
            Vec::new(),
            format!(
                "Pull request #{} closes issue #{}: {}",
                pull_request.number, issue.number, issue.title
            ),
        );
    }
}

fn add_discussion(graph: &mut EngineeringGraph, repository_key: &str, discussion: &Discussion) {
    let key = discussion_key(&discussion.id);
    let mut properties = vec![
        ("external_id", string_value(&discussion.id)),
        ("number", Value::Int(discussion.number)),
        ("title", string_value(&discussion.title)),
        ("body", string_value(&stored_text(&discussion.body_text))),
        ("url", string_value(&discussion.url)),
        ("closed", Value::Bool(discussion.closed)),
        ("locked", Value::Bool(discussion.locked)),
        (
            "comment_total",
            Value::Int(discussion.comments.total_count as i64),
        ),
    ];
    push_datetime(
        &mut properties,
        "created_at",
        "created_at_ms",
        Some(&discussion.created_at),
    );
    push_datetime(
        &mut properties,
        "updated_at",
        "updated_at_ms",
        Some(&discussion.updated_at),
    );
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "Discussion",
            properties,
            title_body_embeddings(
                "GitHub discussion",
                &discussion.title,
                &discussion.body_text,
            ),
            format!("discussion #{} {}", discussion.number, discussion.title),
        ),
        3,
    );
    graph.add_edge(
        repository_key,
        &key,
        "HAS_DISCUSSION",
        Vec::new(),
        format!(
            "Repository contains GitHub discussion #{}: {}",
            discussion.number, discussion.title
        ),
    );
    add_author(
        graph,
        discussion.author.as_ref(),
        &key,
        &format!("discussion #{}", discussion.number),
    );
    let category_key = format!("discussion-category:{}", discussion.category.id);
    graph.upsert_node(
        category_key.clone(),
        GraphNode::new(
            "DiscussionCategory",
            vec![
                ("external_id", string_value(&discussion.category.id)),
                ("name", string_value(&discussion.category.name)),
                ("emoji", string_value(&discussion.category.emoji)),
                (
                    "description",
                    string_value(discussion.category.description.as_deref().unwrap_or("")),
                ),
            ],
            vec![format!(
                "GitHub discussion category {}. {}",
                discussion.category.name,
                discussion.category.description.as_deref().unwrap_or("")
            )],
            discussion.category.name.clone(),
        ),
        2,
    );
    graph.add_edge(
        repository_key,
        &category_key,
        "HAS_DISCUSSION_CATEGORY",
        Vec::new(),
        format!(
            "Repository has GitHub discussion category {}",
            discussion.category.name
        ),
    );
    graph.add_edge(
        &key,
        &category_key,
        "IN_CATEGORY",
        Vec::new(),
        format!(
            "Discussion #{} is in category {}",
            discussion.number, discussion.category.name
        ),
    );
    let answer_id = discussion.answer.as_ref().map(|answer| answer.id.as_str());
    for comment in &discussion.comments.nodes {
        add_comment(
            graph,
            &key,
            comment,
            "DiscussionComment",
            if answer_id == Some(comment.id.as_str()) {
                "ANSWERS"
            } else {
                "COMMENTS_ON"
            },
        );
        let comment_key = comment_key(&comment.id);
        for reply in &comment.replies.nodes {
            add_comment(graph, &comment_key, reply, "DiscussionReply", "REPLIES_TO");
        }
    }
}

fn add_release(graph: &mut EngineeringGraph, repository_key: &str, release: &Release) {
    let key = release_key(&release.id);
    let display_name = release
        .name
        .as_deref()
        .filter(|name| !name.is_empty())
        .unwrap_or(&release.tag_name);
    let mut properties = vec![
        ("external_id", string_value(&release.id)),
        ("name", string_value(display_name)),
        ("tag", string_value(&release.tag_name)),
        (
            "description",
            string_value(&stored_text(release.description.as_deref().unwrap_or(""))),
        ),
        ("url", string_value(&release.url)),
        ("draft", Value::Bool(release.is_draft)),
        ("prerelease", Value::Bool(release.is_prerelease)),
    ];
    push_datetime(
        &mut properties,
        "created_at",
        "created_at_ms",
        Some(&release.created_at),
    );
    push_datetime(
        &mut properties,
        "published_at",
        "published_at_ms",
        release.published_at.as_deref(),
    );
    push_datetime(
        &mut properties,
        "updated_at",
        "updated_at_ms",
        Some(&release.updated_at),
    );
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "Release",
            properties,
            title_body_embeddings(
                "GitHub release",
                display_name,
                release.description.as_deref().unwrap_or(""),
            ),
            format!("release {display_name}"),
        ),
        3,
    );
    graph.add_edge(
        repository_key,
        &key,
        "PUBLISHED_RELEASE",
        Vec::new(),
        format!("Repository published GitHub release {display_name}"),
    );
    add_author(
        graph,
        release.author.as_ref(),
        &key,
        &format!("release {display_name}"),
    );
    if let Some(commit) = &release.tag_commit {
        let commit_key = commit_key(&commit.oid);
        graph.upsert_node(
            commit_key.clone(),
            GraphNode::new(
                "Commit",
                vec![
                    ("oid", string_value(&commit.oid)),
                    ("url", string_value(&commit.url)),
                    ("stub", Value::Bool(true)),
                ],
                vec![format!("Git commit {}", commit.oid)],
                format!("commit {}", short_oid(&commit.oid)),
            ),
            1,
        );
        graph.add_edge(
            &key,
            &commit_key,
            "POINTS_TO",
            Vec::new(),
            format!("Release {display_name} points to Git commit {}", commit.oid),
        );
    }
}

fn add_author(
    graph: &mut EngineeringGraph,
    actor: Option<&Actor>,
    target_key: &str,
    target_summary: &str,
) {
    let Some(actor) = actor else {
        return;
    };
    let actor_key = ensure_actor(graph, actor);
    graph.add_edge(
        &actor_key,
        target_key,
        "AUTHORED",
        Vec::new(),
        format!("GitHub user {} authored {target_summary}", actor.login),
    );
}

fn add_assignees(
    graph: &mut EngineeringGraph,
    target_key: &str,
    assignees: &[Actor],
    target_summary: &str,
) {
    for actor in assignees {
        let actor_key = ensure_actor(graph, actor);
        graph.add_edge(
            target_key,
            &actor_key,
            "ASSIGNED_TO",
            Vec::new(),
            format!(
                "{target_summary} is assigned to GitHub user {}",
                actor.login
            ),
        );
    }
}

fn add_taxonomy(
    graph: &mut EngineeringGraph,
    repository_key: &str,
    target_key: &str,
    labels: &[Label],
    milestone: Option<&Milestone>,
) {
    for label in labels {
        let label_key = format!("label:{}", label.id);
        graph.upsert_node(
            label_key.clone(),
            GraphNode::new(
                "Label",
                vec![
                    ("external_id", string_value(&label.id)),
                    ("name", string_value(&label.name)),
                    ("color", string_value(&label.color)),
                    ("url", string_value(&label.url)),
                    (
                        "description",
                        string_value(label.description.as_deref().unwrap_or("")),
                    ),
                ],
                vec![format!(
                    "GitHub label {}. {}",
                    label.name,
                    label.description.as_deref().unwrap_or("")
                )],
                label.name.clone(),
            ),
            2,
        );
        graph.add_edge(
            target_key,
            &label_key,
            "TAGGED",
            Vec::new(),
            format!("GitHub work item has label {}", label.name),
        );
    }
    if let Some(milestone) = milestone {
        let milestone_key = format!("milestone:{}", milestone.id);
        let mut properties = vec![
            ("external_id", string_value(&milestone.id)),
            ("number", Value::Int(milestone.number)),
            ("title", string_value(&milestone.title)),
            ("state", string_value(&milestone.state)),
            ("url", string_value(&milestone.url)),
            (
                "description",
                string_value(milestone.description.as_deref().unwrap_or("")),
            ),
        ];
        push_datetime(
            &mut properties,
            "created_at",
            "created_at_ms",
            Some(&milestone.created_at),
        );
        push_datetime(
            &mut properties,
            "updated_at",
            "updated_at_ms",
            Some(&milestone.updated_at),
        );
        push_datetime(
            &mut properties,
            "due_at",
            "due_at_ms",
            milestone.due_on.as_deref(),
        );
        graph.upsert_node(
            milestone_key.clone(),
            GraphNode::new(
                "Milestone",
                properties,
                title_body_embeddings(
                    "GitHub milestone",
                    &milestone.title,
                    milestone.description.as_deref().unwrap_or(""),
                ),
                milestone.title.clone(),
            ),
            2,
        );
        graph.add_edge(
            repository_key,
            &milestone_key,
            "HAS_MILESTONE",
            Vec::new(),
            format!("Repository has milestone {}", milestone.title),
        );
        graph.add_edge(
            target_key,
            &milestone_key,
            "IN_MILESTONE",
            Vec::new(),
            format!("GitHub work item is in milestone {}", milestone.title),
        );
    }
}

fn add_comment(
    graph: &mut EngineeringGraph,
    parent_key: &str,
    comment: &Comment,
    kind: &'static str,
    relationship: &'static str,
) {
    let key = comment_key(&comment.id);
    let mut properties = vec![
        ("external_id", string_value(&comment.id)),
        ("body", string_value(&stored_text(&comment.body_text))),
        ("url", string_value(&comment.url)),
        ("kind", string_value(kind)),
    ];
    push_datetime(
        &mut properties,
        "created_at",
        "created_at_ms",
        Some(&comment.created_at),
    );
    push_datetime(
        &mut properties,
        "updated_at",
        "updated_at_ms",
        Some(&comment.updated_at),
    );
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "Comment",
            properties,
            vec![format!("GitHub {kind}: {}", comment.body_text)],
            text_summary(&comment.body_text),
        ),
        3,
    );
    graph.add_edge(
        &key,
        parent_key,
        relationship,
        datetime_properties("created_at", "created_at_ms", Some(&comment.created_at)),
        format!(
            "GitHub {kind} {relationship}: {}",
            text_summary(&comment.body_text)
        ),
    );
    add_author(graph, comment.author.as_ref(), &key, kind);
}

fn add_review(
    graph: &mut EngineeringGraph,
    pull_request_key: &str,
    pull_request_number: i64,
    review: &Review,
) {
    let key = format!("review:{}", review.id);
    let mut properties = vec![
        ("external_id", string_value(&review.id)),
        ("body", string_value(&stored_text(&review.body_text))),
        ("url", string_value(&review.url)),
        ("state", string_value(&review.state)),
    ];
    push_datetime(
        &mut properties,
        "created_at",
        "created_at_ms",
        Some(&review.created_at),
    );
    push_datetime(
        &mut properties,
        "updated_at",
        "updated_at_ms",
        Some(&review.updated_at),
    );
    push_datetime(
        &mut properties,
        "submitted_at",
        "submitted_at_ms",
        review.submitted_at.as_deref(),
    );
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "Review",
            properties,
            vec![format!(
                "GitHub pull request review with state {}. {}",
                review.state, review.body_text
            )],
            format!("{} review", review.state),
        ),
        3,
    );
    graph.add_edge(
        &key,
        pull_request_key,
        "REVIEWS",
        vec![("state", string_value(&review.state))],
        format!(
            "{} review of pull request #{}: {}",
            review.state,
            pull_request_number,
            text_summary(&review.body_text)
        ),
    );
    add_author(graph, review.author.as_ref(), &key, "pull request review");
}

fn add_commit(
    graph: &mut EngineeringGraph,
    pull_request_key: &str,
    pull_request_number: i64,
    commit: &Commit,
    complete: bool,
) {
    let key = commit_key(&commit.oid);
    let mut properties = vec![
        ("external_id", string_value(&commit.id)),
        ("oid", string_value(&commit.oid)),
        ("headline", string_value(&commit.message_headline)),
        ("body", string_value(&stored_text(&commit.message_body))),
        ("url", string_value(&commit.url)),
    ];
    if let Some(author_name) = commit
        .author
        .as_ref()
        .and_then(|author| author.name.as_deref())
    {
        properties.push(("author_name", string_value(author_name)));
    }
    push_datetime(
        &mut properties,
        "authored_at",
        "authored_at_ms",
        Some(&commit.authored_date),
    );
    push_datetime(
        &mut properties,
        "committed_at",
        "committed_at_ms",
        Some(&commit.committed_date),
    );
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "Commit",
            properties,
            title_body_embeddings("Git commit", &commit.message_headline, &commit.message_body),
            format!(
                "commit {} {}",
                short_oid(&commit.oid),
                commit.message_headline
            ),
        ),
        if complete { 3 } else { 1 },
    );
    graph.add_edge(
        pull_request_key,
        &key,
        "HAS_COMMIT",
        Vec::new(),
        format!(
            "Pull request #{} includes commit {}: {}",
            pull_request_number,
            short_oid(&commit.oid),
            commit.message_headline
        ),
    );
    if let Some(actor) = commit
        .author
        .as_ref()
        .and_then(|author| author.user.as_ref())
    {
        add_author(
            graph,
            Some(actor),
            &key,
            &format!("commit {}", short_oid(&commit.oid)),
        );
    }
}

fn add_file(
    graph: &mut EngineeringGraph,
    repository_key: &str,
    pull_request_key: &str,
    pull_request_number: i64,
    file: &ChangedFile,
) {
    let key = format!("file:{}", file.path);
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "File",
            vec![("path", string_value(&file.path))],
            vec![format!("Repository file path {}", file.path)],
            file.path.clone(),
        ),
        2,
    );
    graph.add_edge(
        repository_key,
        &key,
        "CONTAINS_FILE",
        Vec::new(),
        format!("Repository contains file {}", file.path),
    );
    graph.add_edge(
        pull_request_key,
        &key,
        "CHANGES",
        vec![
            ("additions", Value::Int(file.additions)),
            ("deletions", Value::Int(file.deletions)),
            ("change_type", string_value(&file.change_type)),
        ],
        format!(
            "Pull request #{} {} file {} with {} additions and {} deletions",
            pull_request_number,
            file.change_type.to_lowercase(),
            file.path,
            file.additions,
            file.deletions
        ),
    );
}

fn ensure_actor(graph: &mut EngineeringGraph, actor: &Actor) -> String {
    let key = format!("user:{}", actor.login.to_lowercase());
    graph.upsert_node(
        key.clone(),
        GraphNode::new(
            "User",
            vec![
                ("login", string_value(&actor.login)),
                ("url", string_value(&actor.url)),
                ("avatar_url", string_value(&actor.avatar_url)),
                ("bot", Value::Bool(actor.login.ends_with("[bot]"))),
            ],
            vec![format!("GitHub user {}", actor.login)],
            actor.login.clone(),
        ),
        2,
    );
    key
}

struct ContentFields<'a> {
    id: &'a str,
    number: i64,
    title: &'a str,
    body: &'a str,
    url: &'a str,
    state: &'a str,
    created_at: &'a str,
    updated_at: &'a str,
    closed_at: Option<&'a str>,
}

fn content_properties(content: ContentFields<'_>) -> Vec<(&'static str, Value)> {
    let mut properties = vec![
        ("external_id", string_value(content.id)),
        ("number", Value::Int(content.number)),
        ("title", string_value(content.title)),
        ("body", string_value(&stored_text(content.body))),
        ("url", string_value(content.url)),
        ("state", string_value(content.state)),
    ];
    push_datetime(
        &mut properties,
        "created_at",
        "created_at_ms",
        Some(content.created_at),
    );
    push_datetime(
        &mut properties,
        "updated_at",
        "updated_at_ms",
        Some(content.updated_at),
    );
    push_datetime(
        &mut properties,
        "closed_at",
        "closed_at_ms",
        content.closed_at,
    );
    properties
}

fn title_body_embeddings(kind: &str, title: &str, body: &str) -> Vec<String> {
    let mut embeddings = vec![format!("{kind}: {title}")];
    if !body.trim().is_empty() {
        embeddings.push(format!("{kind} {title}. {body}"));
    }
    embeddings
}

fn push_optional_string(
    properties: &mut Vec<(&'static str, Value)>,
    key: &'static str,
    value: Option<&str>,
) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        properties.push((key, string_value(value)));
    }
}

fn push_datetime(
    properties: &mut Vec<(&'static str, Value)>,
    text_key: &'static str,
    millis_key: &'static str,
    value: Option<&str>,
) {
    let Some(value) = value else {
        return;
    };
    properties.push((text_key, string_value(value)));
    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        properties.push((millis_key, Value::Int(parsed.timestamp_millis())));
    }
}

fn datetime_properties(
    text_key: &'static str,
    millis_key: &'static str,
    value: Option<&str>,
) -> Vec<(&'static str, Value)> {
    let mut properties = Vec::new();
    push_datetime(&mut properties, text_key, millis_key, value);
    properties
}

fn string_value(value: &str) -> Value {
    Value::String(Arc::from(value))
}

fn stored_text(value: &str) -> String {
    truncate_utf8(value.trim(), 32 * 1024)
}

fn embedding_payload(value: &str) -> String {
    truncate_utf8(value.trim(), 12 * 1024)
}

fn text_summary(value: &str) -> String {
    truncate_utf8(
        value
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim(),
        160,
    )
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

fn short_oid(oid: &str) -> &str {
    &oid[..oid.len().min(8)]
}

fn repository_key(id: &str) -> String {
    format!("repository:{id}")
}

fn issue_key(id: &str) -> String {
    format!("issue:{id}")
}

fn pull_request_key(id: &str) -> String {
    format!("pull-request:{id}")
}

fn discussion_key(id: &str) -> String {
    format!("discussion:{id}")
}

fn release_key(id: &str) -> String {
    format!("release:{id}")
}

fn comment_key(id: &str) -> String {
    format!("comment:{id}")
}

fn commit_key(oid: &str) -> String {
    format!("commit:{oid}")
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
    let owner = parts.next().filter(|value| !value.is_empty());
    let name = parts.next().filter(|value| !value.is_empty());
    if owner.is_none() || name.is_none() || parts.next().is_some() {
        return Err(format!("expected GitHub repository as owner/name, got {value:?}").into());
    }
    Ok((owner.unwrap().to_owned(), name.unwrap().to_owned()))
}

fn cache_path(database_path: &Path) -> PathBuf {
    if let Some(path) = env::var_os("VG_GITHUB_CACHE") {
        return PathBuf::from(path);
    }
    let file_name = database_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("vectorgraph")
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
                    .header("User-Agent", "VectorGraph-GitHub-Importer")
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
