use super::*;

#[derive(Default)]
pub(super) struct EngineeringGraph {
    pub(super) nodes: Vec<GraphNode>,
    pub(super) node_indexes: HashMap<String, usize>,
    node_ranks: Vec<u8>,
    pub(super) edges: Vec<GraphEdge>,
    edge_keys: HashSet<(String, &'static str, String)>,
}

impl EngineeringGraph {
    pub(super) fn upsert_node(&mut self, key: String, node: GraphNode, rank: u8) {
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

    pub(super) fn add_edge(
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

    pub(super) fn resolve_stubs(&self) {
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

    pub(super) fn embedding_texts(&self) -> impl Iterator<Item = &str> {
        self.nodes
            .iter()
            .flat_map(|node| node.embedding_texts.iter().map(String::as_str))
            .chain(self.edges.iter().map(|edge| edge.embedding_text.as_str()))
    }

    pub(super) fn embedding_text_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.embedding_texts.len())
            .sum::<usize>()
            + self.edges.len()
    }

    pub(super) fn node_label_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for node in &self.nodes {
            *counts.entry(node.label).or_default() += 1;
        }
        counts
    }

    pub(super) fn edge_label_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for edge in &self.edges {
            *counts.entry(edge.label).or_default() += 1;
        }
        counts
    }
}

pub(super) struct GraphNode {
    pub(super) label: &'static str,
    pub(super) properties: Vec<(&'static str, Value)>,
    pub(super) embedding_texts: Vec<String>,
    pub(super) summary: String,
}

impl GraphNode {
    pub(super) fn new(
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

pub(super) struct GraphEdge {
    pub(super) source: String,
    pub(super) target: String,
    pub(super) label: &'static str,
    pub(super) properties: Vec<(&'static str, Value)>,
    pub(super) embedding_text: String,
}

pub(super) fn repository_node(repository: &Repository) -> GraphNode {
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

pub(super) fn add_issue(
    graph: &mut EngineeringGraph,
    repository_key: &str,
    issue: &Issue,
    complete: bool,
) {
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

pub(super) fn add_pull_request(
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

pub(super) fn add_discussion(
    graph: &mut EngineeringGraph,
    repository_key: &str,
    discussion: &Discussion,
) {
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

pub(super) fn add_release(graph: &mut EngineeringGraph, repository_key: &str, release: &Release) {
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

pub(super) fn string_value(value: &str) -> Value {
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

pub(super) fn truncate_utf8(value: &str, max_bytes: usize) -> String {
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

pub(super) fn repository_key(id: &str) -> String {
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
