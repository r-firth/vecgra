# GitHub engineering-history import

`vg import-github` turns a GitHub repository's collaboration history into a
portable VectorGraph database. It is deliberately not a code AST: the graph
captures work, decisions, evidence, authorship, review, discussion, and the
files and commits those activities affected.

```sh
# Uses GITHUB_TOKEN, GH_TOKEN, or the token from `gh auth login`.
target/release/vg import-github OWNER/REPOSITORY graph.vg \
  [issues] [pulls] [discussions] [releases] \
  [dimension] [hash|qwen] [batch-size]
```

The limits default to 1,000 issues, 1,000 pull requests, 300 discussions, and
100 releases. Dimension defaults to 256, the deterministic `hash` embedder is
the offline default, and the embedding batch defaults to 128. `qwen` uses
Qwen3-Embedding-8B through OpenRouter and requires `OPENROUTER_API_KEY`.
The destination must not already exist.

## Graph shape

| Nodes | Important relationships |
|---|---|
| `Repository`, `Topic` | `HAS_TOPIC` |
| `Issue`, `PullRequest` | `HAS_ISSUE`, `HAS_PULL_REQUEST`, `CLOSES` |
| `Discussion`, `DiscussionCategory` | `HAS_DISCUSSION`, `IN_CATEGORY`, `ANSWERS` |
| `User`, `Comment`, `Review` | `AUTHORED`, `ASSIGNED_TO`, `COMMENTS_ON`, `REPLIES_TO`, `REVIEWS` |
| `Commit`, `File`, `Release` | `HAS_COMMIT`, `CHANGES`, `CONTAINS_FILE`, `PUBLISHED_RELEASE`, `POINTS_TO` |
| `Label`, `Milestone` | `TAGGED`, `HAS_MILESTONE`, `IN_MILESTONE` |

Content nodes normally receive separate title/context and body vectors. Every
relationship receives a natural-language embedding payload describing its
typed assertion. This makes relationship search and path scoring first-class:
a query can match the meaning of “this PR closes that failure” independently
of either endpoint's prose.

Properties remain ordinary typed graph data. Content records retain IDs,
numbers, state, URLs, text, ISO dates, numeric millisecond timestamps, and
available counts/diff statistics. Text stored as a property is bounded to 32
KiB and text sent to an embedder is bounded to 12 KiB.

## Completeness and recovery

Top-level limits select the most recently updated records. Nested connections
are intentionally bounded to keep GitHub GraphQL responses reliable: issue and
PR comments and reviews are capped at 20, PR commits at 40, changed files at
50, discussion comments at 40, and discussion replies at 20. The corresponding
GitHub `totalCount` is retained, so a consumer can see when a connection has
more records than this initial snapshot includes.

GitHub can return null tombstones for deleted or inaccessible connection
members. The importer drops only those null children. Rich PR pages are
retried on rate-limit/server failures; if a GitHub resolver rejects an
optional derived field, a lite request preserves the primary PR, author,
counts, and pagination cursor. `Issue` and `PullRequest` nodes expose
`detail_complete`: `false` identifies a cross-reference stub or lite fallback
that a later sync may enrich.

Successful GraphQL pages are cached beside the destination under
`<database>.github-cache`. Cache filenames fingerprint the query and all
variables, preventing a page fetched with one cursor/limit from being reused
for another. Set `VG_GITHUB_CACHE` to choose a reusable cache directory. Tokens
are used only for requests and are never written into the cache or database.

## Queries

```sh
target/release/vg query graph.vg \
  'MATCH (p:PullRequest)-[r:CLOSES]->(i:Issue) RETURN p,r,i LIMIT 20'

target/release/vg query graph.vg \
  'MATCH (r:Review)-[v:REVIEWS]->(p:PullRequest) RETURN r,v,p LIMIT 20'

target/release/vg query-text graph.vg \
  'MATCH (p:PullRequest)-[c:CHANGES]->(f:File) RETURN p,c,f LIMIT 20' \
  'language server performance diagnostics' qwen

target/release/vg semantic-text graph.vg \
  'regressions fixed after reviewer feedback' 20 3 qwen
```

The hash embedder is useful for deterministic plumbing, performance tests, and
offline demos; use a real model when judging semantic quality.
