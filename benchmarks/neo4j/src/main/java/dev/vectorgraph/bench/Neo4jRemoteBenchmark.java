package dev.vectorgraph.bench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Config;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Logging;
import org.neo4j.driver.QueryRunner;
import org.neo4j.driver.Session;
import org.neo4j.driver.SessionConfig;
import org.neo4j.driver.Transaction;
import org.neo4j.driver.Values;

/** Bolt/Enterprise baseline using native VECTOR&lt;FLOAT32&gt; properties. */
public final class Neo4jRemoteBenchmark {
    private static final String VECTOR_INDEX = "vectors";
    private static final String RATING_PROPERTY = "avg_rating";
    private static final ObjectMapper JSON = new ObjectMapper();

    private Neo4jRemoteBenchmark() {}

    public static void main(String[] arguments) throws Exception {
        if (arguments.length == 0) {
            usage();
        }
        switch (arguments[0]) {
            case "probe" -> probe(arguments);
            case "smoke" -> smoke(arguments);
            case "import-fbin" -> importFbin(arguments);
            case "reindex" -> reindex(arguments);
            case "bench-fbin" -> benchmarkFbin(arguments);
            case "bench-gds-bfs" -> benchmarkGdsBfs(arguments);
            default -> usage();
        }
    }

    private static void probe(String[] arguments) {
        requireLength(arguments, 2, "probe <database>");
        try (Driver driver = openDriver();
                Session session = openSession(driver, arguments[1])) {
            var component = session.run(
                            "CALL dbms.components() YIELD name, versions, edition "
                                    + "WHERE name = 'Neo4j Kernel' RETURN versions[0] AS version, edition")
                    .single();
            System.out.printf("server_version\t%s%n", component.get("version").asString());
            System.out.printf("edition\t%s%n", component.get("edition").asString());
            System.out.printf("database\t%s%n", arguments[1]);
        }
    }

    private static void smoke(String[] arguments) {
        requireLength(arguments, 2, "smoke <database>");
        try (Driver driver = openDriver();
                Session session = openSession(driver, arguments[1])) {
            session.run("DROP INDEX " + VECTOR_INDEX + " IF EXISTS").consume();
            session.run("MATCH (n:Vector) DETACH DELETE n").consume();
            session.executeWrite(transaction -> {
                List<Map<String, Object>> rows = List.of(
                        row(0, new float[] {1.0f, 0.0f, 0.1f}, 8.0),
                        row(1, new float[] {0.0f, 1.0f, 0.1f}, 8.5),
                        row(2, new float[] {0.0f, 0.0f, 1.0f}, 9.0),
                        row(3, new float[] {1.0f, 1.0f, 0.1f}, 9.5));
                transaction.run(
                                "UNWIND $rows AS row CREATE (:Vector {external_id: row.id, "
                                        + "embedding: row.embedding, avg_rating: row.rating})",
                                Map.of("rows", rows))
                        .consume();
                return null;
            });
            createVectorIndex(session, 3, true, "none", 6.0, 16, 100);
            var type = session.run(
                            "MATCH (n:Vector) RETURN valueType(n.embedding) AS type LIMIT 1")
                    .single()
                    .get("type")
                    .asString();
            List<Hit> hits = search(
                    session,
                    new float[] {1.0f, 0.0f, 0.1f},
                    2,
                    8.0,
                    3);
            System.out.printf("property_type\t%s%n", type);
            for (Hit hit : hits) {
                System.out.printf(Locale.ROOT, "id\t%d\tscore\t%.6f%n", hit.id(), hit.score());
            }
            session.run("DROP INDEX " + VECTOR_INDEX + " IF EXISTS").consume();
            session.run("MATCH (n:Vector) DETACH DELETE n").consume();
        }
    }

    private static void importFbin(String[] arguments) throws Exception {
        requireLength(
                arguments,
                10,
                "import-fbin <database> <train.fbin> <metadata.jsonl|-> <batch-size> "
                        + "<none|scalar|binary> <expansion> <hnsw-m> <ef-construction>");
        String database = arguments[1];
        Path vectorsPath = Path.of(arguments[2]);
        Path metadataPath = arguments[3].equals("-") ? null : Path.of(arguments[3]);
        int batchSize = positiveInt(arguments[4], "batch size");
        String quantization = parseQuantization(arguments[5]);
        double expansion = positiveDouble(arguments[6], "expansion");
        int hnswM = positiveInt(arguments[7], "HNSW M");
        int efConstruction = positiveInt(arguments[8], "ef-construction");
        if (!arguments[9].equals("native-vector")) {
            throw new IllegalArgumentException("final argument must be native-vector");
        }

        MatrixHeader header = readMatrixHeader(vectorsPath, Float.BYTES);
        long dataStarted = System.nanoTime();
        try (Driver driver = openDriver();
                Session session = openSession(driver, database);
                FileChannel channel = FileChannel.open(vectorsPath, StandardOpenOption.READ);
                BufferedReader metadata = metadataPath == null
                        ? null
                        : Files.newBufferedReader(metadataPath, StandardCharsets.UTF_8)) {
            long existing = session.run("MATCH (n) RETURN count(n) AS count")
                    .single()
                    .get("count")
                    .asLong();
            if (existing != 0) {
                throw new IllegalArgumentException("database is not empty: " + database);
            }
            channel.position(8);
            ByteBuffer encoded = ByteBuffer.allocateDirect(header.columns() * Float.BYTES)
                    .order(ByteOrder.LITTLE_ENDIAN);
            int row = 0;
            while (row < header.rows()) {
                int end = Math.min(header.rows(), row + batchSize);
                List<Map<String, Object>> batch = new ArrayList<>(end - row);
                for (; row < end; row++) {
                    float[] vector = readVector(channel, encoded, header.columns());
                    Double rating = null;
                    if (metadata != null) {
                        String line = metadata.readLine();
                        if (line == null) {
                            throw new IOException("metadata ended before vector row " + row);
                        }
                        JsonNode value = JSON.readTree(line).path("properties").path(RATING_PROPERTY);
                        if (value.isNumber()) {
                            rating = value.doubleValue();
                        }
                    }
                    batch.add(row(row, vector, rating));
                }
                boolean filtered = metadata != null;
                session.executeWrite(transaction -> {
                    String ratingSet = filtered ? ", " + RATING_PROPERTY + ": row.rating" : "";
                    transaction.run(
                                    "UNWIND $rows AS row CREATE (:Vector {external_id: row.id, "
                                            + "embedding: row.embedding" + ratingSet + "})",
                                    Map.of("rows", batch))
                            .consume();
                    return null;
                });
                if (row % 100_000 == 0 || row == header.rows()) {
                    System.err.printf("stored %d/%d vectors%n", row, header.rows());
                }
            }
            if (metadata != null && metadata.readLine() != null) {
                throw new IOException("metadata has more rows than vector matrix");
            }
            long dataNanos = System.nanoTime() - dataStarted;
            long indexStarted = System.nanoTime();
            createVectorIndex(
                    session,
                    header.columns(),
                    metadata != null,
                    quantization,
                    expansion,
                    hnswM,
                    efConstruction);
            long indexNanos = System.nanoTime() - indexStarted;
            System.out.printf("rows\t%d%n", header.rows());
            System.out.printf("dimension\t%d%n", header.columns());
            System.out.printf(Locale.ROOT, "data_import_s\t%.3f%n", seconds(dataNanos));
            System.out.printf(Locale.ROOT, "vector_index_s\t%.3f%n", seconds(indexNanos));
            System.out.printf("quantization\t%s%n", quantization);
            System.out.printf(Locale.ROOT, "expansion_factor\t%.3f%n", expansion);
            System.out.printf("hnsw_m\t%d%n", hnswM);
            System.out.printf("ef_construction\t%d%n", efConstruction);
        }
    }

    private static void reindex(String[] arguments) {
        requireLength(
                arguments,
                9,
                "reindex <database> <dimension> <filtered|unfiltered> "
                        + "<none|scalar|binary> <expansion> <hnsw-m> <ef-construction>");
        String database = arguments[1];
        int dimension = positiveInt(arguments[2], "dimension");
        boolean filtered = switch (arguments[3]) {
            case "filtered" -> true;
            case "unfiltered" -> false;
            default -> throw new IllegalArgumentException("expected filtered or unfiltered");
        };
        String quantization = parseQuantization(arguments[4]);
        double expansion = positiveDouble(arguments[5], "expansion");
        int hnswM = positiveInt(arguments[6], "HNSW M");
        int efConstruction = positiveInt(arguments[7], "ef-construction");
        if (!arguments[8].equals("native-vector")) {
            throw new IllegalArgumentException("final argument must be native-vector");
        }
        long started = System.nanoTime();
        try (Driver driver = openDriver();
                Session session = openSession(driver, database)) {
            session.run("DROP INDEX " + VECTOR_INDEX + " IF EXISTS").consume();
            createVectorIndex(
                    session,
                    dimension,
                    filtered,
                    quantization,
                    expansion,
                    hnswM,
                    efConstruction);
        }
        System.out.printf(Locale.ROOT, "reindex_s\t%.3f%n", seconds(System.nanoTime() - started));
    }

    private static void benchmarkFbin(String[] arguments) throws Exception {
        requireLength(
                arguments,
                10,
                "bench-fbin <database> <queries.fbin> <truth.ibin> <query-count> <k> "
                        + "<inclusive-lower|-> <warmups> <autocommit|single-tx> <dimension>");
        String database = arguments[1];
        Path queriesPath = Path.of(arguments[2]);
        Path truthPath = Path.of(arguments[3]);
        int requestedQueries = positiveInt(arguments[4], "query count");
        int k = positiveInt(arguments[5], "k");
        Double lower = arguments[6].equals("-") ? null : Double.parseDouble(arguments[6]);
        int warmups = positiveOrZeroInt(arguments[7], "warmups");
        String mode = arguments[8];
        if (!mode.equals("autocommit") && !mode.equals("single-tx")) {
            throw new IllegalArgumentException("mode must be autocommit or single-tx");
        }
        int dimension = positiveInt(arguments[9], "dimension");
        FloatMatrix queries = readFloatMatrix(queriesPath, requestedQueries);
        IntMatrix truth = readIntMatrix(truthPath, requestedQueries);
        if (queries.columns() != dimension || queries.rows() != truth.rows() || k > truth.columns()) {
            throw new IllegalArgumentException("query and truth shapes do not agree");
        }

        long driverStarted = System.nanoTime();
        try (Driver driver = openDriver()) {
            driver.verifyConnectivity();
            try (Session session = openSession(driver, database)) {
                long connectNanos = System.nanoTime() - driverStarted;
                long eligible = lower == null
                        ? session.run("MATCH (n:Vector) RETURN count(n) AS count")
                                .single()
                                .get("count")
                                .asLong()
                        : session.run(
                                        "MATCH (n:Vector) WHERE n." + RATING_PROPERTY
                                                + " >= $lower RETURN count(n) AS count",
                                        Map.of("lower", lower))
                                .single()
                                .get("count")
                                .asLong();
                BenchmarkResult result;
                if (mode.equals("single-tx")) {
                    try (Transaction transaction = session.beginTransaction()) {
                        result = runQueries(
                                transaction, queries, truth, k, lower, warmups, dimension);
                        transaction.commit();
                    }
                } else {
                    result = runQueries(session, queries, truth, k, lower, warmups, dimension);
                }
                System.out.printf("queries\t%d%n", queries.rows());
                System.out.printf("k\t%d%n", k);
                System.out.printf("mode\t%s%n", mode);
                System.out.printf("filter_lower\t%s%n", lower == null ? "none" : lower);
                System.out.printf("eligible_vectors\t%d%n", eligible);
                System.out.printf(Locale.ROOT, "recall_at_k\t%.6f%n", result.recall());
                System.out.printf("minimum_results\t%d%n", result.minimumResults());
                System.out.printf(Locale.ROOT, "connect_ms\t%.3f%n", millis(connectNanos));
                System.out.printf(Locale.ROOT, "query_p50_ms\t%.3f%n", millis(percentile(result.samples(), 0.50)));
                System.out.printf(Locale.ROOT, "query_p95_ms\t%.3f%n", millis(percentile(result.samples(), 0.95)));
                System.out.printf(Locale.ROOT, "query_max_ms\t%.3f%n", millis(result.samples()[result.samples().length - 1]));
            }
        }
    }

    private static void benchmarkGdsBfs(String[] arguments) {
        requireLength(
                arguments,
                7,
                "bench-gds-bfs <database> <graph-name> <source-node-id> "
                        + "<warmups> <runs> <concurrency>");
        String database = arguments[1];
        String graphName = arguments[2];
        long sourceNodeId = Long.parseLong(arguments[3]);
        int warmups = positiveOrZeroInt(arguments[4], "warmups");
        int runs = positiveInt(arguments[5], "runs");
        int concurrency = positiveInt(arguments[6], "concurrency");

        try (Driver driver = openDriver();
                Session session = openSession(driver, database)) {
            Map<String, Object> parameters = Map.of(
                    "graphName", graphName,
                    "sourceNodeId", sourceNodeId,
                    "concurrency", concurrency);
            String configuration = "{sourceNode: $sourceNodeId, concurrency: $concurrency, "
                    + "logProgress: false}";
            long reached = session.run(
                            "CALL gds.bfs.stream($graphName, " + configuration + ") "
                                    + "YIELD nodeIds RETURN size(nodeIds) AS reached",
                            parameters)
                    .single()
                    .get("reached")
                    .asLong();
            String statsStatement = "CALL gds.bfs.stats($graphName, " + configuration + ") "
                    + "YIELD preProcessingMillis, computeMillis, postProcessingMillis "
                    + "RETURN preProcessingMillis, computeMillis, postProcessingMillis";
            for (int warmup = 0; warmup < warmups; warmup++) {
                session.run(statsStatement, parameters).consume();
            }
            long[] clientSamples = new long[runs];
            long[] preProcessingSamples = new long[runs];
            long[] computeSamples = new long[runs];
            long[] postProcessingSamples = new long[runs];
            for (int run = 0; run < runs; run++) {
                long started = System.nanoTime();
                var result = session.run(statsStatement, parameters).single();
                clientSamples[run] = System.nanoTime() - started;
                preProcessingSamples[run] = result.get("preProcessingMillis").asLong();
                computeSamples[run] = result.get("computeMillis").asLong();
                postProcessingSamples[run] = result.get("postProcessingMillis").asLong();
            }
            Arrays.sort(clientSamples);
            Arrays.sort(preProcessingSamples);
            Arrays.sort(computeSamples);
            Arrays.sort(postProcessingSamples);
            System.out.printf("graph_name\t%s%n", graphName);
            System.out.printf("source_node_id\t%d%n", sourceNodeId);
            System.out.printf("reached_nodes\t%d%n", reached);
            System.out.printf("runs\t%d%n", runs);
            System.out.printf("concurrency\t%d%n", concurrency);
            System.out.printf("server_preprocess_p50_ms\t%d%n", percentile(preProcessingSamples, 0.50));
            System.out.printf("server_compute_p50_ms\t%d%n", percentile(computeSamples, 0.50));
            System.out.printf("server_compute_p95_ms\t%d%n", percentile(computeSamples, 0.95));
            System.out.printf("server_postprocess_p50_ms\t%d%n", percentile(postProcessingSamples, 0.50));
            System.out.printf(Locale.ROOT, "client_p50_ms\t%.3f%n", millis(percentile(clientSamples, 0.50)));
            System.out.printf(Locale.ROOT, "client_p95_ms\t%.3f%n", millis(percentile(clientSamples, 0.95)));
        }
    }

    private static BenchmarkResult runQueries(
            QueryRunner runner,
            FloatMatrix queries,
            IntMatrix truth,
            int k,
            Double lower,
            int warmups,
            int dimension) {
        for (int warmup = 0; warmup < warmups; warmup++) {
            search(runner, queries.row(warmup % queries.rows()), k, lower, dimension);
        }
        long[] samples = new long[queries.rows()];
        double recall = 0.0;
        int minimumResults = Integer.MAX_VALUE;
        for (int query = 0; query < queries.rows(); query++) {
            long started = System.nanoTime();
            List<Hit> hits = search(runner, queries.row(query), k, lower, dimension);
            samples[query] = System.nanoTime() - started;
            minimumResults = Math.min(minimumResults, hits.size());
            recall += recall(hits, truth.row(query), k);
        }
        Arrays.sort(samples);
        return new BenchmarkResult(samples, recall / queries.rows(), minimumResults);
    }

    private static List<Hit> search(
            QueryRunner runner,
            float[] query,
            int k,
            Double inclusiveLower,
            int dimension) {
        String filter = inclusiveLower == null ? "" : "WHERE n." + RATING_PROPERTY + " >= $lower";
        String statement = String.format(
                Locale.ROOT,
                """
                CYPHER 25
                MATCH (n:Vector)
                  SEARCH n IN (
                    VECTOR INDEX vectors
                    FOR $query
                    %s
                    LIMIT %d
                  ) SCORE AS score
                RETURN n.external_id AS id, score
                """,
                filter,
                k);
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("query", Values.vector(query));
        if (inclusiveLower != null) {
            parameters.put("lower", inclusiveLower);
        }
        return runner.run(statement, parameters).list(record ->
                new Hit(record.get("id").asLong(), record.get("score").asFloat()));
    }

    private static void createVectorIndex(
            Session session,
            int dimensions,
            boolean filtered,
            String quantization,
            double expansion,
            int hnswM,
            int efConstruction) {
        String additionalProperty = filtered ? " WITH [n." + RATING_PROPERTY + "]" : "";
        String statement = String.format(
                Locale.ROOT,
                """
                CYPHER 25
                CREATE VECTOR INDEX vectors IF NOT EXISTS
                FOR (n:Vector) ON n.embedding%s
                OPTIONS {indexConfig: {
                  `vector.dimensions`: %d,
                  `vector.similarity_function`: 'cosine',
                  `vector.quantization.type`: '%s',
                  `vector.default_search_expansion_factor`: %.6f,
                  `vector.hnsw.m`: %d,
                  `vector.hnsw.ef_construction`: %d
                }}
                """,
                additionalProperty,
                dimensions,
                quantization,
                expansion,
                hnswM,
                efConstruction);
        session.run(statement).consume();
        long deadline = System.nanoTime() + 2L * 60 * 60 * 1_000_000_000;
        while (true) {
            var status = session.run(
                            "SHOW VECTOR INDEXES YIELD name, state, populationPercent, failureMessage "
                                    + "WHERE name = $name RETURN state, populationPercent, failureMessage",
                            Map.of("name", VECTOR_INDEX))
                    .single();
            String state = status.get("state").asString();
            if (state.equals("ONLINE")) {
                return;
            }
            if (state.equals("FAILED")) {
                throw new IllegalStateException(status.get("failureMessage").asString());
            }
            if (System.nanoTime() > deadline) {
                throw new IllegalStateException("timed out waiting for vector index");
            }
            try {
                Thread.sleep(200);
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted waiting for vector index", error);
            }
        }
    }

    private static Map<String, Object> row(long id, float[] vector, Double rating) {
        Map<String, Object> row = new HashMap<>();
        row.put("id", id);
        row.put("embedding", Values.vector(vector));
        if (rating != null) {
            row.put("rating", rating);
        }
        return row;
    }

    private static Driver openDriver() {
        String uri = requiredEnvironment("NEO4J_URI");
        String username = requiredEnvironment("NEO4J_USERNAME");
        String password = requiredEnvironment("NEO4J_PASSWORD");
        Config config = Config.builder()
                .withMaxConnectionPoolSize(4)
                .withConnectionAcquisitionTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
                .withLogging(Logging.none())
                .build();
        return GraphDatabase.driver(uri, AuthTokens.basic(username, password), config);
    }

    private static Session openSession(Driver driver, String database) {
        return driver.session(SessionConfig.builder()
                .withDatabase(database)
                .withFetchSize(1_000)
                .build());
    }

    private static String requiredEnvironment(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("missing environment variable " + name);
        }
        return value;
    }

    private static double recall(List<Hit> hits, int[] truth, int k) {
        Set<Long> expected = new HashSet<>(k * 2);
        for (int index = 0; index < k; index++) {
            expected.add(Integer.toUnsignedLong(truth[index]));
        }
        int matches = 0;
        for (Hit hit : hits) {
            if (expected.contains(hit.id())) {
                matches++;
            }
        }
        return matches / (double) k;
    }

    private static float[] readVector(FileChannel channel, ByteBuffer encoded, int dimensions)
            throws IOException {
        encoded.clear();
        while (encoded.hasRemaining()) {
            if (channel.read(encoded) < 0) {
                throw new IOException("vector matrix ended inside a row");
            }
        }
        encoded.flip();
        float[] vector = new float[dimensions];
        encoded.asFloatBuffer().get(vector);
        return vector;
    }

    private static MatrixHeader readMatrixHeader(Path path, int valueWidth) throws IOException {
        long length = Files.size(path);
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            while (header.hasRemaining()) {
                if (channel.read(header) < 0) {
                    throw new IOException("matrix header is truncated");
                }
            }
            header.flip();
            int rows = header.getInt();
            int columns = header.getInt();
            long expected = 8L + Math.multiplyExact(Math.multiplyExact((long) rows, columns), valueWidth);
            if (rows < 0 || columns <= 0 || length != expected) {
                throw new IOException("matrix shape does not match file length: " + path);
            }
            return new MatrixHeader(rows, columns);
        }
    }

    private static FloatMatrix readFloatMatrix(Path path, int requestedRows) throws IOException {
        MatrixHeader header = readMatrixHeader(path, Float.BYTES);
        int rows = Math.min(requestedRows, header.rows());
        float[] values = new float[Math.multiplyExact(rows, header.columns())];
        readFloats(path, values);
        return new FloatMatrix(rows, header.columns(), values);
    }

    private static void readFloats(Path path, float[] values) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            channel.position(8);
            ByteBuffer buffer = ByteBuffer.allocateDirect(Math.min(values.length * Float.BYTES, 1 << 20))
                    .order(ByteOrder.LITTLE_ENDIAN);
            int cursor = 0;
            while (cursor < values.length) {
                buffer.clear();
                buffer.limit(Math.min(buffer.capacity(), (values.length - cursor) * Float.BYTES));
                while (buffer.hasRemaining()) {
                    if (channel.read(buffer) < 0) {
                        throw new IOException("float matrix is truncated");
                    }
                }
                buffer.flip();
                int count = buffer.remaining() / Float.BYTES;
                buffer.asFloatBuffer().get(values, cursor, count);
                cursor += count;
            }
        }
    }

    private static IntMatrix readIntMatrix(Path path, int requestedRows) throws IOException {
        MatrixHeader header = readMatrixHeader(path, Integer.BYTES);
        int rows = Math.min(requestedRows, header.rows());
        int[] values = new int[Math.multiplyExact(rows, header.columns())];
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            channel.position(8);
            ByteBuffer buffer = ByteBuffer.allocateDirect(Math.min(values.length * Integer.BYTES, 1 << 20))
                    .order(ByteOrder.LITTLE_ENDIAN);
            int cursor = 0;
            while (cursor < values.length) {
                buffer.clear();
                buffer.limit(Math.min(buffer.capacity(), (values.length - cursor) * Integer.BYTES));
                while (buffer.hasRemaining()) {
                    if (channel.read(buffer) < 0) {
                        throw new IOException("integer matrix is truncated");
                    }
                }
                buffer.flip();
                int count = buffer.remaining() / Integer.BYTES;
                buffer.asIntBuffer().get(values, cursor, count);
                cursor += count;
            }
        }
        return new IntMatrix(rows, header.columns(), values);
    }

    private static String parseQuantization(String value) {
        return switch (value) {
            case "none", "scalar", "binary" -> value;
            default -> throw new IllegalArgumentException("unknown quantization: " + value);
        };
    }

    private static int positiveInt(String value, String name) {
        int parsed = Integer.parseInt(value);
        if (parsed <= 0) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return parsed;
    }

    private static int positiveOrZeroInt(String value, String name) {
        int parsed = Integer.parseInt(value);
        if (parsed < 0) {
            throw new IllegalArgumentException(name + " must not be negative");
        }
        return parsed;
    }

    private static double positiveDouble(String value, String name) {
        double parsed = Double.parseDouble(value);
        if (!Double.isFinite(parsed) || parsed <= 0.0) {
            throw new IllegalArgumentException(name + " must be finite and positive");
        }
        return parsed;
    }

    private static long percentile(long[] sorted, double percentile) {
        return sorted[(int) Math.ceil((sorted.length - 1) * percentile)];
    }

    private static double seconds(long nanoseconds) {
        return nanoseconds / 1_000_000_000.0;
    }

    private static double millis(long nanoseconds) {
        return nanoseconds / 1_000_000.0;
    }

    private static void requireLength(String[] arguments, int length, String usage) {
        if (arguments.length != length) {
            throw new IllegalArgumentException("usage: " + usage);
        }
    }

    private static void usage() {
        throw new IllegalArgumentException(
                "commands: probe, smoke, import-fbin, reindex, bench-fbin, bench-gds-bfs");
    }

    private record MatrixHeader(int rows, int columns) {}

    private record FloatMatrix(int rows, int columns, float[] values) {
        float[] row(int row) {
            return Arrays.copyOfRange(values, row * columns, (row + 1) * columns);
        }
    }

    private record IntMatrix(int rows, int columns, int[] values) {
        int[] row(int row) {
            return Arrays.copyOfRange(values, row * columns, (row + 1) * columns);
        }
    }

    private record Hit(long id, float score) {}

    private record BenchmarkResult(long[] samples, double recall, int minimumResults) {}
}
