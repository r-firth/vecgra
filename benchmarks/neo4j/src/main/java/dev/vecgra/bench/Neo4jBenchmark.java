package dev.vecgra.bench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.BufferedInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.dbms.api.DatabaseManagementServiceBuilder;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Direction;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.RelationshipType;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;
import org.neo4j.io.ByteUnit;

/** Same-process Neo4j baseline for Vecgra's external fbin workloads. */
public final class Neo4jBenchmark {
    private static final String DATABASE_NAME = "neo4j";
    private static final String VECTOR_LABEL = "Vector";
    private static final String VECTOR_INDEX = "vectors";
    private static final String ID_PROPERTY = "external_id";
    private static final String VECTOR_PROPERTY = "embedding";
    private static final String RATING_PROPERTY = "avg_rating";
    private static final String GRAPH_LABEL = "GraphVertex";
    private static final RelationshipType GRAPH_EDGE = RelationshipType.withName("LINK");
    private static final ObjectMapper JSON = new ObjectMapper();

    private Neo4jBenchmark() {}

    public static void main(String[] arguments) throws Exception {
        if (arguments.length == 0) {
            usage();
        }
        switch (arguments[0]) {
            case "smoke" -> smoke(arguments);
            case "import-fbin" -> importFbin(arguments);
            case "reindex" -> reindex(arguments);
            case "bench-fbin" -> benchmarkFbin(arguments);
            case "import-graphalytics" -> importGraphalytics(arguments);
            case "bench-bfs" -> benchmarkBfs(arguments);
            case "remote" -> Neo4jRemoteBenchmark.main(Arrays.copyOfRange(arguments, 1, arguments.length));
            default -> usage();
        }
    }

    private static void smoke(String[] arguments) throws Exception {
        requireLength(arguments, 2, "smoke <database-directory>");
        Path directory = Path.of(arguments[1]);
        DatabaseManagementService management = open(directory);
        try {
            GraphDatabaseService database = management.database(DATABASE_NAME);
            try (Transaction transaction = database.beginTx()) {
                for (int index = 0; index < 4; index++) {
                    Node node = transaction.createNode(Label.label(VECTOR_LABEL));
                    node.setProperty(ID_PROPERTY, (long) index);
                    node.setProperty(RATING_PROPERTY, 8.0 + index * 0.5);
                    node.setProperty(
                            VECTOR_PROPERTY,
                            new float[] {
                                index == 0 ? 1.0f : 0.0f,
                                index == 1 ? 1.0f : 0.0f,
                                0.1f
                            });
                }
                transaction.commit();
            }
            createVectorIndex(database, 3, true, "scalar", 1.5, 16, 100);
            List<Hit> hits = search(database, new float[] {1.0f, 0.0f, 0.1f}, 2, 8.0);
            for (Hit hit : hits) {
                System.out.printf(Locale.ROOT, "id=%d score=%f%n", hit.id(), hit.score());
            }
        } finally {
            management.shutdown();
        }
    }

    private static void importFbin(String[] arguments) throws Exception {
        requireLength(
                arguments,
                7,
                "import-fbin <database-directory> <train.fbin> <metadata.jsonl|-> "
                        + "<batch-size> <none|scalar|binary> <expansion-factor>");
        Path directory = Path.of(arguments[1]);
        Path vectorsPath = Path.of(arguments[2]);
        Path metadataPath = arguments[3].equals("-") ? null : Path.of(arguments[3]);
        int batchSize = positiveInt(arguments[4], "batch size");
        String quantization = parseQuantization(arguments[5]);
        double expansion = positiveDouble(arguments[6], "expansion factor");
        if (Files.exists(directory) && directorySize(directory) != 0) {
            throw new IllegalArgumentException("database directory must be empty: " + directory);
        }
        Files.createDirectories(directory);

        MatrixHeader header = readMatrixHeader(vectorsPath, Float.BYTES);
        DatabaseManagementService management = open(directory);
        long importStarted = System.nanoTime();
        long indexNanos;
        try {
            GraphDatabaseService database = management.database(DATABASE_NAME);
            try (FileChannel channel = FileChannel.open(vectorsPath, StandardOpenOption.READ);
                    BufferedReader metadata = metadataPath == null
                            ? null
                            : Files.newBufferedReader(metadataPath, StandardCharsets.UTF_8)) {
                channel.position(8);
                ByteBuffer encoded = ByteBuffer.allocateDirect(header.columns() * Float.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
                int row = 0;
                while (row < header.rows()) {
                    int end = Math.min(header.rows(), row + batchSize);
                    try (Transaction transaction = database.beginTx()) {
                        for (; row < end; row++) {
                            float[] vector = readVector(channel, encoded, header.columns());
                            Node node = transaction.createNode(Label.label(VECTOR_LABEL));
                            node.setProperty(ID_PROPERTY, (long) row);
                            node.setProperty(VECTOR_PROPERTY, vector);
                            if (metadata != null) {
                                String line = metadata.readLine();
                                if (line == null) {
                                    throw new IOException("metadata ended before vector row " + row);
                                }
                                JsonNode record = JSON.readTree(line);
                                JsonNode rating = record.path("properties").path(RATING_PROPERTY);
                                if (rating.isNumber()) {
                                    node.setProperty(RATING_PROPERTY, rating.doubleValue());
                                }
                            }
                        }
                        transaction.commit();
                    }
                    if (row % 100_000 == 0 || row == header.rows()) {
                        System.err.printf("stored %d/%d vectors%n", row, header.rows());
                    }
                }
                if (metadata != null && metadata.readLine() != null) {
                    throw new IOException("metadata has more rows than the vector matrix");
                }
            }
            long dataNanos = System.nanoTime() - importStarted;
            long indexStarted = System.nanoTime();
            createVectorIndex(
                    database,
                    header.columns(),
                    metadataPath != null,
                    quantization,
                    expansion,
                    16,
                    100);
            indexNanos = System.nanoTime() - indexStarted;
            System.out.printf("rows\t%d%n", header.rows());
            System.out.printf("dimension\t%d%n", header.columns());
            System.out.printf(Locale.ROOT, "data_import_s\t%.3f%n", seconds(dataNanos));
            System.out.printf(Locale.ROOT, "vector_index_s\t%.3f%n", seconds(indexNanos));
            System.out.printf("quantization\t%s%n", quantization);
            System.out.printf(Locale.ROOT, "expansion_factor\t%.3f%n", expansion);
        } finally {
            management.shutdown();
        }
        System.out.printf("store_bytes\t%d%n", directorySize(directory));
    }

    private static void benchmarkFbin(String[] arguments) throws Exception {
        requireLength(
                arguments,
                8,
                "bench-fbin <database-directory> <queries.fbin> <truth.ibin> "
                        + "<query-count> <k> <inclusive-lower-bound|-> <warmup-count>");
        Path directory = Path.of(arguments[1]);
        Path queriesPath = Path.of(arguments[2]);
        Path truthPath = Path.of(arguments[3]);
        int requestedQueries = positiveInt(arguments[4], "query count");
        int k = positiveInt(arguments[5], "k");
        Double lower = arguments[6].equals("-") ? null : Double.parseDouble(arguments[6]);
        int warmupCount = positiveInt(arguments[7], "warmup count");

        FloatMatrix queries = readFloatMatrix(queriesPath, requestedQueries);
        IntMatrix truth = readIntMatrix(truthPath, requestedQueries);
        if (queries.rows() != truth.rows() || k > truth.columns()) {
            throw new IllegalArgumentException("query and truth shapes do not agree");
        }

        long openStarted = System.nanoTime();
        DatabaseManagementService management = open(directory);
        try {
            GraphDatabaseService database = management.database(DATABASE_NAME);
            long openNanos = System.nanoTime() - openStarted;
            for (int warmup = 0; warmup < warmupCount; warmup++) {
                search(database, queries.row(warmup % queries.rows()), k, lower);
            }
            long[] samples = new long[queries.rows()];
            double recall = 0.0;
            int minimumResults = Integer.MAX_VALUE;
            long eligibleVectors = lower == null ? countVectors(database) : countEligible(database, lower);
            for (int query = 0; query < queries.rows(); query++) {
                long started = System.nanoTime();
                List<Hit> hits = search(database, queries.row(query), k, lower);
                samples[query] = System.nanoTime() - started;
                minimumResults = Math.min(minimumResults, hits.size());
                recall += recall(hits, truth.row(query), k);
            }
            Arrays.sort(samples);
            System.out.printf("queries\t%d%n", queries.rows());
            System.out.printf("k\t%d%n", k);
            System.out.printf("filter_lower\t%s%n", lower == null ? "none" : lower);
            System.out.printf("eligible_vectors\t%d%n", eligibleVectors);
            System.out.printf(Locale.ROOT, "recall_at_k\t%.6f%n", recall / queries.rows());
            System.out.printf("minimum_results\t%d%n", minimumResults);
            System.out.printf(Locale.ROOT, "open_ms\t%.3f%n", millis(openNanos));
            System.out.printf(Locale.ROOT, "query_p50_ms\t%.3f%n", millis(percentile(samples, 0.50)));
            System.out.printf(Locale.ROOT, "query_p95_ms\t%.3f%n", millis(percentile(samples, 0.95)));
            System.out.printf(Locale.ROOT, "query_max_ms\t%.3f%n", millis(samples[samples.length - 1]));
            System.out.printf("store_bytes\t%d%n", directorySize(directory));
        } finally {
            management.shutdown();
        }
    }

    private static void reindex(String[] arguments) throws Exception {
        if (arguments.length != 6 && arguments.length != 8) {
            throw new IllegalArgumentException(
                    "usage: reindex <database-directory> <dimension> <filtered|unfiltered> "
                            + "<none|scalar|binary> <expansion-factor> [hnsw-m ef-construction]");
        }
        Path directory = Path.of(arguments[1]);
        int dimensions = positiveInt(arguments[2], "dimension");
        boolean filtered = switch (arguments[3]) {
            case "filtered" -> true;
            case "unfiltered" -> false;
            default -> throw new IllegalArgumentException("expected filtered or unfiltered");
        };
        String quantization = parseQuantization(arguments[4]);
        double expansion = positiveDouble(arguments[5], "expansion factor");
        int hnswM = arguments.length == 8 ? positiveInt(arguments[6], "HNSW M") : 16;
        int efConstruction = arguments.length == 8
                ? positiveInt(arguments[7], "HNSW ef-construction")
                : 100;
        DatabaseManagementService management = open(directory);
        long started = System.nanoTime();
        try {
            GraphDatabaseService database = management.database(DATABASE_NAME);
            database.executeTransactionally("CYPHER 25 DROP INDEX " + VECTOR_INDEX + " IF EXISTS");
            createVectorIndex(
                    database,
                    dimensions,
                    filtered,
                    quantization,
                    expansion,
                    hnswM,
                    efConstruction);
        } finally {
            management.shutdown();
        }
        System.out.printf(Locale.ROOT, "reindex_s\t%.3f%n", seconds(System.nanoTime() - started));
        System.out.printf("quantization\t%s%n", quantization);
        System.out.printf(Locale.ROOT, "expansion_factor\t%.3f%n", expansion);
        System.out.printf("hnsw_m\t%d%n", hnswM);
        System.out.printf("ef_construction\t%d%n", efConstruction);
        System.out.printf("store_bytes\t%d%n", directorySize(directory));
    }

    private static void importGraphalytics(String[] arguments) throws Exception {
        requireLength(
                arguments,
                5,
                "import-graphalytics <database-directory> <vertices> <edges> <batch-size>");
        Path directory = Path.of(arguments[1]);
        Path verticesPath = Path.of(arguments[2]);
        Path edgesPath = Path.of(arguments[3]);
        int batchSize = positiveInt(arguments[4], "batch size");
        if (Files.exists(directory) && directorySize(directory) != 0) {
            throw new IllegalArgumentException("database directory must be empty: " + directory);
        }
        Files.createDirectories(directory);

        long started = System.nanoTime();
        DatabaseManagementService management = open(directory);
        long nodeCount = 0;
        long edgeCount = 0;
        try {
            GraphDatabaseService database = management.database(DATABASE_NAME);
            try (FastLongReader vertices = new FastLongReader(verticesPath)) {
                while (vertices.hasNext()) {
                    try (Transaction transaction = database.beginTx()) {
                        for (int batch = 0; batch < batchSize && vertices.hasNext(); batch++) {
                            long externalId = vertices.nextLong();
                            Node node = transaction.createNode(Label.label(GRAPH_LABEL));
                            if (node.getId() != externalId) {
                                throw new IOException(
                                        "Graphalytics IDs are not dense Neo4j IDs at " + externalId);
                            }
                            nodeCount++;
                        }
                        transaction.commit();
                    }
                    if (nodeCount % 1_000_000 == 0) {
                        System.err.printf("stored %d vertices%n", nodeCount);
                    }
                }
            }
            try (FastLongReader edges = new FastLongReader(edgesPath)) {
                while (edges.hasNext()) {
                    try (Transaction transaction = database.beginTx()) {
                        for (int batch = 0; batch < batchSize && edges.hasNext(); batch++) {
                            long source = edges.nextLong();
                            long target = edges.nextLong();
                            transaction.getNodeById(source)
                                    .createRelationshipTo(transaction.getNodeById(target), GRAPH_EDGE);
                            edgeCount++;
                        }
                        transaction.commit();
                    }
                    if (edgeCount % 1_000_000 == 0) {
                        System.err.printf("stored %d edges%n", edgeCount);
                    }
                }
            }
        } finally {
            management.shutdown();
        }
        System.out.printf("nodes\t%d%n", nodeCount);
        System.out.printf("edges\t%d%n", edgeCount);
        System.out.printf(Locale.ROOT, "import_s\t%.3f%n", seconds(System.nanoTime() - started));
        System.out.printf("store_bytes\t%d%n", directorySize(directory));
    }

    private static void benchmarkBfs(String[] arguments) throws Exception {
        requireLength(
                arguments,
                6,
                "bench-bfs <database-directory> <source> <reference-output|-> <iterations> <warmups>");
        Path directory = Path.of(arguments[1]);
        int source = positiveOrZeroInt(arguments[2], "source");
        Path reference = arguments[3].equals("-") ? null : Path.of(arguments[3]);
        int iterations = positiveInt(arguments[4], "iterations");
        int warmups = positiveOrZeroInt(arguments[5], "warmups");

        long openStarted = System.nanoTime();
        DatabaseManagementService management = open(directory);
        try {
            GraphDatabaseService database = management.database(DATABASE_NAME);
            long openNanos = System.nanoTime() - openStarted;
            int nodeCount = Math.toIntExact(count(
                    database,
                    "CYPHER 25 MATCH (n:" + GRAPH_LABEL + ") RETURN count(n) AS count",
                    Map.of()));
            if (source >= nodeCount) {
                throw new IllegalArgumentException("source is outside the graph");
            }

            int[] first = bfs(database, nodeCount, source);
            if (reference != null) {
                validateBfs(reference, first);
            }
            for (int warmup = 0; warmup < warmups; warmup++) {
                bfs(database, nodeCount, source);
            }
            long[] samples = new long[iterations];
            for (int iteration = 0; iteration < iterations; iteration++) {
                long started = System.nanoTime();
                int[] distances = bfs(database, nodeCount, source);
                samples[iteration] = System.nanoTime() - started;
                if (!Arrays.equals(first, distances)) {
                    throw new IllegalStateException("BFS result changed between iterations");
                }
            }
            Arrays.sort(samples);
            int reached = 0;
            int maximumDistance = 0;
            for (int distance : first) {
                if (distance != Integer.MAX_VALUE) {
                    reached++;
                    maximumDistance = Math.max(maximumDistance, distance);
                }
            }
            System.out.printf("nodes\t%d%n", nodeCount);
            System.out.printf("source\t%d%n", source);
            System.out.printf("reached\t%d%n", reached);
            System.out.printf("max_distance\t%d%n", maximumDistance);
            System.out.printf("reference_validated\t%s%n", reference != null);
            System.out.printf(Locale.ROOT, "open_ms\t%.3f%n", millis(openNanos));
            System.out.printf(Locale.ROOT, "bfs_p50_ms\t%.3f%n", millis(percentile(samples, 0.50)));
            System.out.printf(Locale.ROOT, "bfs_p95_ms\t%.3f%n", millis(percentile(samples, 0.95)));
            System.out.printf(Locale.ROOT, "bfs_max_ms\t%.3f%n", millis(samples[samples.length - 1]));
            System.out.printf("store_bytes\t%d%n", directorySize(directory));
        } finally {
            management.shutdown();
        }
    }

    private static int[] bfs(GraphDatabaseService database, int nodeCount, int source) {
        int[] distances = new int[nodeCount];
        Arrays.fill(distances, Integer.MAX_VALUE);
        int[] queue = new int[nodeCount];
        int head = 0;
        int tail = 0;
        distances[source] = 0;
        queue[tail++] = source;
        try (Transaction transaction = database.beginTx()) {
            while (head < tail) {
                int current = queue[head++];
                int nextDistance = distances[current] + 1;
                Node node = transaction.getNodeById(current);
                for (Relationship relationship : node.getRelationships(Direction.OUTGOING, GRAPH_EDGE)) {
                    int target = Math.toIntExact(relationship.getEndNodeId());
                    if (distances[target] == Integer.MAX_VALUE) {
                        distances[target] = nextDistance;
                        queue[tail++] = target;
                    }
                }
            }
            transaction.commit();
        }
        return distances;
    }

    private static void validateBfs(Path reference, int[] distances) throws IOException {
        int rows = 0;
        try (FastLongReader expected = new FastLongReader(reference)) {
            while (expected.hasNext()) {
                int id = Math.toIntExact(expected.nextLong());
                long distance = expected.nextLong();
                long actual = distances[id] == Integer.MAX_VALUE
                        ? Long.MAX_VALUE
                        : distances[id];
                if (actual != distance) {
                    throw new IOException(
                            "BFS mismatch at vertex " + id + ": expected " + distance + ", got " + actual);
                }
                rows++;
            }
        }
        if (rows != distances.length) {
            throw new IOException("BFS reference has " + rows + " rows for " + distances.length + " vertices");
        }
    }

    private static DatabaseManagementService open(Path directory) {
        return new DatabaseManagementServiceBuilder(directory)
                .setConfig(GraphDatabaseSettings.pagecache_memory, ByteUnit.gibiBytes(1))
                .build();
    }

    private static void createVectorIndex(
            GraphDatabaseService database,
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
                CREATE VECTOR INDEX %s IF NOT EXISTS
                FOR (n:%s) ON n.%s%s
                OPTIONS {indexConfig: {
                  `vector.dimensions`: %d,
                  `vector.similarity_function`: 'cosine',
                  `vector.quantization.type`: '%s',
                  `vector.default_search_expansion_factor`: %.6f,
                  `vector.hnsw.m`: %d,
                  `vector.hnsw.ef_construction`: %d
                }}
                """,
                VECTOR_INDEX,
                VECTOR_LABEL,
                VECTOR_PROPERTY,
                additionalProperty,
                dimensions,
                quantization,
                expansion,
                hnswM,
                efConstruction);
        database.executeTransactionally(statement);
        try (Transaction transaction = database.beginTx()) {
            transaction.schema().awaitIndexesOnline(2, TimeUnit.HOURS);
            transaction.commit();
        }
    }

    private static List<Hit> search(
            GraphDatabaseService database, float[] query, int k, Double inclusiveLower) {
        String filter = inclusiveLower == null ? "" : "WHERE n." + RATING_PROPERTY + " >= $lower";
        String statement = String.format(
                Locale.ROOT,
                """
                CYPHER 25
                MATCH (n:%s)
                  SEARCH n IN (
                    VECTOR INDEX %s
                    FOR $query
                    %s
                    LIMIT %d
                  ) SCORE AS score
                RETURN n.%s AS id, score
                """,
                VECTOR_LABEL,
                VECTOR_INDEX,
                filter,
                k,
                ID_PROPERTY);
        Map<String, Object> parameters = inclusiveLower == null
                ? Map.of("query", boxed(query))
                : Map.of("query", boxed(query), "lower", inclusiveLower);
        List<Hit> hits = new ArrayList<>(k);
        try (Transaction transaction = database.beginTx();
                Result result = transaction.execute(statement, parameters)) {
            while (result.hasNext()) {
                Map<String, Object> row = result.next();
                hits.add(new Hit(((Number) row.get("id")).longValue(), ((Number) row.get("score")).floatValue()));
            }
            transaction.commit();
        }
        return hits;
    }

    private static long countVectors(GraphDatabaseService database) {
        return count(database, "CYPHER 25 MATCH (n:" + VECTOR_LABEL + ") RETURN count(n) AS count", Map.of());
    }

    private static long countEligible(GraphDatabaseService database, double inclusiveLower) {
        return count(
                database,
                "CYPHER 25 MATCH (n:" + VECTOR_LABEL + ") WHERE n." + RATING_PROPERTY
                        + " >= $lower RETURN count(n) AS count",
                Map.of("lower", inclusiveLower));
    }

    private static long count(GraphDatabaseService database, String statement, Map<String, Object> parameters) {
        try (Transaction transaction = database.beginTx();
                Result result = transaction.execute(statement, parameters)) {
            long value = ((Number) result.next().get("count")).longValue();
            transaction.commit();
            return value;
        }
    }

    private static List<Float> boxed(float[] values) {
        List<Float> boxed = new ArrayList<>(values.length);
        for (float value : values) {
            boxed.add(value);
        }
        return boxed;
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
        return new FloatMatrix(rows, header.columns(), values);
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

    private static long directorySize(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return 0;
        }
        try (var paths = Files.walk(directory)) {
            return paths.filter(Files::isRegularFile).mapToLong(path -> {
                try {
                    return Files.size(path);
                } catch (IOException error) {
                    throw new DirectorySizeException(error);
                }
            }).sum();
        } catch (DirectorySizeException error) {
            throw error.cause;
        }
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
                "commands: smoke, import-fbin, reindex, bench-fbin, import-graphalytics, bench-bfs, remote; "
                        + "run without enough arguments for syntax");
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

    private static final class DirectorySizeException extends RuntimeException {
        private final IOException cause;

        DirectorySizeException(IOException cause) {
            super(cause);
            this.cause = cause;
        }
    }

    private static final class FastLongReader implements AutoCloseable {
        private final InputStream input;
        private int next = -2;

        FastLongReader(Path path) throws IOException {
            input = new BufferedInputStream(Files.newInputStream(path), 1 << 20);
        }

        boolean hasNext() throws IOException {
            if (next != -2) {
                return next >= 0;
            }
            do {
                next = input.read();
            } while (next >= 0 && next <= ' ');
            return next >= 0;
        }

        long nextLong() throws IOException {
            if (!hasNext()) {
                throw new EOFException("expected another integer");
            }
            boolean negative = next == '-';
            long value = 0;
            int current = negative ? input.read() : next;
            while (current > ' ') {
                if (current < '0' || current > '9') {
                    throw new IOException("invalid integer byte: " + current);
                }
                value = Math.addExact(Math.multiplyExact(value, 10), current - '0');
                current = input.read();
            }
            next = -2;
            return negative ? -value : value;
        }

        @Override
        public void close() throws IOException {
            input.close();
        }
    }
}
