plugins {
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.neo4j:neo4j:2026.06.0")
    implementation("org.neo4j.driver:neo4j-java-driver:6.2.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.20.0")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

application {
    mainClass = "dev.vecgra.bench.Neo4jBenchmark"
    applicationDefaultJvmArgs = listOf(
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-Xms2g",
        "-Xmx4g",
        "-XX:+UseZGC",
    )
}
