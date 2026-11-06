plugins {
    id("org.jetbrains.kotlin.jvm")
}

group = "com.holos.tts"
version = "1.0.0"

dependencies {
    implementation(kotlin("stdlib"))
    testImplementation(kotlin("test"))
    testImplementation(kotlin("test-junit"))
}
