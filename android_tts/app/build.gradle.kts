plugins {
    id("com.android.application")
}

android {
    namespace = "com.holos.tts.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.holos.tts.app"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
    }

    signingConfigs {
        create("release") {
            storeFile = file(System.getenv("RELEASE_KEYSTORE") ?: "")
            storePassword = System.getenv("RELEASE_KEYSTORE_PASSWORD") ?: ""
            keyAlias = System.getenv("RELEASE_KEY_ALIAS") ?: ""
            keyPassword = System.getenv("RELEASE_KEY_PASSWORD") ?: ""
        }
    }

    buildTypes {
        debug {
            // AGP signs debug builds with the default debug keystore automatically
        }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1,LGPL3.0}"
        }
    }
}

dependencies {
    implementation(project(":holos-onnx"))
    implementation(project(":g2p"))
    implementation("androidx.core:core-ktx:1.12.0")
}
