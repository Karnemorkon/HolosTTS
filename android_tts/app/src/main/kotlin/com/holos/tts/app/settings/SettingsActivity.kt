package com.holos.tts.app.settings

import android.app.Activity
import android.os.Bundle
import android.widget.TextView

class SettingsActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this)
        tv.text = "Налаштування HolosTTS\nГолос: перший пресет\nДвигун: ONNX (CPU INT8)\nШвидкість: 1.0"
        setContentView(tv)
    }
}
