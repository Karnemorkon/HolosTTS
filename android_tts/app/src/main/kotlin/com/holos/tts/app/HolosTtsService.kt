package com.holos.tts.app

import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.AudioAttributes
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.LocaleList
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeechService
import android.speech.tts.TextToSpeech
import android.util.Log
import com.holos.tts.g2p.StressDictionary
import com.holos.tts.g2p.UkrainianTextToPhonemes
import com.holos.tts.HolosOnnx
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

class HolosTtsService : TextToSpeechService() {

    private companion object {
        const val TAG = "HolosTtsService"
        const val LANG = "uk"
        const val SAMPLE_RATE = 24000
    }

    private lateinit var synthThread: HandlerThread
    private lateinit var synthHandler: Handler
    private lateinit var onnx: HolosOnnx
    private var isReleased = false
    private var engineReady = false

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "=== HolosTtsService onCreate ===")
        synthThread = HandlerThread("HolosTts-synth", android.os.Process.THREAD_PRIORITY_BACKGROUND)
        synthThread.start()
        synthHandler = Handler(synthThread.looper)
        Log.i(TAG, "HandlerThread started")
    }

    override fun onBind(intent: Intent): android.os.IBinder? {
        Log.i(TAG, "=== onBind: ${intent.action} ===")
        val binder = super.onBind(intent)
        Log.i(TAG, "onBind returning binder: ${binder != null}")
        return binder
    }

    override fun onGetLanguage(): Array<String> {
        Log.i(TAG, "=== onGetLanguage ===")
        return arrayOf("uk")
    }

    override fun onIsLanguageAvailable(language: String, country: String?, variant: String?): Int {
        Log.i(TAG, "=== onIsLanguageAvailable: lang='$language' country='$country' variant='$variant' ===")
        val available = if (language.equals(LANG, ignoreCase = true) && (country == null || country.equals("UA", ignoreCase = true) || country.isEmpty())) {
            Log.i(TAG, "Language uk available")
            TextToSpeech.LANG_AVAILABLE
        } else {
            Log.i(TAG, "Language NOT supported")
            TextToSpeech.LANG_NOT_SUPPORTED
        }
        return available
    }

    override fun onLoadLanguage(language: String, country: String?, variant: String?): Int {
        Log.i(TAG, "=== onLoadLanguage: lang='$language' country='$country' variant='$variant' ===")
        if (language.equals(LANG, ignoreCase = true)) {
            Log.i(TAG, "Loading uk language, initializing engine...")
            synthHandler.post { initEngine() }
            return TextToSpeech.LANG_AVAILABLE
        }
        return TextToSpeech.LANG_NOT_SUPPORTED
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        val text = request.text?.toString() ?: ""
        Log.i(TAG, "=== onSynthesizeText: '$text' ===")
        synthHandler.post { synthesize(text, callback) }
    }

    override fun onStop() {
        Log.i(TAG, "=== onStop ===")
        synthHandler.removeCallbacksAndMessages(null)
        isReleased = true
    }

    override fun onDestroy() {
        Log.i(TAG, "=== onDestroy ===")
        synthThread.quitSafely()
        try { onnx.close() } catch (e: Exception) { Log.e(TAG, "Error closing ONNX", e) }
        super.onDestroy()
    }

    private fun initEngine() {
        Log.i(TAG, "=== initEngine ===")
        try {
            val assets = assets
            Log.i(TAG, "Loading stress dictionary from assets...")
            StressDictionary.load(assets.open("word_stress.txt.gz"))
            Log.i(TAG, "Stress dictionary loaded, words=${StressDictionary.size}")
            Log.i(TAG, "Loading ONNX model from models dir...")
            onnx = HolosOnnx(getModelDir(this), streaming = true)
            Log.i(TAG, "ONNX engine loaded, sampleRate=${onnx.sampleRate}, voices=${onnx.voices.size}")
            engineReady = true
            Log.i(TAG, "=== Engine ready ===")
        } catch (e: Exception) {
            Log.e(TAG, "=== Failed to init engine ===", e)
        }
    }

    private fun getModelDir(context: Context): File {
        val dir = File(context.filesDir, "onnx-int8")
        dir.mkdirs()
        val required = listOf("encoder.onnx", "prosody.onnx", "decoder_window.onnx", "source.onnx", "holos_onnx.json", "voices.npz")
        for (name in required) {
            val out = File(dir, name)
            if (!out.exists()) {
                Log.i(TAG, "Extracting $name from assets to ${out.absolutePath}")
                assets.open(name).use { input ->
                    input.copyTo(FileOutputStream(out), 8192)
                }
            } else {
                Log.i(TAG, "$name already exists (${out.length()} bytes)")
            }
        }
        return dir
    }

    private fun synthesize(text: String, callback: SynthesisCallback) {
        if (!engineReady) {
            Log.e(TAG, "Engine not initialized, cannot synthesize")
            return
        }
        val phonemes = UkrainianTextToPhonemes.toPhonemes(text, StressDictionary)
        Log.d(TAG, "Synthesizing: '$text' -> '$phonemes'")

        val sampleRate = onnx.sampleRate
        val minBufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build())
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .build())
            .setBufferSizeInBytes(max(minBufferSize, sampleRate * 2 * 2))
            .build()
        track.play()

        val voiceName = onnx.voices.firstOrNull()
        try {
            onnx.stream(phonemes, voiceName, 1.0f) { chunk: FloatArray ->
                if (isReleased) return@stream
                val pcm = floatToPcm16(chunk)
                val bytePcm = shortToByteArray(pcm)
                callback.audioAvailable(bytePcm, 0, bytePcm.size)
                track.write(pcm, 0, pcm.size, AudioTrack.WRITE_BLOCKING)
            }
        } finally {
            track.stop()
            track.release()
            callback.done()
        }
    }

    private fun floatToPcm16(samples: FloatArray): ShortArray {
        val pcm = ShortArray(samples.size)
        for (i in samples.indices) {
            val s = samples[i].coerceIn(-1f, 1f)
            pcm[i] = (s * 32767f).toInt().toShort()
        }
        return pcm
    }

    private fun shortToByteArray(shorts: ShortArray): ByteArray {
        val bytes = ByteArray(shorts.size * 2)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (s in shorts) buf.putShort(s)
        return bytes
    }
}
