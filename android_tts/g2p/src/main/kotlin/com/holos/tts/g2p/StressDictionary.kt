package com.holos.tts.g2p

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.util.zip.GZIPInputStream

/**
 * Читає стиснений словник наголосів української мови.
 * Формат файлу (gzip): кожен рядок — `слово\tпозиція1,позиція2,...`
 * де позиції — індекси голосних з наголосом у слові (0-based).
 *
 * Словник конвертовано з `stress.trie` проєкту ukrainian-word-stress
 * (marisa_trie.BytesTrie → простий текстовий формат).
 */
object StressDictionary {
    private val stressMap = mutableMapOf<String, IntArray>()

    /**
     * Завантажує словник із вхідного потоку (наприклад, з Android assets).
     * Можна викликати декілька разів — попередні дані будуть замінені.
     */
    fun load(inputStream: InputStream) {
        stressMap.clear()
        GZIPInputStream(inputStream).use { gzip ->
            InputStreamReader(gzip, java.nio.charset.StandardCharsets.UTF_8).use { reader ->
                BufferedReader(reader).use { br ->
                    br.lines().forEach { line ->
                        if (line.isBlank()) return@forEach
                        val parts = line.split("\t", limit = 2)
                        if (parts.size != 2) return@forEach
                        val word = parts[0]
                        val accents = parts[1].split(",").map { it.toInt() }.toIntArray()
                        stressMap[word] = accents
                    }
                }
            }
        }
    }

    /** Повертає індекси голосних з наголосом для слова, або `null`, якщо слова немає в словнику. */
    fun getAccents(word: String): IntArray? = stressMap[word]

    val size: Int get() = stressMap.size
}
