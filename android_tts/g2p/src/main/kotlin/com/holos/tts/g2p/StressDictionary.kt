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
     * Формат визначається автоматично: якщо потік починається з магії gzip
     * (1F 8B) — він розпаковується, інакше читається як plain text UTF-8.
     * Можна викликати декілька разів — попередні дані будуть замінені.
     */
    fun load(inputStream: InputStream) {
        stressMap.clear()
        // Buffer the head to sniff the format without consuming the stream permanently.
        val head = ByteArray(2)
        var read = 0
        while (read < 2) {
            val n = inputStream.read(head, read, 2 - read)
            if (n < 0) break
            read += n
        }
        val isGzip = read == 2 && (head[0].toInt() and 0xFF) == 0x1F && (head[1].toInt() and 0xFF) == 0x8B
        val source: InputStream = if (isGzip) {
            java.io.SequenceInputStream(java.io.ByteArrayInputStream(head, 0, read), inputStream)
                .let { GZIPInputStream(it) }
        } else {
            java.io.SequenceInputStream(java.io.ByteArrayInputStream(head, 0, read), inputStream)
        }
        source.use { raw ->
            InputStreamReader(raw, java.nio.charset.StandardCharsets.UTF_8).use { reader ->
                BufferedReader(reader).use { br ->
                    br.lines().forEach { line ->
                        if (line.isBlank()) return@forEach
                        val parts = line.split("\t", limit = 2)
                        if (parts.size != 2) return@forEach
                        val word = parts[0]
                        // Друге поле може бути порожнім (слова без наголосу, напр. «а», «б»).
                        if (parts[1].isBlank()) return@forEach
                        val accents = try {
                            parts[1].split(",").map { it.toInt() }.toIntArray()
                        } catch (e: NumberFormatException) {
                            // Пропускаємо рядок із некоректними індексами.
                            return@forEach
                        }
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
