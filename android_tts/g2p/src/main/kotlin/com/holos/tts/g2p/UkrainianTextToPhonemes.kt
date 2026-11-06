package com.holos.tts.g2p

/**
 * Публічний API G2P: український текст → фонемний рядок (IPA), готовий для HolosTTS.
 *
 * Потік роботи:
 *   1. Розбиваємо текст на речення (punktuation: . ! ? …)
 *   2. Для кожного речення — ставимо наголоси через StressDictionary (словник + heuristic)
 *   3. Транскрибуємо кожне слово в IPA через UkrainianG2PEngine
 *   4. Повертаємо рядок фонем, розділених пробілами (речення — через "|")
 *
 * Model HolosTTS очікує фонEMI, не сирий текст, тому цей клас — вхідний鍵 модуля G2P.
 */
object UkrainianTextToPhonemes {

    private val sentenceSplitter = Regex("[.!?…]")

    /**
     * Перетворює сирий український текст на IPA-фонEMI.
     *
     * @param text сирий український текст (без наголосів)
     * @param stressDictionary завантажений словник наголосів
     * @return рядок IPA-фонем, де речення розділені "|", слова — пробілами
     */
    fun toPhonemes(text: String, stressDictionary: StressDictionary): String {
        val sentences = text.split(sentenceSplitter)
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        return sentences.map { sentence ->
            sentenceToPhonemes(sentence, stressDictionary)
        }.joinToString(" | ")
    }

    private fun sentenceToPhonemes(sentence: String, stressDictionary: StressDictionary): String {
        val words = sentence.split(Regex("\\s+")).filter { it.isNotEmpty() }
        return words.map { word ->
            val accents = stressDictionary.getAccents(normalizeWord(word))
            val accented = if (accents != null) applyAccents(word, accents) else word
            UkrainianG2PEngine.ipa(accented, checkAccent = false)
        }.joinToString(" ")
    }

    /** Прибирає пунктуацію і прізвища з крапками, приводить до вигляду для словника. */
    private fun normalizeWord(word: String): String {
        return word
            .replace(Regex("[.,;:()\"'«»–—]"), "")
            .lowercase()
    }

    /**
     * Накладає наголос (Combining Acute U+0301) на голосні за індексами зі словника.
     * Індекси — 0-based позиції голосних у слові.
     */
    private fun applyAccents(word: String, accentPositions: IntArray): String {
        val vowels = "аеєиіїоуюяАЕЄИІОУЮЯ"
        val result = StringBuilder(word)
        var vowelIndex = 0
        var offset = 0
        for (i in result.indices) {
            if (vowels.contains(result[i])) {
                if (vowelIndex in accentPositions) {
                    result.insert(i + offset, "\u0301")
                    offset += 1
                }
                vowelIndex++
            }
        }
        return result.toString()
    }
}
