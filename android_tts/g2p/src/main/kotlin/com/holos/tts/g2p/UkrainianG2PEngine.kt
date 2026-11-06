package com.holos.tts.g2p

import kotlin.text.Regex
import kotlin.text.RegexOption.IGNORE_CASE

/**
 * Порт української фонетичної транскрипції з [ipa-uk](https://github.com/patriotyk/ipa-uk)
 * (алгоритм Вікіпедії: https://en.wiktionary.org/wiki/Module:uk-pronunciation).
 *
 * Приймає текст із наголосою ( Combining Acute Accent U+0301 ) і повертає IPA-транскрипцію,
 * як вимагає модель HolosTTS.
 */
object UkrainianG2PEngine {

    private const val ACUTE = "\u0301"
    private const val GRAVE = "\u0300"

    private val vowelNoI = "aɛɪuɔɐoʊe"
    private val vowel = vowelNoI + "i"
    private val vowelC = "[$vowel]"

    private val consonantNoW = "bdzʒɡɦmnlrpftskxʃj"
    private val consonantNoWC = "[$consonantNoW]"
    private val consonant = consonantNoW + "ʋβ̞wʍ"
    private val consonantC = "[$consonant]"

    private val palatalizable = "tdsznlrbpʋfɡmkɦxʃʒ"
    private val palatalizableC = "[$palatalizable]"

    private val voicedObstruent = "[bdzʒɡɦ]"

    /** Гучні приголосні → голосні (асиміляція). */
    private val voicing = mapOf(
        "p" to "b",
        "f" to "v",
        "t" to "d",
        "tʲ" to "dʲ",
        "s" to "z",
        "sʲ" to "zʲ",
        "ʃ" to "ʒ",
        "k" to "ɡ",
        "x" to "ɦ",
        "t͡s" to "d͡z",
        "t͡sʲ" to "d͡zʲ",
        "t͡ʃ" to "d͡ʒ",
        "ʃt͡ʃ" to "ʒd͡ʒ",
    )

    /** Допустимі початкові групи приголосних (syllable onsets). */
    private val permSylOnset = setOf(
        "spr", "str", "skr", "spl", "skl", "sp", "st", "sk", "sf", "sx",
        "pr", "br", "tr", "dr", "kr", "gr", "ɦr", "fr", "xr",
        "pl", "bl", "kl", "gl", "ɦl", "fl", "xl",
    )

    /** Серії замін трьох знаків (довжини 3). */
    private val tripleReplacements = mapOf(
        "дзь" to "d͡zʲ",
        "тьс" to "t͡sʲː",
    )

    /** Серії замін двох знаків (довжини 2). */
    private val doubleReplacements = mapOf(
        "дж" to "d͡ʒ",
        "дз" to "d͡z",
        "дс" to "d͡zs",
        "дш" to "d͡ʒʃ",
        "дч" to "d͡ʒt͡ʃ",
        "дц" to "d͡zt͡s",
        "тс" to "t͡s",
        "тш" to "t͡ʃʃ",
        "тч" to "t͡ʃː",
        "тц" to "t͡sː",
    )

    /** Одиночні символи кирилиці → IPA. */
    private val singleReplacements = mapOf(
        "а" to "a",
        "б" to "b",
        "в" to "ʋ",
        "г" to "ɦ",
        "ґ" to "ɡ",
        "д" to "d",
        "е" to "ɛ",
        "є" to "jɛ",
        "ж" to "ʒ",
        "з" to "z",
        "и" to "ɪ",
        "і" to "i",
        "ї" to "ji",
        "й" to "j",
        "к" to "k",
        "л" to "l",
        "м" to "m",
        "н" to "n",
        "о" to "ɔ",
        "п" to "p",
        "р" to "r",
        "с" to "s",
        "т" to "t",
        "у" to "u",
        "ф" to "f",
        "х" to "x",
        "ц" to "t͡s",
        "ч" to "t͡ʃ",
        "ш" to "ʃ",
        "щ" to "ʃt͡ʃ",
        "ь" to "ʲ",
        "ю" to "ju",
        "я" to "ja",
        "’" to "j",
        ACUTE to "ˈ",
        GRAVE to "ˌ",
    )

    /** Ортографічні заміни (упорядковані, як у оригіналі). */
    private val orthographicReplacements = listOf(
        "нтськ" to "ньськ",
        "стськ" to "ськ",
        "нтст" to "нст",
        "стч" to "шч",
        "стд" to "зд",
        "стс" to "сː",
        "#зш" to "#шː",
        "зш" to "жш",
        "#зч" to "#шч",
        "зч" to "жч",
    )

    /** Довгі приголосні (ортографічно двогрошені). */
    private val geminateRegexes = listOf(
        Regex("([бвгґд])\\1") to "\\1ː",
        Regex("([^д]+)жж") to "\\1жː",
        Regex("([^д]+)зз") to "\\1зː",
        Regex("([йклмнпрстфхцчшщ])\\1") to "\\1ː",
    )
    private val geminateSpecial = listOf(
        "дждж" to "джː",
        "дздз" to "дзː",
    )

    /** Максимальна довжина суфіксу IPA-маркування (міягкість + довжина). */
    private val ipaSuffixRegex = Regex("[ʲː]*")

    /**
     * Транскрибує одне слово (або речення) українською кирилицею із наголосою у IPA.
     * Порт функції `ipa` з ipa-uk.
     *
     * @param text текст із наголосою (Combining Acute Accent U+0301 над голосною)
     * @param checkAccent якщо true — викидає AccentIsMissing для слів >1 голосної без наголосу
     */
    @Throws(AccentIsMissing::class)
    fun ipa(text: String, checkAccent: Boolean = false): String {
        var phonetic = decomposeGrave(text).lowercase()
        phonetic = phonetic.replace(Regex("\\s+"), " ")

        if (checkAccent && !phonetic.contains(ACUTE) && !phonetic.contains(GRAVE)) {
            val vowelCount = phonetic.count { it in "аеєиіїоуюя" }
            if (vowelCount > 1) {
                throw AccentIsMissing(
                    "The provided text is missing an accent (and has more than one syllable). " +
                            "Set checkAccent=false to disable that check"
                )
            }
        }

        return collectPhonetics(phonetic)
    }

    /** Створює список транскрипцій для слів речення. */
    private fun collectPhonetics(text: String): String {
        val pronuns = mutableListOf<String>()
        for (rawWord in text.split(Regex("[\\s\\-]+"))) {
            if (rawWord.isBlank()) continue
            pronuns.add(transcribeWord("#" + rawWord + "#"))
        }
        return pronuns.joinToString(" ")
    }

    /**
     * Транскрибує одне слово (з маркерами межі слова #).
     * Порт тіла циклу for phonetic з ipa-uk.
     */
    private fun transcribeWord(phonetic: String): String {
        var p = phonetic

        // --- orthographic cluster simplifications ---
        for ((regex, repl) in orthographicReplacements) {
            p = p.replace(regex, repl)
        }

        // --- long consonants (geminates) ---
        for ((regex, repl) in geminateRegexes) {
            p = p.replace(regex, repl)
        }
        for ((key, repl) in geminateSpecial) {
            p = p.replace(key, repl)
        }

        // --- remap apostrophe to '%' so it doesn't conflict with IPA stress mark ---
        p = p.replace("'", "%")

        // --- replace multi-character sequences (triples then doubles then singles) ---
        for ((key, repl) in tripleReplacements) {
            p = p.replace(key, repl, ignoreCase = false)
        }
        for ((key, repl) in doubleReplacements) {
            p = p.replace(key, repl, ignoreCase = false)
        }
        for ((key, repl) in singleReplacements) {
            p = p.replace(key, repl, ignoreCase = false)
        }

        // --- move stress mark before vowel ---
        p = p.replace(Regex("([aɛiɪuɔ])([ˈˌ])"), "\\2\\1")

        // --- auto-stress monosyllabic words ---
        val vowelCount = Regex("[$vowelNoI]").findAll(p).count()
        if (vowelCount == 1) {
            p = p.replace(Regex("([aɛiɪuɔ])"), "⁀\\1")
        }

        // --- palatalizable consonants before /i/ or /j/ become palatalized ---
        p = p.replace(Regex("($palatalizableC)([ː]?)([ˈˌ⁀]?)i"), "\\1ʲ\\2\\3i")
        p = p.replace(Regex("($palatalizableC)([ː]?)j"), "\\1ʲ\\2")

        // --- eliminate garbage [ʲːj] sequences ---
        p = p.replace("ʲːj", "ʲː")

        // --- consonant simplification ст + ц' → [с'ц'] ---
        p = p.replace(Regex("st͡sʲ([ː]?)"), "sʲt͡sʲ")

        // --- voicing assimilation ---
        for ((voiceless, voiced) in voicing) {
            p = p.replace(Regex(voiceless + "(" + voicedObstruent + "+)"), voiced + "$1")
        }

        // --- softening of first consonant before soft second ---
        p = p.replace(Regex("([tdszn])(.)ʲ"), "\\1ʲ\\2ʲ")
        p = p.replace(Regex("([tdsznl])t͡sʲ"), "\\1ʲt͡sʲ")
        p = p.replace(Regex("([tdsznl])d͡zʲ"), "\\1ʲd͡zʲ")
        p = p.replace(Regex("t͡s(.)ʲ"), "t͡sʲ\\1ʲ")
        p = p.replace(Regex("d͡z(.)ʲ"), "d͡zʲ\\1ʲ")
        p = p.replace("d͡zt͡sʲ", "d͡zʲt͡sʲ")
        p = p.replace("t͡sd͡zʲ", "t͡sʲd͡zʲ")

        // --- hushing assimilation ---
        p = p.replace("ʒt͡sʲ", "zʲt͡sʲ")
        p = p.replace("t͡ʃt͡sʲ", "t͡sʲː")
        p = p.replace("ʃt͡sʲ", "sʲt͡sʲ")
        p = p.replace("ʃsʲ", "sʲː")

        // --- hissing before hushing ---
        p = p.replace("zʒ", "ʒː")
        p = p.replace("sʃ", "ʃː")
        p = p.replace("zt͡ʃ", "ʒt͡ʃ")
        p = p.replace("zd͡ʒ", "ʒd͡ʒ")
        p = p.replace("t͡ʒ", "d͡ʒ")
        p = p.replace("t͡z", "d͡z")

        // --- cleanup excessive palatalization CʲCʲCʲ → CCʲCʲ ---
        p = p.replace(Regex("([^aɛiɪuɔ]+)ʲ([^aɛiɪuɔ]+)ʲ([^aɛiɪuɔ]+)ʲ"), "\\1\\2ʲ\\3ʲ")

        // --- unstressed vowel allophones ---
        p = p.replace(Regex("([^ˈˌ⁀])a"), "\\1ɐ")
        p = p.replace(Regex("([^ˈˌ⁀])u"), "\\1ʊ")
        p = p.replace(Regex("ɔ([bdzʒɡɦmnlrpftskxʲʃ͡]+)([ˈˌ⁀][uiʊ])"), "o$1$2")
        p = p.replace(Regex("([^ˈˌ⁀])[ɛ]"), "\\1e")

        // --- remove auto-added monosyllabic stress marker ---
        p = p.replace("⁀", "")

        // --- /ʋ/ allophones ---
        p = p.replace(Regex("($vowelC)ʋ([$consonantNoW#])"), "\\1u̯\\2")
        p = p.replace(Regex("ʋ([ˈˌ]?[ɔuoʊbdzʒɡɦmnlr])"), "w\\1")
        p = p.replace(Regex("ʋ([pftskxʃ])"), "ʍ\\1")

        // --- /j/ allophones ---
        p = p.replace(Regex("($vowelC)j([$consonantNoW#])"), "\\1i̯\\2")
        p = p.replace(Regex("#j($consonantNoWC)"), "#i̯\\1")

        // --- remove old orthographic apostrophe ---
        p = p.replace("%", "")

        // --- stress mark in correct place (1): before final consonant of cluster ---
        p = p.replace(Regex("([^#" + vowel + "]?[ʲː]*)([ˈˌ])"), "\\2\\1")
        // --- (2): over affricate tie bar ---
        p = p.replace(Regex("([^#" + vowel + "]͡)([ˈˌ])"), "\\2\\1")

        // --- (3): over permanent onset clusters ---
        p = onsetReplacement(p)

        // --- (4) ---
        p = p.replace(Regex("([^#" + vowel + "]͡)([ˈˌ])(.ʲ?j)"), "\\2\\1\\3")
        p = p.replace(Regex("([^#" + vowel + "]͡)([ˈˌ])(.ʲ?)"), "\\1\\3\\2")
        // --- (5): move back over remaining consonants at word start ---
        p = p.replace(Regex("#([^#" + vowel + "]+)([ˈˌ])"), "#\\2\\1")
        // --- (6): move back over u̯ / i̯ at word start ---
        p = p.replace(Regex("#([ui]̯)([ˈˌ])"), "#\\2\\1")

        p = p.replace("ʲ?ːʲ", "ʲː")

        // --- dark [ɫ] for non-palatal /l/ ---
        p = p.replace(Regex("l([^ʲ])"), "ɫ\\1")

        return p
    }

    /**
     * Переставляє знак наголосу над останнім приголосним кластера (permanent onset).
     * Порт `onset_replacement` з ipa-uk.
     */
    private fun onsetReplacement(p: String): String {
        val pattern = Regex("(.)(ʲ?)($consonantC)(ʲ?)([ˈˌ])($consonantC)")
        var result = p
        // String.replace with transform
        result = pattern.replace(result) { match ->
            val a = match.groupValues[1]
            val aj = match.groupValues[2]
            val b = match.groupValues[3]
            val bj = match.groupValues[4]
            val stress = match.groupValues[5]
            val c = match.groupValues[6]
            val clusterKey = a + b + c
            if (clusterKey in permSylOnset) {
                stress + a + aj + b + bj + c
            } else if ((b + c in permSylOnset) || c == "j") {
                a + aj + stress + b + bj + c
            } else {
                a + aj + b + bj + stress + c
            }
        }
        return result
    }

    /**
     * Розбирає попередньо згорнені кириличні символи з grave наголосом.
     */
    private fun decomposeGrave(word: String): String {
        val graveMap = mapOf(
            "ѐ" to "е$GRAVE",
            "Ѐ" to "Е$GRAVE",
            "ѝ" to "и$GRAVE",
            "Ѝ" to "И$GRAVE",
        )
        var result = word
        for ((from, to) in graveMap) {
            result = result.replace(from, to)
        }
        return result
    }

    /** Виняток: відсутній наголос у багатьослоговому слові. */
    class AccentIsMissing(message: String) : IllegalArgumentException(message)
}
