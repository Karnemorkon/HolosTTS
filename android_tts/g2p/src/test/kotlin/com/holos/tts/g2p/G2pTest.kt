package com.holos.tts.g2p

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import java.io.InputStream

class G2pTest {

    @Test
    fun ` StressDictionary loads from gzip stream`() {
        val dict = StressDictionary
        val stream: InputStream = javaClass.getClassLoader()
            .getResourceAsStream("word_stress.txt.gz") ?: return
        dict.load(stream)
        assertTrue(dict.size > 0)
    }

    @Test
    fun ` ipa basic word`() {
        val result = UkrainianG2PEngine.ipa("Приві́т", checkAccent = true)
        assertNotNull(result)
        assertTrue(result.contains("ˈ"), "stress mark must be present: $result")
    }

    @Test
    fun ` ipa no accent monosyllabic ok`() {
        val result = UkrainianG2PEngine.ipa("Так", checkAccent = true)
        assertNotNull(result)
    }

    @Test
    fun ` text to phonemes`() {
        val dict = StressDictionary
        // lightweight inline dictionary for tests
        dict.load(javaClass.getClassLoader().getResourceAsStream("word_stress.txt.gz"))
        val phonemes = UkrainianTextToPhonemes.toPhonemes("Привіт, як справи?", dict)
        assertTrue(phonemes.isNotEmpty())
        assertTrue(phonemes.contains("|") || phonemes.contains("ˈ"), phonemes)
    }
}
