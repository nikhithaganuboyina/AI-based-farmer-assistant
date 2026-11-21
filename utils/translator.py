from deep_translator import GoogleTranslator

_TRANSLATION_CACHE = {}

def translate_text(text, lang="en"):
    if not text or lang == "en":
        return text

    cache_key = (text, lang)
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    try:
        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        _TRANSLATION_CACHE[cache_key] = translated
        return translated
    except Exception as e:
        print("🔥 TRANSLATION ERROR:", e)
        return text  # fallback (no crash)
