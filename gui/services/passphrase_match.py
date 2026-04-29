import unicodedata


MATCH_MODE_EXACT = "normalized_exact"
MATCH_MODE_CONTAINS = "normalized_contains"
MATCH_MODE_CHOICES = (MATCH_MODE_EXACT, MATCH_MODE_CONTAINS)


def normalize_passphrase_text(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    chars = []
    for ch in text:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            continue
        chars.append(ch.lower())
    return "".join(chars)


def match_passphrase(recognized_text, expected_text, mode=MATCH_MODE_EXACT):
    normalized_recognized = normalize_passphrase_text(recognized_text)
    normalized_expected = normalize_passphrase_text(expected_text)

    if not normalized_expected:
        return {
            "matched": False,
            "normalized_recognized": normalized_recognized,
            "normalized_expected": normalized_expected,
            "mode": mode,
        }

    if mode == MATCH_MODE_CONTAINS:
        matched = normalized_expected in normalized_recognized
    else:
        matched = normalized_recognized == normalized_expected

    return {
        "matched": bool(matched),
        "normalized_recognized": normalized_recognized,
        "normalized_expected": normalized_expected,
        "mode": mode,
    }
