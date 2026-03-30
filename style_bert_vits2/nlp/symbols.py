from __future__ import annotations


# Punctuations
PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]

# Punctuations and special tokens
PUNCTUATION_SYMBOLS = PUNCTUATIONS + ["SP", "UNK"]

# Padding
PAD = "_"

# Nanairo 専用の絵文字モーラ
## すでに学習済みの non-JP-Extra / JP-Extra 系モデルとの語彙 ID の互換性を維持するため、
## 通常の SYMBOLS には追加せず、Nanairo 専用の追加モーラとして SYMBOLS の末尾に追加し、Nanairo 互換モデルでのみ用いる
## 含まれるのは日本語 BERT のトークナイザできちんと1文字として分割される（複数コードポイントに分割されない）絵文字のみ
## NOTE: 学習完了後は語彙 ID が確定するため、既存世代の定義順の変更や途中への挿入は禁止
## 万が一今後絵文字を追加する場合は、既存の語彙 ID に影響しないよう、新しい世代のリストを末尾に連結すること
## 各世代内は sorted() でコードポイント昇順を保証する
_NANAIRO_EMOJI_SYMBOLS_V1 = sorted(
    [
        "⏩",  # U+23E9
        "⏸",  # U+23F8
        "🌬",  # U+1F32C
        "🎵",  # U+1F3B5
        "🐢",  # U+1F422
        "👂",  # U+1F442
        "👅",  # U+1F445
        "👌",  # U+1F44C
        "💋",  # U+1F48B
        "💨",  # U+1F4A8
        "📞",  # U+1F4DE
        "📢",  # U+1F4E2
        "😆",  # U+1F606
        "😊",  # U+1F60A
        "😌",  # U+1F60C
        "😏",  # U+1F60F
        "😒",  # U+1F612
        "😖",  # U+1F616
        "😟",  # U+1F61F
        "😠",  # U+1F620
        "😪",  # U+1F62A
        "😭",  # U+1F62D
        "😮",  # U+1F62E
        "😰",  # U+1F630
        "😱",  # U+1F631
        "😲",  # U+1F632
        "🙄",  # U+1F644
        "🙏",  # U+1F64F
        "🤐",  # U+1F910
        "🤔",  # U+1F914
        "🤧",  # U+1F927
        "🤭",  # U+1F92D
        "🥤",  # U+1F964
        "🥱",  # U+1F971
        "🥴",  # U+1F974
        "🥵",  # U+1F975
        "🥺",  # U+1F97A
        "🫣",  # U+1FAE3
        "🫶",  # U+1FAF6
    ]
)
NANAIRO_EMOJI_SYMBOLS: list[str] = [
    *_NANAIRO_EMOJI_SYMBOLS_V1,
]
NANAIRO_EMOJI_SYMBOL_SET = set(NANAIRO_EMOJI_SYMBOLS)
_NANAIRO_EMOJI_SYMBOLS_SORTED = sorted(
    NANAIRO_EMOJI_SYMBOLS,
    key=len,
    reverse=True,
)

# Chinese symbols
ZH_SYMBOLS = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
NUM_ZH_TONES = 6

# Japanese
JP_SYMBOLS = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "v",
    "w",
    "y",
    "z",
    "zy",
]
NUM_JP_TONES = 2

# English
EN_SYMBOLS = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
NUM_EN_TONES = 4

# Combine all symbols
NORMAL_SYMBOLS = sorted(set(ZH_SYMBOLS + JP_SYMBOLS + EN_SYMBOLS))
SYMBOLS = [PAD] + NORMAL_SYMBOLS + PUNCTUATION_SYMBOLS
NANAIRO_SYMBOLS = SYMBOLS + NANAIRO_EMOJI_SYMBOLS

# Combine all tones
NUM_TONES = NUM_ZH_TONES + NUM_JP_TONES + NUM_EN_TONES

# Language maps
LANGUAGE_ID_MAP = {"ZH": 0, "JP": 1, "EN": 2}
NUM_LANGUAGES = len(LANGUAGE_ID_MAP.keys())

# Language tone start map
LANGUAGE_TONE_START_MAP = {
    "ZH": 0,
    "JP": NUM_ZH_TONES,
    "EN": NUM_ZH_TONES + NUM_JP_TONES,
}


def is_nanairo_emoji_symbol(symbol: str) -> bool:
    """
    `symbol` が、Nanairo 専用の絵文字モーラ 1 件と完全一致するかを返す。

    Args:
        symbol (str): 1 セグメント（通常は `split_text_by_nanairo_emoji_symbols` の要素）として判定する文字列

    Returns:
        bool: `NANAIRO_EMOJI_SYMBOLS` のいずれかと一致すれば True、それ以外は False
    """

    return symbol in NANAIRO_EMOJI_SYMBOL_SET


def contains_nanairo_emoji_symbols(text: str) -> bool:
    """
    `text` のどこかに、Nanairo 専用の絵文字モーラが部分文字列として現れるかを返す。

    Args:
        text (str): 走査する入力全文

    Returns:
        bool: いずれかの絵文字モーラが 1 件でも部分一致していれば True
    """

    return any(emoji_symbol in text for emoji_symbol in NANAIRO_EMOJI_SYMBOLS)


def split_text_by_nanairo_emoji_symbols(text: str) -> list[str]:
    """
    `text` を左から読み、現在位置が絵文字モーラの先頭ならその絵文字モーラ全体を 1 セグメントに切り出す。
    そうでなければ `str` の 1 要素（1 Unicode コードポイント）ずつ読み、連続した非絵文字は 1 セグメントにまとめる。
    絵文字モーラは長い定義から順に照合し、短い絵文字モーラだけが先にマッチして壊れることを防ぐ。

    Args:
        text (str): 分割する入力全文

    Returns:
        list[str]: 先頭から順のセグメント列。各要素は「絵文字モーラ 1 件」または「それ以外の文字の連なり」のどちらか
    """

    segments: list[str] = []
    current_chunk: list[str] = []
    current_index = 0
    while current_index < len(text):
        matched_emoji_symbol: str | None = None
        for emoji_symbol in _NANAIRO_EMOJI_SYMBOLS_SORTED:
            if text.startswith(emoji_symbol, current_index):
                matched_emoji_symbol = emoji_symbol
                break
        if matched_emoji_symbol is not None:
            if current_chunk:
                segments.append("".join(current_chunk))
                current_chunk = []
            segments.append(matched_emoji_symbol)
            current_index += len(matched_emoji_symbol)
            continue
        current_chunk.append(text[current_index])
        current_index += 1
    if current_chunk:
        segments.append("".join(current_chunk))
    return segments


def strip_nanairo_emoji_symbols(text: str) -> str:
    """
    `split_text_by_nanairo_emoji_symbols` と同じ規則で、`NANAIRO_EMOJI_SYMBOLS` に属するセグメントだけを取り除き、残りを連結する。

    Args:
        text (str): Nanairo 専用の絵文字モーラを削除したい入力全文

    Returns:
        str: Nanairo 専用の絵文字モーラを除いたあとの文字列
    """

    return "".join(
        segment
        for segment in split_text_by_nanairo_emoji_symbols(text)
        if is_nanairo_emoji_symbol(segment) is False
    )


if __name__ == "__main__":
    a = set(ZH_SYMBOLS)
    b = set(EN_SYMBOLS)
    print(sorted(a & b))
