"""
Nanairo 絵文字モーラの正規化・判定・分割ユーティリティ。

symbols.py で定義された `NANAIRO_EMOJI_SYMBOLS` を参照し、入力テキスト中の絵文字を
Nanairo 定義済み絵文字に正規化したり、テキストを絵文字とそれ以外のセグメントに分割する。
"""

from __future__ import annotations

import re

from style_bert_vits2.nlp.symbols import NANAIRO_EMOJI_SYMBOLS


# ============================================================
# 内部キャッシュ
# ============================================================

_NANAIRO_EMOJI_SYMBOL_SET: frozenset[str] = frozenset(NANAIRO_EMOJI_SYMBOLS)

# 絵文字モーラを長い定義から順に照合するためのソート済みリスト
# 現状は全て 1 コードポイントなので長さは同一だが、将来の拡張に備える
_NANAIRO_EMOJI_SYMBOLS_SORTED: list[str] = sorted(
    NANAIRO_EMOJI_SYMBOLS,
    key=len,
    reverse=True,
)

# ============================================================
# 正規化マッピングテーブル
# ============================================================

# Nanairo 定義済み絵文字の直後に付く Unicode 修飾子を除去するための正規表現パターン
# VS16 (U+FE0F) と肌色修飾子 (U+1F3FB-U+1F3FF) が対象
# Nanairo 定義済み絵文字以外の記号に付く VS16 には影響しない（♾️ → ♾ のような誤変換を防ぐ）
_NANAIRO_EMOJI_MODIFIER_PATTERN: re.Pattern[str] = re.compile(
    "("
    + "|".join(re.escape(emoji) for emoji in _NANAIRO_EMOJI_SYMBOLS_SORTED)
    + ")[\ufe0f\U0001f3fb-\U0001f3ff]+"
)

# ZWJ (Zero Width Joiner) で結合された絵文字シーケンスを、対応する Nanairo 定義済み絵文字に変換する
_NANAIRO_EMOJI_ZWJ_MAP: dict[str, str] = {
    "😮\u200d💨": "💨",  # Face Exhaling (ZWJ sequence) → 💨 (Dash Symbol)
}

# Nanairo に定義されていないが意味的に近い絵文字を、定義済み絵文字にマッピングする
# 主観的なマッピングのため、明確に同義と言える絵文字のみに限定し、将来の追加・修正が容易な構造にする
_NANAIRO_EMOJI_SEMANTIC_MAP_BASE: dict[str, str] = {
    # 怒り系 → 😠
    "😡": "😠",  # Pouting Face → Angry Face
    "😤": "😠",  # Face with Steam from Nose → Angry Face
    # 笑い系 → 😆
    "😂": "😆",  # Face with Tears of Joy → Grinning Squinting Face
    "🤣": "😆",  # Rolling on the Floor Laughing → Grinning Squinting Face
    # 泣き系 → 😭
    "😢": "😭",  # Crying Face → Loudly Crying Face
    "🥲": "😭",  # Smiling Face with Tear → Loudly Crying Face
    # 不安・恐怖系 → 😟
    "😨": "😟",  # Fearful Face → Worried Face
    "😥": "😟",  # Sad but Relieved Face → Worried Face
    # 驚き系 → 😲
    "😧": "😲",  # Anguished Face → Astonished Face
    "😦": "😲",  # Frowning Face with Open Mouth → Astonished Face
    # 苦痛系 → 😖
    "😣": "😖",  # Persevering Face → Confounded Face
    "😫": "😖",  # Tired Face → Confounded Face
    # パニック系 → 😰
    "😓": "😰",  # Downcast Face with Sweat → Anxious Face with Sweat
    # 喜び系 → 😊
    "🥰": "😊",  # Smiling Face with Hearts → Smiling Face with Smiling Eyes
    "☺": "😊",  # Smiling Face (text) → Smiling Face with Smiling Eyes
    # 愛情系 → 🫶
    "❤": "🫶",  # Red Heart → Heart Hands
    "💕": "🫶",  # Two Hearts → Heart Hands
    # 考え中・疑問系 → 🤔
    "🤨": "🤔",  # Face with Raised Eyebrow → Thinking Face
    # からかい系 → 😏
    "😜": "😏",  # Winking Face with Tongue → Smirking Face
    "😼": "😏",  # Cat with Wry Smile → Smirking Face
    # 嘔吐系 → 🥴
    "🤢": "🥴",  # Nauseated Face → Face with Uneven Eyes
    "🤮": "🥴",  # Vomiting Face → Face with Uneven Eyes
}
# ソース側に VS16 (U+FE0F) が付いたバリアント（例: ❤️）もマッチするよう展開する
_NANAIRO_EMOJI_SEMANTIC_MAP: dict[str, str] = {}
for _src, _tgt in _NANAIRO_EMOJI_SEMANTIC_MAP_BASE.items():
    _NANAIRO_EMOJI_SEMANTIC_MAP[_src] = _tgt
    _NANAIRO_EMOJI_SEMANTIC_MAP[_src + "\ufe0f"] = _tgt
# VS16 付きバリアント（2文字）が基底（1文字）より先にマッチするよう、長いキーから順に処理する
_NANAIRO_EMOJI_SEMANTIC_KEYS_SORTED: list[str] = sorted(
    _NANAIRO_EMOJI_SEMANTIC_MAP.keys(),
    key=len,
    reverse=True,
)

# ZWJ マップのキーを長い順にソートしたリスト（長い ZWJ sequence を先にマッチさせる）
# ZWJ sequence 内に VS16 が含まれるバリアント（例: 😮‍💨 の各構成要素に VS16 が付く）も登録する
_NANAIRO_EMOJI_ZWJ_MAP_EXPANDED: dict[str, str] = {}
for _zwj_seq, _zwj_tgt in _NANAIRO_EMOJI_ZWJ_MAP.items():
    _NANAIRO_EMOJI_ZWJ_MAP_EXPANDED[_zwj_seq] = _zwj_tgt
    # ZWJ sequence の各構成要素に VS16 が付いたバリアントも登録
    _vs16_variant = _zwj_seq.replace("\u200d", "\ufe0f\u200d")
    if _vs16_variant != _zwj_seq:
        _NANAIRO_EMOJI_ZWJ_MAP_EXPANDED[_vs16_variant] = _zwj_tgt
_NANAIRO_EMOJI_ZWJ_KEYS_SORTED: list[str] = sorted(
    _NANAIRO_EMOJI_ZWJ_MAP_EXPANDED.keys(),
    key=len,
    reverse=True,
)


# ============================================================
# 公開関数
# ============================================================


def normalize_nanairo_emoji_text(text: str) -> str:
    """
    入力テキスト中の絵文字を Nanairo 定義済み絵文字に正規化する。

    以下の3段階で処理する。
    1. Unicode 修飾子の除去: VS16 (U+FE0F) と肌色修飾子 (U+1F3FB-U+1F3FF) を除去し、基底絵文字に戻す
    2. ZWJ sequence の分解: ZWJ (U+200D) で結合された絵文字を定義済み絵文字にマッピングする
    3. 意味的マッピング: Nanairo に定義されていないが意味的に近い絵文字を定義済み絵文字に変換する

    Args:
        text (str): 正規化する入力テキスト

    Returns:
        str: Nanairo 定義済み絵文字に正規化されたテキスト
    """

    # レイヤー 1: Nanairo 定義済み絵文字に付く Unicode 修飾子のみ除去
    # VS16 (U+FE0F) と肌色修飾子 (U+1F3FB-U+1F3FF) を対象とするが、
    # Nanairo 定義済み絵文字以外の記号に付く VS16 には影響しない（♾️ → ♾ のような誤変換を防ぐ）
    normalized_text = _NANAIRO_EMOJI_MODIFIER_PATTERN.sub(r"\1", text)

    # レイヤー 2: ZWJ sequence の分解
    # 長いキーを先にマッチさせることで、部分一致の誤りを防ぐ
    # VS16 付きバリアント（例: 😮️‍💨）も展開済みテーブルでカバーする
    for zwj_sequence in _NANAIRO_EMOJI_ZWJ_KEYS_SORTED:
        if zwj_sequence in normalized_text:
            normalized_text = normalized_text.replace(
                zwj_sequence, _NANAIRO_EMOJI_ZWJ_MAP_EXPANDED[zwj_sequence]
            )

    # レイヤー 3: 意味的マッピング
    # VS16 付きバリアント（例: ❤️）が基底（❤）より先にマッチするよう、長いキーから順に処理する
    for source_emoji in _NANAIRO_EMOJI_SEMANTIC_KEYS_SORTED:
        if source_emoji in normalized_text:
            normalized_text = normalized_text.replace(
                source_emoji, _NANAIRO_EMOJI_SEMANTIC_MAP[source_emoji]
            )

    return normalized_text


def is_nanairo_emoji_symbol(symbol: str) -> bool:
    """
    `symbol` が、Nanairo 専用の絵文字モーラ 1 件と完全一致するかを返す。

    Args:
        symbol (str): 1 セグメント（通常は `split_text_by_nanairo_emoji_symbols` の要素）として判定する文字列

    Returns:
        bool: `NANAIRO_EMOJI_SYMBOLS` のいずれかと一致すれば True、それ以外は False
    """

    return symbol in _NANAIRO_EMOJI_SYMBOL_SET


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
