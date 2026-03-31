"""
日本語 g2p 処理の回帰テスト。
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from preprocess_text import process_line
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import InvalidPhoneError, clean_text_with_given_phone_tone
from style_bert_vits2.nlp.japanese.g2p_utils import (
    kata_tone2phone_tone,
    phone_tone2kata_tone,
)
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker import (
    initialize_worker,
    terminate_worker,
)
from style_bert_vits2.nlp.nanairo_emoji import normalize_nanairo_emoji_text


NANAIRO_LAUGH_TEXT = "そう?🤭こうしてると,なんだか淳之介くんとデートしてるみたいだね?"
NANAIRO_PAUSE_TEXT = (
    "加藤の件はそこまでにしましょう.⏸小波さん,無理を承知でお願いしたいのですが."
)
NANAIRO_ANGER_TEXT = (
    "センパイってば,寧々先輩となんか2人だけのヒミツの話があったっぽいし😠"
)


# G2P テスト実行時のみ pyopenjtalk worker を初期化する
## import 時の副作用を避けつつ、既存の preprocess_text.py と同じ worker を使って回帰を検証する
@pytest.fixture(scope="module", autouse=True)
def pyopenjtalk_worker_fixture() -> Iterator[None]:
    """モジュール単位で pyopenjtalk worker を初期化し、終了時に明示終了する。"""

    initialize_worker()
    yield
    terminate_worker()


def _assert_phone_tone_word2ph_consistency(
    phones: list[str],
    tones: list[int],
    word2ph: list[int],
) -> None:
    """
    g2p 結果の基本整合性を検証する。

    Args:
        phones (list[str]): 音素列
        tones (list[int]): アクセント列
        word2ph (list[int]): 文字ごとの音素数
    """

    assert len(phones) == len(tones) == sum(word2ph)
    assert phones[0] == "_"
    assert phones[-1] == "_"


def test_clean_text_with_given_phone_tone_nanairo_keeps_emoji_mora() -> None:
    """Nanairo モードでは絵文字モーラを 1 記号として保持する。"""

    norm_text, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text=NANAIRO_LAUGH_TEXT,
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )

    assert "🤭" in norm_text
    assert "🤭" in phones
    assert tones[phones.index("🤭")] == 0
    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)
    assert len(word2ph) == len(norm_text) + 2


def test_clean_text_with_given_phone_tone_standard_strips_emoji_mora() -> None:
    """従来モデルでは絵文字モーラを text / phone / tone から除去する。"""

    norm_text, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text=NANAIRO_LAUGH_TEXT,
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=False,
        raise_yomi_error=False,
    )

    assert "🤭" not in norm_text
    assert "🤭" not in phones
    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)
    assert len(word2ph) == len(norm_text) + 2


def test_clean_text_with_given_phone_tone_given_phone_round_trip() -> None:
    """6 列入力相当の given_phone / given_tone でも絵文字モーラを保持できる。"""

    norm_text, phones, tones, _, _, _, _ = clean_text_with_given_phone_tone(
        text=NANAIRO_PAUSE_TEXT,
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )

    round_tripped = clean_text_with_given_phone_tone(
        text=norm_text,
        language=Languages.JP,
        given_phone=phones,
        given_tone=tones,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )
    standard_result = clean_text_with_given_phone_tone(
        text=norm_text,
        language=Languages.JP,
        given_phone=phones,
        given_tone=tones,
        use_jp_extra=True,
        use_nanairo=False,
        raise_yomi_error=False,
    )

    assert round_tripped[1] == phones
    assert round_tripped[2] == tones
    assert "⏸" in round_tripped[0]
    assert "⏸" in round_tripped[1]
    assert round_tripped[2][round_tripped[1].index("⏸")] == 0
    assert "⏸" not in standard_result[0]
    assert "⏸" not in standard_result[1]
    _assert_phone_tone_word2ph_consistency(
        round_tripped[1],
        round_tripped[2],
        round_tripped[3],
    )
    _assert_phone_tone_word2ph_consistency(
        standard_result[1],
        standard_result[2],
        standard_result[3],
    )


def test_clean_text_with_given_phone_tone_non_nanairo_validates_given_phone_tone_length() -> (
    None
):
    """Nanairo 以外でも given_phone と given_tone の長さ不整合は InvalidPhoneError とする。"""

    with pytest.raises(InvalidPhoneError):
        clean_text_with_given_phone_tone(
            text="あ💨",
            language=Languages.JP,
            given_phone=["_", "a", "💨", "_"],
            given_tone=[0, 0, 0, 0, 0],
            use_jp_extra=True,
            use_nanairo=False,
            raise_yomi_error=False,
        )


def test_clean_text_with_given_phone_tone_non_nanairo_reconciles_emoji_only_given_phone() -> (
    None
):
    """
    Nanairo 以外で given_phone が Nanairo 絵文字のみの場合でも、g2p 結果との整合のために
    word2ph を調整し、最終的な実音素列は生成側に合わせる。
    """

    norm_text, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="あ",
        language=Languages.JP,
        given_phone=["💨"],
        given_tone=[0],
        use_jp_extra=True,
        use_nanairo=False,
        raise_yomi_error=False,
    )

    assert norm_text == "あ"
    assert phones == ["_", "a", "_"]
    assert tones == [0, 0, 0]
    assert len(phones) == len(tones) == sum(word2ph)


def test_process_line_nanairo_preserves_emoji_and_standard_ignores_it(
    tmp_path: Path,
) -> None:
    """preprocess_text.process_line() でも use_nanairo=True であれば絵文字モーラを保持する。"""

    norm_text, phones, tones, _, _, _, _ = clean_text_with_given_phone_tone(
        text=NANAIRO_ANGER_TEXT,
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )
    # wavs_dir は process_line のパス正規化に使われるだけなので、任意のディレクトリでよい
    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    line = (
        f"{wavs_dir / 'sample.ogg'}|spk_0000|JP|"
        f"{norm_text}|{' '.join(phones)}|{' '.join(str(tone) for tone in tones)}"
    )

    nanairo_processed = process_line(
        line=line,
        wavs_dir=wavs_dir,
        use_jp_extra=True,
        use_nanairo=True,
        yomi_error="raise",
    ).strip()
    standard_processed = process_line(
        line=line,
        wavs_dir=wavs_dir,
        use_jp_extra=True,
        use_nanairo=False,
        yomi_error="raise",
    ).strip()

    nanairo_fields = nanairo_processed.split("|")
    standard_fields = standard_processed.split("|")

    assert len(nanairo_fields) == 7
    assert len(standard_fields) == 7
    assert "😠" in nanairo_fields[3]
    assert "😠" in nanairo_fields[4].split()
    assert "😠" not in standard_fields[3]
    assert "😠" not in standard_fields[4].split()

    nanairo_phones = nanairo_fields[4].split()
    nanairo_tones = [int(tone) for tone in nanairo_fields[5].split()]
    nanairo_word2ph = [int(phone_count) for phone_count in nanairo_fields[6].split()]
    emoji_index = nanairo_phones.index("😠")

    assert nanairo_tones[emoji_index] == 0
    _assert_phone_tone_word2ph_consistency(
        nanairo_phones,
        nanairo_tones,
        nanairo_word2ph,
    )

    standard_phones = standard_fields[4].split()
    standard_tones = [int(tone) for tone in standard_fields[5].split()]
    standard_word2ph = [int(phone_count) for phone_count in standard_fields[6].split()]

    _assert_phone_tone_word2ph_consistency(
        standard_phones,
        standard_tones,
        standard_word2ph,
    )


def test_g2p_basic_sentence() -> None:
    """基本的な文でも phones / tones / word2ph の整合性が保たれる。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="今日はいい天気ですね.",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_punctuation_marks() -> None:
    """句読点を含む文でも punctuation が保持される。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="マジすか...本当にダメなんですか?そんなに...?",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    assert "." in phones
    assert "?" in phones
    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_exclamation_and_question() -> None:
    """感嘆符と疑問符が混在しても整合性が保たれる。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="何やってるんですか!?早く来てよ!",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_long_text() -> None:
    """長文でも phones / tones / word2ph の整合性が崩れない。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text=(
            "いい機会じゃないか.天のもたらすみそぎの聖水だと思えよ."
            "澱と積もった不浄な我欲を残さず洗いながすがいい."
        ),
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_katakana_text() -> None:
    """カタカナ混じりの台詞でも整合性が保たれる。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="センパイってば,ヒミツの話があったっぽいし.",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_english_mixed() -> None:
    """英語混じりの短文でも InvalidPhoneError なく処理できる。"""

    _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
        text="OKです,ありがとう.",
        language=Languages.JP,
        use_jp_extra=True,
        raise_yomi_error=False,
    )

    _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_empty_and_minimal() -> None:
    """最小級の短文でも整合性が保たれる。"""

    for text in ("あ.", "うん."):
        _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
            text=text,
            language=Languages.JP,
            use_jp_extra=True,
            raise_yomi_error=False,
        )

        _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


def test_g2p_ellipsis_and_dash() -> None:
    """三点リーダー相当や間を置く文でも整合性が保たれる。"""

    for text in (
        "私みたいに...全力で幸せな人生を歩んでね.",
        "まったくもって.大したものです.",
    ):
        _, phones, tones, word2ph, _, _, _ = clean_text_with_given_phone_tone(
            text=text,
            language=Languages.JP,
            use_jp_extra=True,
            raise_yomi_error=False,
        )

        _assert_phone_tone_word2ph_consistency(phones, tones, word2ph)


# ============================================================
# phone_tone2kata_tone / kata_tone2phone_tone の Nanairo 絵文字パススルーテスト
# ============================================================


def test_phone_tone2kata_tone_passes_through_nanairo_emoji() -> None:
    """phone_tone2kata_tone が Nanairo 絵文字モーラを句読点と同様にそのままパススルーする。"""

    # 実運用相当: 「そう?🤭こうしてると」の g2p 出力を模擬
    _, phones, tones, _, _, _, _ = clean_text_with_given_phone_tone(
        text="そう?🤭こうしてると.",
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )
    phone_tone = list(zip(phones, tones))
    kata_tone = phone_tone2kata_tone(phone_tone)

    # 🤭 がカタカナ列の中にそのまま残っている
    kata_symbols = [k for k, _ in kata_tone]
    assert "🤭" in kata_symbols, f"Emoji 🤭 was dropped from kata_tone: {kata_tone}"

    # 往復変換で絵文字が保持される
    round_tripped = kata_tone2phone_tone(kata_tone)
    round_tripped_phones = [p for p, _ in round_tripped]
    assert "🤭" in round_tripped_phones, (
        f"Emoji 🤭 was dropped after round-trip: {round_tripped}"
    )


def test_phone_tone2kata_tone_handles_trailing_emoji() -> None:
    """文末の方式B 絵文字モーラも正しくパススルーされる。"""

    # 実運用相当: 「ありがとう😊」の g2p 出力を模擬
    _, phones, tones, _, _, _, _ = clean_text_with_given_phone_tone(
        text="ありがとう.😊",
        language=Languages.JP,
        use_jp_extra=True,
        use_nanairo=True,
        raise_yomi_error=False,
    )
    phone_tone = list(zip(phones, tones))
    kata_tone = phone_tone2kata_tone(phone_tone)

    kata_symbols = [k for k, _ in kata_tone]
    assert "😊" in kata_symbols, f"Trailing emoji 😊 was dropped: {kata_tone}"

    round_tripped = kata_tone2phone_tone(kata_tone)
    round_tripped_phones = [p for p, _ in round_tripped]
    assert "😊" in round_tripped_phones


def test_kata_tone2phone_tone_preserves_emoji_position() -> None:
    """kata_tone2phone_tone が絵文字の位置と tone を正しく保持する。"""

    # 手動で構築: カタカナ「ア」+ 絵文字 😠 + カタカナ「イ」
    kata_tone: list[tuple[str, int]] = [("ア", 1), ("😠", 0), ("イ", 0)]
    result = kata_tone2phone_tone(kata_tone)

    phones = [p for p, _ in result]
    tones = [t for _, t in result]

    # 先頭 _ + a + 😠 + i + _ の 5 トークン
    assert phones == ["_", "a", "😠", "i", "_"]
    assert tones[phones.index("😠")] == 0


# ============================================================
# normalize_nanairo_emoji_text のテスト
# ============================================================


def test_normalize_nanairo_emoji_text_removes_vs16_only_for_nanairo_emojis() -> None:
    """Nanairo 定義済み絵文字に付く VS16 のみ除去し、それ以外の記号の VS16 は保持する。"""

    # Nanairo 定義済み: ⏸️ (U+23F8 + U+FE0F) → ⏸ (U+23F8)
    assert normalize_nanairo_emoji_text("テスト⏸\ufe0fです") == "テスト⏸です"
    # Nanairo 定義済み: 🌬️ (U+1F32C + U+FE0F) → 🌬 (U+1F32C)
    assert normalize_nanairo_emoji_text("🌬\ufe0f") == "🌬"
    # Nanairo 非定義: ♾️ (U+267E + U+FE0F) → VS16 は保持される（normalize_text が ♾️ を認識するのを壊さない）
    assert normalize_nanairo_emoji_text("♾\ufe0f") == "♾\ufe0f"
    # Nanairo 非定義: 👍 + 肌色修飾子 → 保持される
    assert normalize_nanairo_emoji_text("👍\U0001f3fd") == "👍\U0001f3fd"


def test_normalize_nanairo_emoji_text_removes_skin_tone_modifiers() -> None:
    """肌色修飾子 (U+1F3FB-U+1F3FF) を除去する。"""

    # 🙏🏽 (U+1F64F + U+1F3FD) → 🙏 (U+1F64F)
    assert normalize_nanairo_emoji_text("🙏\U0001f3fd") == "🙏"
    # 👌🏻 (U+1F44C + U+1F3FB) → 👌 (U+1F44C)
    assert normalize_nanairo_emoji_text("テスト👌\U0001f3fbです") == "テスト👌です"


def test_normalize_nanairo_emoji_text_resolves_zwj_sequence() -> None:
    """ZWJ sequence を定義済み絵文字に分解する。"""

    # 😮‍💨 (U+1F62E + U+200D + U+1F4A8) → 💨 (U+1F4A8)
    assert normalize_nanairo_emoji_text("テスト😮\u200d💨です") == "テスト💨です"


def test_normalize_nanairo_emoji_text_applies_semantic_mapping() -> None:
    """意味的に近い絵文字を定義済み絵文字にマッピングする。"""

    # 😡 → 😠 (怒り系)
    assert normalize_nanairo_emoji_text("なんで😡") == "なんで😠"
    # 😂 → 😆 (笑い系)
    assert normalize_nanairo_emoji_text("ウケる😂") == "ウケる😆"
    # 😢 → 😭 (泣き系)
    assert normalize_nanairo_emoji_text("悲しい😢") == "悲しい😭"


def test_normalize_nanairo_emoji_text_preserves_defined_emojis() -> None:
    """Nanairo 定義済み絵文字はそのまま保持する。"""

    assert normalize_nanairo_emoji_text("テスト🤭です😠よ") == "テスト🤭です😠よ"


def test_normalize_nanairo_emoji_text_combined() -> None:
    """VS16 + 意味的マッピングが同時に適用される。"""

    # ❤️ (U+2764 + U+FE0F) → VS16除去 → ❤ → 意味的マッピング → 🫶
    assert normalize_nanairo_emoji_text("大好き❤\ufe0f") == "大好き🫶"
