"""
学習前データの整合性チェック用スクリプト。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.utils import load_filepaths_and_text


def _parse_args() -> argparse.Namespace:
    """
    CLI 引数を解析する。

    Returns:
        argparse.Namespace: 解析済み引数
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON used for training.",
    )
    parser.add_argument(
        "--max_errors",
        type=int,
        default=20,
        help="Maximum number of missing file errors to print per category.",
    )
    return parser.parse_args()


def _ensure_list_path(path_str: str) -> Path:
    """
    list ファイルのパスを解決する。

    Args:
        path_str (str): config から読み込んだ list のパス

    Returns:
        Path: 解決済みのパス
    """

    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _collect_entries(list_path: Path) -> list[list[str]]:
    """
    list ファイルを読み込む。

    Args:
        list_path (Path): list ファイルのパス

    Returns:
        list[list[str]]: list エントリ
    """

    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    return load_filepaths_and_text(list_path)


def _warn_missing(
    label: str,
    missing_paths: list[str],
    max_errors: int,
) -> None:
    """
    欠損ファイルのパスを出力する。

    Args:
        label (str): 欠損カテゴリ名
        missing_paths (list[str]): 欠損パス
        max_errors (int): 表示する最大件数
    """

    if not missing_paths:
        return
    logger.warning(f"{label}: {len(missing_paths)} missing files")
    for missing_path in missing_paths[:max_errors]:
        logger.warning(f"{label} missing: {missing_path}")


def _check_entries(
    entries: list[list[str]],
    use_external_speaker_embedding: bool,
    max_errors: int,
) -> tuple[set[str], int]:
    """
    音声 / style / BERT / external embedding の存在チェックを行う。

    Args:
        entries (list[list[str]]): list エントリ
        use_external_speaker_embedding (bool): external embedding の有無
        max_errors (int): 表示する最大件数

    Returns:
        tuple[set[str], int]: 話者 ID の集合と欠損数
    """

    missing_audio: list[str] = []
    missing_bert: list[str] = []
    missing_style: list[str] = []
    missing_external: list[str] = []
    speakers: set[str] = set()

    for fields in entries:
        if len(fields) < 2:
            raise ValueError("List entry must include audio path and speaker id")
        audio_path = fields[0]
        speaker = fields[1]
        speakers.add(speaker)

        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            missing_audio.append(audio_path)

        bert_path = str(Path(audio_path).with_suffix(".bert.pt"))
        if not Path(bert_path).exists():
            missing_bert.append(bert_path)

        style_path = f"{audio_path}.npy"
        if not Path(style_path).exists():
            missing_style.append(style_path)

        if use_external_speaker_embedding:
            external_path = f"{audio_path}.spk.npy"
            if not Path(external_path).exists():
                missing_external.append(external_path)

    _warn_missing("Audio", missing_audio, max_errors)
    _warn_missing("Style", missing_style, max_errors)
    _warn_missing("BERT", missing_bert, max_errors)
    if use_external_speaker_embedding:
        _warn_missing("External embedding", missing_external, max_errors)

    missing_total = (
        len(missing_audio)
        + len(missing_style)
        + len(missing_bert)
        + len(missing_external)
    )
    return speakers, missing_total


def main() -> None:
    """
    エントリポイント。
    """

    args = _parse_args()
    hps = HyperParameters.load_from_json(args.config)

    train_list_path = _ensure_list_path(hps.data.training_files)
    val_list_path = _ensure_list_path(hps.data.validation_files)

    logger.info(f"Checking training list: {train_list_path}")
    train_entries = _collect_entries(train_list_path)
    logger.info(f"Checking validation list: {val_list_path}")
    val_entries = _collect_entries(val_list_path)

    speakers_train, missing_train = _check_entries(
        train_entries,
        hps.data.use_external_speaker_embedding,
        args.max_errors,
    )
    speakers_val, missing_val = _check_entries(
        val_entries,
        hps.data.use_external_speaker_embedding,
        args.max_errors,
    )
    speakers_all = speakers_train | speakers_val

    spk2id_keys = set(hps.data.spk2id.keys())
    missing_speakers = speakers_all - spk2id_keys
    unused_speakers = spk2id_keys - speakers_all

    if missing_speakers:
        logger.warning(f"spk2id missing speakers: {len(missing_speakers)} entries")
        for speaker in sorted(missing_speakers)[: args.max_errors]:
            logger.warning(f"spk2id missing: {speaker}")
    if unused_speakers:
        logger.info(f"spk2id unused speakers: {len(unused_speakers)} entries")

    total_missing = missing_train + missing_val + len(missing_speakers)
    if total_missing > 0:
        raise SystemExit("Integrity check failed")

    logger.info("Integrity check passed")


if __name__ == "__main__":
    main()
