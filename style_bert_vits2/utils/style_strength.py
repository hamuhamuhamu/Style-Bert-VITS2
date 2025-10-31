"""スタイルベクトルの強度調整を行うユーティリティ関数群。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from style_bert_vits2.logging import logger


def _get_style_resource_paths(assets_root: Path, model_name: str) -> tuple[Path, Path]:
    """スタイルベクトルと設定ファイルのパスを返す。

    Args:
        assets_root (Path): モデル資産を格納しているルートディレクトリ。
        model_name (str): 調整対象となるモデルのディレクトリ名。

    Returns:
        tuple[Path, Path]: スタイルベクトルと config.json のパスを格納するタプル。
    """

    model_dir = assets_root / model_name
    style_vector_path = model_dir / "style_vectors.npy"
    config_path = model_dir / "config.json"
    return style_vector_path, config_path


def _create_style_vector_backup(style_vector_path: Path) -> Path:
    """style_vectors.npy を上書きする前にバックアップを作成する。

    Args:
        style_vector_path (Path): バックアップ対象となる style_vectors.npy のパス。

    Returns:
        Path: 作成したバックアップファイルのパス。
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = style_vector_path.with_name(
        f"{style_vector_path.name}.bak_{timestamp}",
    )
    shutil.copy(style_vector_path, backup_path)
    logger.info(f"Backup style_vectors to {backup_path}")
    return backup_path


def load_style_strength(
    model_name: str,
    assets_root: Path,
) -> tuple[list[tuple[str, int]], str, bool]:
    """スタイル強度調整 UI 向けにスタイル一覧を取得する。

    Args:
        model_name (str): 調整対象モデルの名前。
        assets_root (Path): モデル資産を格納しているルートディレクトリ。

    Returns:
        tuple[list[tuple[str, int]], str, bool]: (スタイル名, スタイル ID) のリスト、メッセージ、取得成功フラグ。
    """

    if model_name.strip() == "":
        return [], "モデル名を入力してください。", False

    style_vector_path, config_path = _get_style_resource_paths(assets_root, model_name)
    if not config_path.exists():
        return [], f"{config_path} が存在しません。", False
    if not style_vector_path.exists():
        return [], f"{style_vector_path} が存在しません。", False

    with config_path.open(encoding="utf-8") as config_file:
        config_dict = json.load(config_file)

    style2id = config_dict.get("data", {}).get("style2id", {})
    if not style2id:
        return [], "config.json に style2id が含まれていません。", False

    sorted_styles = sorted(style2id.items(), key=lambda item: item[1])
    style_entries: list[tuple[str, int]] = []
    for style_name, style_id in sorted_styles:
        style_entries.append((style_name, style_id))

    info_message = "各スタイルのスタイル強度を徐々に上げていき、耳で聞いて「これ以上上げると音声が不自然になる」と感じた値を入力してください。"
    return style_entries, info_message, True


def apply_style_strength(
    model_name: str,
    assets_root: Path,
    rows: list[list[Any]] | None,
) -> tuple[bool, str]:
    """入力テーブルに基づいてスタイルベクトルを再スケーリングする。

    Args:
        model_name (str): 調整対象モデルの名前。
        assets_root (Path): モデル資産を格納しているルートディレクトリ。
        rows (list[list[Any]] | None): UI から渡されるスタイル名と重みのテーブル。

    Returns:
        tuple[bool, str]: 処理の成否とユーザー向けメッセージ。
    """

    if model_name.strip() == "":
        return False, "モデル名を入力してください。"

    style_vector_path, config_path = _get_style_resource_paths(assets_root, model_name)
    if not config_path.exists():
        return False, f"{config_path} が存在しません。"
    if not style_vector_path.exists():
        return False, f"{style_vector_path} が存在しません。"

    with config_path.open(encoding="utf-8") as config_file:
        config_dict = json.load(config_file)

    style2id = config_dict.get("data", {}).get("style2id", {})
    if not style2id:
        return False, "config.json に style2id が含まれていません。"

    style_vectors = np.load(style_vector_path)
    mean_vector = style_vectors[0]
    updated_styles: list[str] = []
    skipped_styles: list[str] = []

    for row in rows or []:
        try:
            style_name = str(row[0]).strip()
        except (IndexError, TypeError):
            continue
        if style_name == "":
            continue
        if style_name not in style2id:
            skipped_styles.append(style_name)
            continue
        style_id = style2id[style_name]
        if style_id == 0:
            # Neutral (平均スタイル) は差分がゼロなので調整対象から外す
            skipped_styles.append(style_name)
            continue
        try:
            current_weight = float(row[1])
            target_weight = float(row[2])
        except (ValueError, TypeError, IndexError):
            return False, f"{style_name} の数値が不正です。数値を入力してください。"
        if current_weight <= 0 or target_weight <= 0:
            return False, f"{style_name} の重みは 0 より大きな値を指定してください。"
        # current_weight / target_weight にすることで、測定した最大値を共通の目標値に線形マッピングする
        gain = current_weight / target_weight
        style_diff = style_vectors[style_id] - mean_vector
        style_vectors[style_id] = mean_vector + style_diff * gain
        updated_styles.append(f"{style_name} (gain={gain:.3f})")

    if not updated_styles:
        return False, "更新対象のスタイルがありませんでした。"

    _create_style_vector_backup(style_vector_path)
    np.save(style_vector_path, style_vectors)
    logger.info("Style vector strength updated successfully")

    skipped_message = ""
    if skipped_styles:
        skipped_unique = sorted(set(skipped_styles))
        skipped_message = "\n無視したスタイル: " + ", ".join(skipped_unique)

    result_message = (
        "スタイルベクトルを更新しました。\n"
        + "\n".join(updated_styles)
        + skipped_message
    )
    return True, result_message
