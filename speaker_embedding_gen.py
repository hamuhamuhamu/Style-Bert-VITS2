"""
Anime speaker embedding を事前に生成するスクリプト。

デフォルトでは char + va の 384 次元ベクトルを生成する。
--variant オプションで char のみ (192 次元) や va のみ (192 次元) の生成も可能。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from anime_speaker_embedding import AnimeSpeakerEmbedding

from style_bert_vits2.logging import logger
from style_bert_vits2.utils.paths import TrainingModelPaths, add_model_argument
from training.utils import load_filepaths_and_text


# --variant で選択可能なバリアント名
VARIANT_CHOICES: list[str] = ["both", "char", "va"]


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 引数
    """

    parser = argparse.ArgumentParser()
    add_model_argument(parser)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--variant",
        type=str,
        choices=VARIANT_CHOICES,
        default="both",
        help="Embedding variant to generate. "
        '"both" concatenates char (192-dim) and va (192-dim) into 384-dim. '
        '"char" or "va" generates a single 192-dim embedding. (default: both)',
    )
    return parser.parse_args()


def get_audio_paths(list_path: Path) -> list[str]:
    """
    list ファイルから音声パスを取得する。

    Args:
        list_path (Path): list ファイルのパス

    Returns:
        list[str]: 音声ファイルのパス
    """

    if not list_path.exists():
        logger.warning(f"List file not found. Skipping: {list_path}")
        return []

    entries = load_filepaths_and_text(list_path)
    audio_paths = []
    for fields in entries:
        if len(fields) < 1:
            continue
        audio_paths.append(fields[0])
    return audio_paths


def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """
    L2 正規化: embedding のノルムを 1 に揃える。

    ノルムがばらつくと Adapter の学習が不安定になるため、事前に正規化して保存する。

    Args:
        embedding (np.ndarray): 正規化対象の embedding ベクトル

    Returns:
        np.ndarray: L2 正規化された embedding ベクトル
    """

    embedding_norm = np.linalg.norm(embedding)
    if embedding_norm > 0:
        embedding = embedding / embedding_norm
    return embedding


def main() -> None:
    """
    Anime speaker embedding を生成し、各 wav の隣に .spk.npy を保存する。

    デフォルトでは char + va の 384 次元ベクトルを生成する。
    --variant オプションで char のみ (192 次元) や va のみ (192 次元) の生成も可能。
    """

    args = _parse_args()
    model_folder_name: str = args.model
    variant: str = args.variant
    paths = TrainingModelPaths(model_folder_name)

    # train.list と val.list から音声パスを取得し、重複を除去する
    audio_paths_raw: list[str] = []
    audio_paths_raw.extend(get_audio_paths(paths.train_list_path))
    audio_paths_raw.extend(get_audio_paths(paths.val_list_path))
    audio_paths = list(dict.fromkeys(audio_paths_raw))

    if len(audio_paths) == 0:
        raise ValueError("No audio paths found in list files.")

    # "both" モードでは char と va の 2 つのモデルをロードする
    is_both_mode = variant == "both"
    model_char: AnimeSpeakerEmbedding | None = None
    model_va: AnimeSpeakerEmbedding | None = None
    model_single: AnimeSpeakerEmbedding | None = None
    if is_both_mode is True:
        logger.info(
            'Loading two anime speaker embedding models (char + va) for "both" mode.'
        )
        model_char = AnimeSpeakerEmbedding(device=args.device, variant="char")
        model_va = AnimeSpeakerEmbedding(device=args.device, variant="va")
    else:
        logger.info(f"Loading anime speaker embedding model. variant: {variant}")
        model_single = AnimeSpeakerEmbedding(device=args.device, variant=variant)

    for i, audio_path in enumerate(audio_paths):
        output_path = Path(f"{audio_path}.spk.npy")
        if args.skip_existing is True and output_path.exists() is True:
            continue

        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"Processing {i + 1}/{len(audio_paths)}: {audio_path}")

        try:
            if is_both_mode is True:
                assert model_char is not None and model_va is not None

                # char embedding と va embedding をそれぞれ取得し、個別に L2 正規化してから連結する
                ## 混合してから正規化すると情報が歪むため、各バリアントごとに正規化する
                char_embedding = model_char.get_embedding(audio_path)
                char_embedding = np.asarray(char_embedding, dtype=np.float32).reshape(
                    -1
                )
                char_embedding = _l2_normalize(char_embedding)

                va_embedding = model_va.get_embedding(audio_path)
                va_embedding = np.asarray(va_embedding, dtype=np.float32).reshape(-1)
                va_embedding = _l2_normalize(va_embedding)

                # char (192-dim) + va (192-dim) → 384-dim
                embedding = np.concatenate([char_embedding, va_embedding])
            else:
                assert model_single is not None

                embedding = model_single.get_embedding(audio_path)
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                # L2 正規化: embedding のノルムを 1 に揃える
                # ノルムがばらつくと Adapter の学習が不安定になるため、事前に正規化して保存する
                embedding = _l2_normalize(embedding)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embedding)
        except Exception as ex:
            logger.error(f"Failed to extract embedding: {audio_path}", exc_info=ex)
            continue

    logger.info("Embedding extraction finished.")


if __name__ == "__main__":
    main()
