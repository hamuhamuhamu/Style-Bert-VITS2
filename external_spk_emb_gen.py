"""
Anime speaker embedding を事前に生成するスクリプト。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from anime_speaker_embedding import AnimeSpeakerEmbedding

from style_bert_vits2.logging import logger
from style_bert_vits2.models.utils import load_filepaths_and_text


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 引数
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def _iter_audio_paths(list_path: Path) -> list[str]:
    """
    train.list から音声パスを取得する。

    Args:
        list_path (Path): list ファイルのパス

    Returns:
        list[str]: 音声ファイルのパス
    """

    entries = load_filepaths_and_text(list_path)
    audio_paths = []
    for fields in entries:
        if len(fields) < 1:
            continue
        audio_paths.append(fields[0])
    return audio_paths


def main() -> None:
    """
    Anime speaker embedding を生成し、各 wav の隣に .spk.npy を保存する。
    """

    args = _parse_args()
    list_path = Path(args.list)
    audio_paths = _iter_audio_paths(list_path)

    if len(audio_paths) == 0:
        raise ValueError("No audio paths found in list file.")

    logger.info("Loading anime speaker embedding model.")
    model = AnimeSpeakerEmbedding(device=args.device, variant="char")

    for audio_path in audio_paths:
        output_path = Path(f"{audio_path}.spk.npy")
        if args.skip_existing and output_path.exists():
            continue

        embedding = model.get_embedding(audio_path)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding)

    logger.info("Embedding extraction finished.")


if __name__ == "__main__":
    main()
