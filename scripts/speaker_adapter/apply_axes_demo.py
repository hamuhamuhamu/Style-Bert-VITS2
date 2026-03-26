"""
speaker embedding と g 軸を使った音声生成のデモ。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io.wavfile import write as wav_write

from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.utils.paths import get_paths_config


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 引数
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Model folder name in model_assets/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint filename (relative to model_assets/{model-name}/)",
    )
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker_embedding", type=str, required=True)
    parser.add_argument("--axes_npz", type=str, required=True)
    parser.add_argument("--axes_json", type=str, required=True)
    parser.add_argument("--axis_name", type=str, required=True)
    parser.add_argument("--alphas", type=str, default="-2,-1,0,1,2")
    parser.add_argument("--language", type=str, default="JP")
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _load_axis(axes_npz: Path, axes_json: Path, axis_name: str) -> NDArray[np.float32]:
    """
    axes から対象の軸ベクトルを取得する。

    Args:
        axes_npz (Path): axes.npz のパス
        axes_json (Path): axes.json のパス
        axis_name (str): 軸名

    Returns:
        np.ndarray: 軸ベクトル
    """

    axes_data = np.load(axes_npz)
    names = axes_data["names"].tolist()
    if axis_name not in names:
        with axes_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        available = [axis["name"] for axis in meta.get("axes", [])]
        raise ValueError(f"Axis {axis_name} is not found. Available: {available}")
    axis_index = names.index(axis_name)
    return axes_data["axes"][axis_index]


def main() -> None:
    """
    指定した軸に沿って g を操作し、音声を出力する。
    """

    args = _parse_args()
    model_name: str = args.model
    model_dir = get_paths_config().assets_root / model_name
    model_path = model_dir / args.checkpoint
    config_path = model_dir / "config.json"
    style_vec_path = model_dir / "style_vectors.npy"
    output_dir = Path(args.output_dir)
    speaker_embedding_path = Path(args.speaker_embedding)

    axes_npz_path = Path(args.axes_npz)
    axes_json_path = Path(args.axes_json)

    for required_path in (
        model_path,
        config_path,
        style_vec_path,
        speaker_embedding_path,
        axes_npz_path,
        axes_json_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    axis = _load_axis(axes_npz_path, axes_json_path, args.axis_name)
    speaker_embedding = np.load(speaker_embedding_path)

    tts_model = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device=args.device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_values = [float(value) for value in args.alphas.split(",")]
    for alpha in alpha_values:
        g_adjust = axis * alpha
        sr, audio = tts_model.infer(
            text=args.text,
            language=Languages[args.language],
            speaker_id=args.speaker_id,
            speaker_embedding=speaker_embedding,
            g_adjust=g_adjust,
        )
        output_path = output_dir / f"axis_{args.axis_name}_alpha_{alpha:.2f}.wav"
        wav_write(output_path, sr, audio)


if __name__ == "__main__":
    main()
