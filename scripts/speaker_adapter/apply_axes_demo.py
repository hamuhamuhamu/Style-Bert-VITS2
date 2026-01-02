"""
外部 speaker embedding と g 軸を使った音声生成のデモ。
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


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 引数
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--style_vec", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--external_embedding", type=str, required=True)
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
    model_path = Path(args.model)
    config_path = Path(args.config)
    style_vec_path = Path(args.style_vec)
    output_dir = Path(args.output_dir)

    axis = _load_axis(Path(args.axes_npz), Path(args.axes_json), args.axis_name)
    external_embedding = np.load(args.external_embedding)

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
            external_speaker_embedding=external_embedding,
            g_adjust=g_adjust,
        )
        output_path = output_dir / f"axis_{args.axis_name}_alpha_{alpha:.2f}.wav"
        wav_write(output_path, sr, audio)


if __name__ == "__main__":
    main()
