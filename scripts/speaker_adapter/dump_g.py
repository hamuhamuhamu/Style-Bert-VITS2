"""
学習用 .spk.npy (anime-speaker-embedding) から g をダンプするスクリプト。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g
from style_bert_vits2.models.models_nanairo import (
    SynthesizerTrn as SynthesizerTrnNanairo,
)
from style_bert_vits2.utils.paths import TrainingModelPaths, add_model_argument
from training.utils import load_filepaths_and_text


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 引数
    """

    parser = argparse.ArgumentParser()
    add_model_argument(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint filename (relative to Data/{model-name}/)",
    )
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--output_meta", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _load_entries(list_path: Path) -> list[dict[str, str]]:
    """
    学習用リストを読み込み、必要なメタ情報を抽出する。

    Args:
        list_path (Path): train.list のパス

    Returns:
        list[dict[str, str]]: エントリの配列
    """

    entries = []
    for fields in load_filepaths_and_text(list_path):
        if len(fields) < 7:
            raise ValueError("train.list format must have at least 7 fields.")
        entries.append(
            {
                "audio_path": fields[0],
                "speaker": fields[1],
                "language": fields[2],
                "text": fields[3],
                "phones": fields[4],
            }
        )
    return entries


def _load_speaker_embedding(audio_path: str) -> NDArray[Any]:
    """
    speaker embedding (.spk.npy) を読み込む。

    Args:
        audio_path (str): 音声ファイルのパス

    Returns:
        NDArray[Any]: speaker embedding
    """

    embedding_path = f"{audio_path}.spk.npy"
    return np.load(embedding_path)


def main() -> None:
    """
    speaker embedding (.spk.npy) から g を生成し、npz と JSONL に保存する。
    """

    args = _parse_args()
    model_folder_name: str = args.model
    paths = TrainingModelPaths(model_folder_name)
    config_path = paths.config_path
    model_path = paths.models_dir / args.checkpoint
    list_path = paths.train_list_path
    output_npz = Path(args.output_npz)
    output_meta = Path(args.output_meta)

    hps = HyperParameters.load_from_json(config_path)
    if hps.model.use_speaker_adapter is not True:
        raise ValueError("use_speaker_adapter must be true.")

    net_g = get_net_g(
        model_path=str(model_path),
        version=hps.version,
        device=args.device,
        hps=hps,
    )
    net_g.eval()

    entries = _load_entries(list_path)
    g_list: list[NDArray[np.float32]] = []

    output_meta.parent.mkdir(parents=True, exist_ok=True)
    net_g_nanairo = cast(SynthesizerTrnNanairo, net_g)
    # Pyright が nn.Module 属性の Optional を絞り込めるよう、直接参照してから検証する
    speaker_control_encoder = net_g_nanairo.speaker_control_encoder
    speaker_adapter = net_g_nanairo.speaker_adapter
    if speaker_control_encoder is None or speaker_adapter is None:
        raise ValueError("Speaker control encoder and adapter are not initialized.")
    with output_meta.open("w", encoding="utf-8") as meta_file, torch.inference_mode():
        for idx, entry in enumerate(entries):
            audio_path = Path(entry["audio_path"])
            if audio_path.is_absolute() is False:
                audio_path = paths.wavs_dir / audio_path
            embedding = _load_speaker_embedding(str(audio_path))
            embedding_tensor = torch.from_numpy(embedding).float().to(args.device)
            if embedding_tensor.dim() == 1:
                embedding_tensor = embedding_tensor.unsqueeze(0)
            # g_neutral にゲート付き SpeakerAdapter の出力を加算して g を求める
            ctrl = speaker_control_encoder(embedding_tensor)
            g_tensor = net_g_nanairo.g_neutral + speaker_adapter(ctrl)
            g = g_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            g_list.append(g)

            phone_count = len(entry["phones"].split())
            meta = {
                "index": idx,
                "audio_path": entry["audio_path"],
                "speaker": entry["speaker"],
                "language": entry["language"],
                "text": entry["text"],
                "phone_count": phone_count,
            }
            meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")

    g_matrix = np.stack(g_list, axis=0).astype(np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, g=g_matrix)


if __name__ == "__main__":
    main()
