"""
External speaker embedding から g をダンプするスクリプト。
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
from style_bert_vits2.models.utils import load_filepaths_and_text


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 引数
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--list", type=str, required=True)
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


def _load_external_embedding(audio_path: str) -> NDArray[Any]:
    """
    外部 speaker embedding を読み込む。

    Args:
        audio_path (str): 音声ファイルのパス
    Returns:
        np.ndarray: speaker embedding
    """

    embedding_path = f"{audio_path}.spk.npy"
    return np.load(embedding_path)


def main() -> None:
    """
    外部 speaker embedding から g を生成し、npz と JSONL に保存する。
    """

    args = _parse_args()
    config_path = Path(args.config)
    model_path = Path(args.model)
    list_path = Path(args.list)
    output_npz = Path(args.output_npz)
    output_meta = Path(args.output_meta)

    hps = HyperParameters.load_from_json(config_path)
    if hps.model.use_external_speaker_adapter is not True:
        raise ValueError("use_external_speaker_adapter must be true.")

    net_g = get_net_g(
        model_path=str(model_path),
        version=hps.version,
        device=args.device,
        hps=hps,
    )
    net_g.eval()
    if getattr(net_g, "ext_spk_adapter", None) is None:
        raise ValueError("External speaker adapter is not initialized.")

    entries = _load_entries(list_path)
    g_list: list[NDArray[np.float32]] = []

    output_meta.parent.mkdir(parents=True, exist_ok=True)
    net_g_nanairo = cast(SynthesizerTrnNanairo, net_g)
    assert net_g_nanairo.ext_spk_adapter is not None, (
        "External speaker adapter is not initialized"
    )
    with output_meta.open("w", encoding="utf-8") as meta_file, torch.inference_mode():
        for idx, entry in enumerate(entries):
            embedding = _load_external_embedding(entry["audio_path"])
            embedding_tensor = torch.from_numpy(embedding).float().to(args.device)
            if embedding_tensor.dim() == 1:
                embedding_tensor = embedding_tensor.unsqueeze(0)
            g_tensor = net_g_nanairo.ext_spk_adapter(embedding_tensor)
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
