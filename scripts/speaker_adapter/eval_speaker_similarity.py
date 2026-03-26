"""
Speaker embedding の cosine 類似度を評価するスクリプト。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 引数
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_jsonl", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--embedding_suffix", type=str, default=".spk.npy")
    return parser.parse_args()


def _load_pairs(path: Path) -> list[dict[str, str]]:
    """
    参照/生成のペア一覧を読み込む。

    Args:
        path (Path): JSONL のパス

    Returns:
        list[dict[str, str]]: ペア一覧
    """

    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip() == "":
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as ex:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {line.strip()!r}"
                ) from ex
    return pairs


def _load_embedding(audio_path: str, suffix: str) -> NDArray[Any]:
    """
    埋め込みファイルを読み込む。

    Args:
        audio_path (str): 音声パス
        suffix (str): 埋め込みサフィックス

    Returns:
        np.ndarray: 埋め込みベクトル
    """

    emb_path = f"{audio_path}{suffix}"
    return np.load(emb_path).reshape(-1)


def _cosine_similarity(a: NDArray[Any], b: NDArray[Any]) -> float:
    """
    cosine 類似度を計算する。

    Args:
        a (np.ndarray): ベクトル
        b (np.ndarray): ベクトル

    Returns:
        float: 類似度
    """

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    """
    cosine 類似度を集計し、統計量を出力する。
    """

    args = _parse_args()
    pairs = _load_pairs(Path(args.pairs_jsonl))

    scores = []
    details = []
    for pair in pairs:
        ref_audio = pair["ref_audio_path"]
        gen_audio = pair["gen_audio_path"]
        ref_emb = _load_embedding(ref_audio, args.embedding_suffix)
        gen_emb = _load_embedding(gen_audio, args.embedding_suffix)
        score = _cosine_similarity(ref_emb, gen_emb)
        scores.append(score)
        details.append(
            {
                "ref_audio_path": ref_audio,
                "gen_audio_path": gen_audio,
                "cosine": score,
                "speaker": pair.get("speaker", ""),
            }
        )

    if not scores:
        raise ValueError("No pairs found")

    scores_np = np.array(scores)
    summary = {
        "count": len(scores),
        "mean": float(np.mean(scores_np)),
        "median": float(np.median(scores_np)),
        "p10": float(np.percentile(scores_np, 10)),
        "p90": float(np.percentile(scores_np, 90)),
        "min": float(np.min(scores_np)),
        "max": float(np.max(scores_np)),
    }
    print("Cosine similarity summary")
    print(summary)

    if args.output_json != "":
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "details": details},
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
