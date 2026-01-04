"""
Adapter 後の g と音響特徴量から軸ベクトルを抽出するスクリプト。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyworld
from numpy.typing import NDArray

from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.utils.paths import TrainingModelPaths, add_model_argument


def _parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 引数
    """

    parser = argparse.ArgumentParser()
    add_model_argument(parser)
    parser.add_argument("--g_npz", type=str, required=True)
    parser.add_argument("--meta_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bootstrap", type=int, default=50)
    return parser.parse_args()


def _load_metadata(path: Path) -> list[dict[str, object]]:
    """
    JSONL のメタ情報を読み込む。

    Args:
        path (Path): JSONL のパス

    Returns:
        list[dict[str, object]]: メタ情報
    """

    items: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                continue
            items.append(json.loads(line))
    return items


def _compute_spectral_tilt(magnitude: NDArray[Any], freqs: NDArray[Any]) -> float:
    """
    スペクトルの低域/高域の差分から簡易 tilt を計算する。

    Args:
        magnitude (np.ndarray): 振幅スペクトル (freq, frame)
        freqs (np.ndarray): 周波数ビン

    Returns:
        float: スペクトル tilt
    """

    low_mask = freqs <= 2000
    high_mask = freqs >= 4000
    if not np.any(low_mask) or not np.any(high_mask):
        return 0.0
    low_mean = np.mean(magnitude[low_mask])
    high_mean = np.mean(magnitude[high_mask])
    return float(np.log10(high_mean + 1e-8) - np.log10(low_mean + 1e-8))


def _extract_features(
    audio_path: Path, sampling_rate: int, phone_count: int
) -> dict[str, float]:
    """
    音声から簡易音響特徴量を抽出する。

    Args:
        audio_path (Path): 音声ファイルのパス
        sampling_rate (int): サンプリングレート
        phone_count (int): 音素数

    Returns:
        dict[str, float]: 特徴量
    """

    wav, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
    if wav.size == 0:
        raise ValueError("Audio file is empty")

    f0, t = pyworld.dio(wav.astype(np.float64), int(sr))
    f0 = pyworld.stonemask(wav.astype(np.float64), f0, t, int(sr))
    voiced = f0 > 0
    if np.any(voiced):
        f0_mean = float(np.mean(f0[voiced]))
        f0_median = float(np.median(f0[voiced]))
    else:
        f0_mean = 0.0
        f0_median = 0.0

    energy_rms = float(np.sqrt(np.mean(wav**2)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr)))

    stft = librosa.stft(wav, n_fft=1024, hop_length=256)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    spectral_tilt = _compute_spectral_tilt(magnitude, freqs)

    duration_sec = float(len(wav) / sr)
    if duration_sec > 0:
        speaking_rate = float(phone_count / duration_sec)
    else:
        speaking_rate = 0.0

    return {
        "f0_mean": f0_mean,
        "f0_median": f0_median,
        "energy_rms": energy_rms,
        "spectral_centroid": spectral_centroid,
        "spectral_tilt": spectral_tilt,
        "speaking_rate": speaking_rate,
    }


def _ridge_regression(
    x: NDArray[np.float32], y: NDArray[np.float32], alpha: float
) -> NDArray[np.float32]:
    """
    Ridge 回帰の係数を計算する。

    Args:
        x (np.ndarray): 特徴行列 (N, D)
        y (np.ndarray): 目的変数 (N,)
        alpha (float): Ridge 係数

    Returns:
        np.ndarray: 係数ベクトル (D,)
    """

    xtx = x.T @ x
    reg = alpha * np.eye(xtx.shape[0], dtype=xtx.dtype)
    return np.linalg.solve(xtx + reg, x.T @ y)


def _r2_score(y_true: NDArray[np.float32], y_pred: NDArray[np.float32]) -> float:
    """
    R^2 を計算する。

    Args:
        y_true (np.ndarray): 正解値
        y_pred (np.ndarray): 予測値

    Returns:
        float: R^2
    """

    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def main() -> None:
    """
    g と音響特徴量から軸ベクトルを抽出し、npz と json に保存する。
    """

    args = _parse_args()
    model_folder_name: str = args.model
    paths = TrainingModelPaths(model_folder_name)
    config_path = paths.config_path
    g_npz = Path(args.g_npz)
    meta_jsonl = Path(args.meta_jsonl)
    output_dir = Path(args.output_dir)

    hps = HyperParameters.load_from_json(config_path)
    g_matrix = np.load(g_npz)["g"]
    meta = _load_metadata(meta_jsonl)

    if g_matrix.shape[0] != len(meta):
        raise ValueError("g matrix size does not match metadata")

    feature_names = [
        "f0_mean",
        "f0_median",
        "energy_rms",
        "spectral_centroid",
        "spectral_tilt",
        "speaking_rate",
    ]
    features = []
    for item in meta:
        audio_path = Path(str(item["audio_path"]))
        phone_count_raw = item.get("phone_count", 0)
        if not isinstance(phone_count_raw, (int, float)):
            raise ValueError(
                f"phone_count must be int or float, got {type(phone_count_raw)}"
            )
        phone_count = int(phone_count_raw)
        features.append(
            _extract_features(audio_path, hps.data.sampling_rate, phone_count)
        )

    y_matrix = np.stack(
        [[feature[name] for name in feature_names] for feature in features], axis=0
    ).astype(np.float32)

    y_mean = np.mean(y_matrix, axis=0)
    y_std = np.std(y_matrix, axis=0)
    y_std[y_std == 0] = 1.0
    y_norm = (y_matrix - y_mean) / y_std

    rng = np.random.default_rng(args.seed)
    indices = np.arange(g_matrix.shape[0])
    rng.shuffle(indices)
    val_count = int(len(indices) * args.val_ratio)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    axes = []
    axes_meta = []
    for feature_idx, feature_name in enumerate(feature_names):
        y_train = y_norm[train_indices, feature_idx]
        y_val = y_norm[val_indices, feature_idx]
        g_train = g_matrix[train_indices]
        g_val = g_matrix[val_indices]

        w = _ridge_regression(g_train, y_train, args.alpha)
        y_pred = g_val @ w
        r2 = _r2_score(y_val, y_pred)

        norm = np.linalg.norm(w)
        if norm > 0:
            d = w / norm
        else:
            d = w

        cosine_scores = []
        for _ in range(args.bootstrap):
            boot_indices = rng.choice(
                train_indices, size=len(train_indices), replace=True
            )
            w_boot = _ridge_regression(
                g_matrix[boot_indices], y_norm[boot_indices, feature_idx], args.alpha
            )
            denom = np.linalg.norm(w_boot) * np.linalg.norm(w)
            if denom > 0:
                cosine_scores.append(float(np.dot(w_boot, w) / denom))
        if cosine_scores:
            cosine_mean = float(np.mean(cosine_scores))
            cosine_std = float(np.std(cosine_scores))
        else:
            cosine_mean = 0.0
            cosine_std = 0.0

        axes.append(d.astype(np.float32))
        axes_meta.append(
            {
                "name": feature_name,
                "r2": r2,
                "alpha": args.alpha,
                "cosine_mean": cosine_mean,
                "cosine_std": cosine_std,
                "y_mean": float(y_mean[feature_idx]),
                "y_std": float(y_std[feature_idx]),
            }
        )

    axes_array = np.stack(axes, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "axes.npz", axes=axes_array, names=np.array(feature_names))
    with (output_dir / "axes.json").open("w", encoding="utf-8") as f:
        json.dump({"axes": axes_meta}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
