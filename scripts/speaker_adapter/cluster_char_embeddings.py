"""
話者ラベルを自動クラスタリングするスクリプト。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from anime_speaker_embedding import AnimeSpeakerEmbedding
from numpy.typing import NDArray

from style_bert_vits2.logging import logger
from style_bert_vits2.models.utils import load_filepaths_and_text
from style_bert_vits2.utils.paths import TrainingModelPaths, add_model_argument


DEFAULT_TARGET_CLUSTER_SIZE = 20
DEFAULT_KMEANS_ITERS = 20
DEFAULT_KMEANS_RESTARTS = 3
DEFAULT_RANDOM_SEED = 1234
DEFAULT_MAX_CLUSTER_CANDIDATES = 12
SINGLE_SPEAKER_MEAN_SIM_THRESHOLD = 0.78
SINGLE_SPEAKER_P10_SIM_THRESHOLD = 0.68
SINGLE_SPEAKER_PAIRWISE_MEDIAN_THRESHOLD = 0.60
SINGLE_SPEAKER_PAIRWISE_P10_THRESHOLD = 0.45
PAIRWISE_SAMPLE_SIZE = 20000
ELBOW_RELATIVE_IMPROVEMENT_THRESHOLD = 0.15
ELBOW_MIN_ABS_IMPROVEMENT = 0.01
SILHOUETTE_SPLIT_THRESHOLD = 0.10
MIN_CLUSTER_RATIO_FOR_SPLIT = 0.12


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
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _load_list_entries(list_path: Path) -> list[list[str]]:
    """
    list ファイルを読み込む。

    Args:
        list_path (Path): list ファイルのパス

    Returns:
        list[list[str]]: list エントリ
    """

    return load_filepaths_and_text(list_path)


def _load_all_list_entries(dataset_dir: Path) -> dict[Path, list[list[str]]]:
    """
    dataset_dir 配下の list ファイルを読み込む。

    Args:
        dataset_dir (Path): データセットディレクトリ

    Returns:
        dict[Path, list[list[str]]]: list ファイルとエントリの辞書
    """

    list_files = sorted(
        list_path for list_path in dataset_dir.glob("*.list*") if list_path.is_file()
    )
    if not list_files:
        raise ValueError("No .list files found in dataset_dir.")

    return {list_path: _load_list_entries(list_path) for list_path in list_files}


def _iter_audio_paths(entries: list[list[str]]) -> list[str]:
    """
    list エントリから音声パスを抽出する。

    Args:
        entries (list[list[str]]): list エントリ

    Returns:
        list[str]: 音声パス
    """

    audio_paths = []
    for fields in entries:
        if len(fields) < 1:
            continue
        audio_paths.append(fields[0])
    return audio_paths


def _load_or_extract_embedding(
    audio_path: Path,
    model: AnimeSpeakerEmbedding,
    skip_existing: bool,
) -> NDArray[np.float32]:
    """
    embedding を読み込むか、新規生成する。

    Args:
        audio_path (Path): 音声ファイルのパス
        model (AnimeSpeakerEmbedding): embedding 生成モデル
        skip_existing (bool): 既存ファイルを再計算しないか

    Returns:
        np.ndarray: embedding
    """

    embedding_path = Path(f"{audio_path}.spk.npy")
    if embedding_path.exists():
        return np.load(embedding_path).astype(np.float32).reshape(-1)

    if skip_existing:
        raise ValueError("Embedding file is missing while skip_existing is enabled.")

    embedding = model.get_embedding(str(audio_path))
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embedding)
    return embedding


def _normalize_embeddings(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    埋め込みを正規化する。

    Args:
        embeddings (np.ndarray): 埋め込み行列

    Returns:
        np.ndarray: 正規化済み埋め込み
    """

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _kmeans_cosine(
    embeddings: NDArray[np.float32],
    k: int,
    iters: int,
    seed: int,
) -> tuple[NDArray[np.int_], NDArray[np.float32]]:
    """
    cosine 距離相当の k-means を実行する。

    Args:
        embeddings (np.ndarray): 埋め込み行列 (N, D)
        k (int): クラスタ数
        iters (int): 反復回数
        seed (int): 乱数シード

    Returns:
        tuple[np.ndarray, np.ndarray]: クラスタ割当 (N,), セントロイド (K, D)
    """

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(embeddings), size=k, replace=False)
    centroids = embeddings[indices]

    assignments: NDArray[np.int_] | None = None
    for _ in range(iters):
        scores = embeddings @ centroids.T
        assignments = np.argmax(scores, axis=1)
        new_centroids = []
        for cluster_id in range(k):
            members = embeddings[assignments == cluster_id]
            if len(members) == 0:
                new_centroids.append(centroids[cluster_id])
                continue
            centroid = np.mean(members, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            new_centroids.append(centroid)
        centroids = np.stack(new_centroids, axis=0)

    if assignments is None:
        raise RuntimeError("k-means did not produce assignments.")
    return assignments, centroids


def _decide_cluster_count(sample_count: int) -> int:
    """
    サンプル数からクラスタ数を決定する。

    Args:
        sample_count (int): サンプル数

    Returns:
        int: クラスタ数
    """

    if sample_count <= 0:
        return 1
    return max(1, round(sample_count / DEFAULT_TARGET_CLUSTER_SIZE))


def _compute_similarity_stats(
    embeddings: NDArray[np.float32],
    seed: int,
) -> dict[str, float]:
    """
    単一話者判定用の類似度統計量を計算する。

    Args:
        embeddings (np.ndarray): 正規化済み埋め込み行列 (N, D)
        seed (int): 乱数シード

    Returns:
        dict[str, float]: 統計量
    """

    centroid = np.mean(embeddings, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    centroid_sims = embeddings @ centroid
    centroid_sims_sorted = np.sort(centroid_sims)
    centroid_mean = float(np.mean(centroid_sims))
    centroid_p10 = float(np.percentile(centroid_sims_sorted, 10))

    rng = np.random.default_rng(seed)
    sample_count = len(embeddings)
    max_pairs = min(PAIRWISE_SAMPLE_SIZE, sample_count * (sample_count - 1) // 2)
    if max_pairs <= 0:
        pairwise_median = 0.0
        pairwise_p10 = 0.0
    else:
        if sample_count < 2:
            pairwise_median = 0.0
            pairwise_p10 = 0.0
        else:
            pair_indices = rng.integers(
                0,
                sample_count,
                size=(max_pairs, 2),
            )
            pair_indices = pair_indices[pair_indices[:, 0] != pair_indices[:, 1]]
            if pair_indices.size == 0:
                pairwise_median = 0.0
                pairwise_p10 = 0.0
            else:
                pair_sims = np.sum(
                    embeddings[pair_indices[:, 0]] * embeddings[pair_indices[:, 1]],
                    axis=1,
                )
                pairwise_median = float(np.median(pair_sims))
                pairwise_p10 = float(np.percentile(pair_sims, 10))

    return {
        "centroid_mean": centroid_mean,
        "centroid_p10": centroid_p10,
        "pairwise_median": pairwise_median,
        "pairwise_p10": pairwise_p10,
    }


def _is_single_speaker(sim_stats: dict[str, float]) -> bool:
    """
    類似度統計量から単一話者か判定する。

    Args:
        sim_stats (dict[str, float]): 類似度統計量

    Returns:
        bool: 単一話者であれば True
    """

    if (
        sim_stats["centroid_mean"] >= SINGLE_SPEAKER_MEAN_SIM_THRESHOLD
        and sim_stats["centroid_p10"] >= SINGLE_SPEAKER_P10_SIM_THRESHOLD
        and sim_stats["pairwise_median"] >= SINGLE_SPEAKER_PAIRWISE_MEDIAN_THRESHOLD
        and sim_stats["pairwise_p10"] >= SINGLE_SPEAKER_PAIRWISE_P10_THRESHOLD
    ):
        return True
    return False


def _compute_intra_cluster_distance(
    embeddings: NDArray[np.float32],
    assignments: NDArray[np.int_],
    centroids: NDArray[np.float32],
) -> float:
    """
    1 - cosine 距離の平均値を計算する。

    Args:
        embeddings (np.ndarray): 正規化済み埋め込み行列 (N, D)
        assignments (np.ndarray): クラスタ割当 (N,)
        centroids (np.ndarray): セントロイド (K, D)

    Returns:
        float: 平均距離
    """

    sims = np.sum(embeddings * centroids[assignments], axis=1)
    return float(np.mean(1.0 - sims))


def _compute_silhouette_score(
    embeddings: NDArray[np.float32],
    assignments: NDArray[np.int_],
) -> float:
    """
    cosine 距離の silhouette score を計算する。

    Args:
        embeddings (np.ndarray): 正規化済み埋め込み行列 (N, D)
        assignments (np.ndarray): クラスタ割当 (N,)

    Returns:
        float: silhouette score
    """

    unique_labels = np.unique(assignments)
    if unique_labels.size < 2 or len(embeddings) < 3:
        return 0.0

    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.warning("sklearn is not available. Skipping silhouette score.")
        return 0.0

    return float(silhouette_score(embeddings, assignments, metric="cosine"))


def _compute_min_cluster_ratio(
    assignments: NDArray[np.int_],
) -> float:
    """
    最小クラスタの比率を計算する。

    Args:
        assignments (np.ndarray): クラスタ割当 (N,)

    Returns:
        float: 最小クラスタ比率
    """

    total_count = len(assignments)
    if total_count <= 0:
        return 0.0
    counts = np.bincount(assignments.astype(np.int_))
    if counts.size == 0:
        return 0.0
    return float(np.min(counts) / total_count)


def _kmeans_cosine_multi_start(
    embeddings: NDArray[np.float32],
    k: int,
    iters: int,
    seed: int,
    restarts: int,
) -> tuple[NDArray[np.int_], NDArray[np.float32], float]:
    """
    複数初期値で k-means を実行し最良結果を返す。

    Args:
        embeddings (np.ndarray): 埋め込み行列 (N, D)
        k (int): クラスタ数
        iters (int): 反復回数
        seed (int): 乱数シード
        restarts (int): 再初期化回数

    Returns:
        tuple[np.ndarray, np.ndarray, float]: 割当 (N,), セントロイド (K, D), 平均距離
    """

    best_assignments: NDArray[np.int_] | None = None
    best_centroids: NDArray[np.float32] | None = None
    best_distance = float("inf")
    for offset in range(restarts):
        assignments, centroids = _kmeans_cosine(
            embeddings,
            k=k,
            iters=iters,
            seed=seed + offset,
        )
        distance = _compute_intra_cluster_distance(embeddings, assignments, centroids)
        if distance < best_distance:
            best_distance = distance
            best_assignments = assignments
            best_centroids = centroids

    if best_assignments is None or best_centroids is None:
        raise RuntimeError("k-means did not produce assignments.")

    return best_assignments, best_centroids, best_distance


def _decide_cluster_count_elbow(
    embeddings: NDArray[np.float32],
    max_candidates: int,
    seed: int,
    iters: int,
    restarts: int,
) -> tuple[int, dict[int, float]]:
    """
    エルボー法でクラスタ数を決定する。

    Args:
        embeddings (np.ndarray): 正規化済み埋め込み行列 (N, D)
        max_candidates (int): 最大クラスタ候補数
        seed (int): 乱数シード
        iters (int): 反復回数
        restarts (int): 再初期化回数

    Returns:
        tuple[int, dict[int, float]]: 決定クラスタ数, k ごとの平均距離
    """

    max_k = min(max_candidates, len(embeddings))
    if max_k <= 1:
        return 1, {1: 0.0}

    distances: dict[int, float] = {}
    best_k = 1
    prev_distance = None
    for k in range(1, max_k + 1):
        _assignments, _centroids, distance = _kmeans_cosine_multi_start(
            embeddings,
            k=k,
            iters=iters,
            seed=seed,
            restarts=restarts,
        )
        distances[k] = distance
        if prev_distance is None:
            prev_distance = distance
            continue

        improvement = prev_distance - distance
        relative_improvement = improvement / max(prev_distance, 1e-6)
        if (
            improvement < ELBOW_MIN_ABS_IMPROVEMENT
            or relative_improvement < ELBOW_RELATIVE_IMPROVEMENT_THRESHOLD
        ):
            break
        best_k = k
        prev_distance = distance

    return best_k, distances


def _summarize_cluster_counts(label_counts: dict[int, int]) -> dict[str, float]:
    """
    クラスタサイズの統計量を計算する。

    Args:
        label_counts (dict[int, int]): クラスタごとの件数

    Returns:
        dict[str, float]: 統計量
    """

    counts = np.array(list(label_counts.values()), dtype=np.float32)
    if counts.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "min": float(np.min(counts)),
        "max": float(np.max(counts)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
    }


def _plot_cluster_histogram(
    label_counts: dict[int, int],
    output_path: Path,
) -> None:
    """
    クラスタサイズの分布を可視化する。

    Args:
        label_counts (dict[int, int]): クラスタごとの件数
        output_path (Path): 出力パス
    """

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available. Skipping plot output.")
        return

    matplotlib.use("Agg")

    cluster_ids = sorted(label_counts.keys())
    counts = [label_counts[cluster_id] for cluster_id in cluster_ids]
    if not counts:
        logger.warning("No cluster counts available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(counts)), counts, color="#4c78a8")
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster Index (sorted)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_cluster_scatter_tsne(
    embeddings: NDArray[np.float32],
    assignments: NDArray[np.int_],
    output_path: Path,
    seed: int,
) -> None:
    """
    t-SNE で埋め込みの散布図を出力する。

    Args:
        embeddings (np.ndarray): 正規化済み埋め込み行列 (N, D)
        assignments (np.ndarray): クラスタ割当 (N,)
        output_path (Path): 出力パス
        seed (int): 乱数シード
    """

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("matplotlib or sklearn is not available. Skipping plot output.")
        return

    matplotlib.use("Agg")

    reducer = TSNE(n_components=2, random_state=seed, metric="cosine")
    embeddings_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(assignments.tolist()))
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(unique_labels):
        mask = assignments == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            color=cmap(idx % cmap.N),
            label=f"Cluster {label}",
            alpha=0.7,
            s=18,
        )

    ax.set_title("t-SNE of Speaker Embeddings by Cluster.")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(markerscale=1.2, fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _update_config_json(
    config_path: Path,
    cluster_count: int,
    dry_run: bool,
) -> None:
    """
    config.json の spk2id / n_speakers を更新する。

    Args:
        config_path (Path): config.json のパス
        cluster_count (int): クラスタ数
        dry_run (bool): 書き込みを行わない場合 True
    """

    if not config_path.exists():
        raise ValueError("config.json not found in dataset_dir.")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    spk2id = {f"char_{idx:04d}": idx for idx in range(cluster_count)}
    config.setdefault("data", {})
    config["data"]["spk2id"] = spk2id
    config["data"]["n_speakers"] = int(cluster_count)

    if dry_run:
        logger.info(
            f"[dry-run] config.json would be updated with n_speakers={cluster_count}."
        )
        return

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Char embedding でクラスタリングし、list を再生成する。
    """

    args = _parse_args()
    model_folder_name: str = args.model
    paths = TrainingModelPaths(model_folder_name)
    dataset_dir = paths.dataset_dir
    list_entries_map = _load_all_list_entries(dataset_dir)
    all_entries: list[list[str]] = []
    for entries in list_entries_map.values():
        all_entries.extend(entries)
    audio_paths = _iter_audio_paths(all_entries)
    audio_paths = list(dict.fromkeys(audio_paths))
    resolved_audio_paths = []
    base_dir = Path(__file__).parent.parent.parent
    for audio_path in audio_paths:
        resolved_path = Path(audio_path)
        if resolved_path.is_absolute():
            resolved_audio_paths.append(resolved_path)
            continue
        base_candidate_path = base_dir / resolved_path
        if base_candidate_path.exists():
            resolved_audio_paths.append(base_candidate_path)
            continue
        dataset_candidate_path = dataset_dir / resolved_path
        if dataset_candidate_path.exists():
            resolved_audio_paths.append(dataset_candidate_path)
            continue
        wavs_candidate_path = dataset_dir / "wavs" / resolved_path
        resolved_audio_paths.append(wavs_candidate_path)

    if len(audio_paths) == 0:
        raise ValueError("No audio paths found in list file.")

    logger.info("Loading anime speaker embedding model.")
    model = AnimeSpeakerEmbedding(device=args.device, variant="char")

    embeddings = []
    for audio_path in resolved_audio_paths:
        embeddings.append(
            _load_or_extract_embedding(audio_path, model, args.skip_existing)
        )

    embedding_matrix = np.stack(embeddings, axis=0)
    embedding_matrix = _normalize_embeddings(embedding_matrix)
    sim_stats = _compute_similarity_stats(embedding_matrix, DEFAULT_RANDOM_SEED)
    is_single_speaker = _is_single_speaker(sim_stats)
    silhouette_score_k2 = 0.0
    min_cluster_ratio_k2 = 0.0

    if is_single_speaker is True and len(embedding_matrix) >= 4:
        k2_assignments, _centroids_k2, _distance_k2 = _kmeans_cosine_multi_start(
            embedding_matrix,
            k=2,
            iters=DEFAULT_KMEANS_ITERS,
            seed=DEFAULT_RANDOM_SEED,
            restarts=DEFAULT_KMEANS_RESTARTS,
        )
        silhouette_score_k2 = _compute_silhouette_score(
            embedding_matrix,
            k2_assignments,
        )
        min_cluster_ratio_k2 = _compute_min_cluster_ratio(k2_assignments)
        if (
            silhouette_score_k2 >= SILHOUETTE_SPLIT_THRESHOLD
            and min_cluster_ratio_k2 >= MIN_CLUSTER_RATIO_FOR_SPLIT
        ):
            logger.info(
                "Silhouette suggests split despite single-speaker stats. "
                "Overriding single-speaker detection."
            )
            is_single_speaker = False

    if is_single_speaker is True:
        cluster_count = 1
        assignments = np.zeros(len(audio_paths), dtype=np.int_)
        distances_by_k = {1: 0.0}
        logger.info(
            "Single-speaker detected based on similarity stats. Using 1 cluster."
        )
    else:
        candidate_count = _decide_cluster_count(len(audio_paths))
        max_candidates = max(candidate_count, DEFAULT_MAX_CLUSTER_CANDIDATES)
        cluster_count, distances_by_k = _decide_cluster_count_elbow(
            embedding_matrix,
            max_candidates=max_candidates,
            seed=DEFAULT_RANDOM_SEED,
            iters=DEFAULT_KMEANS_ITERS,
            restarts=DEFAULT_KMEANS_RESTARTS,
        )
        logger.info(
            f"Clustering {len(audio_paths)} samples into {cluster_count} clusters."
        )
        assignments, _centroids, _distance = _kmeans_cosine_multi_start(
            embedding_matrix,
            k=cluster_count,
            iters=DEFAULT_KMEANS_ITERS,
            seed=DEFAULT_RANDOM_SEED,
            restarts=DEFAULT_KMEANS_RESTARTS,
        )

    label_counts: dict[int, int] = {}
    for cluster_id in assignments:
        label_counts[int(cluster_id)] = label_counts.get(int(cluster_id), 0) + 1
    cluster_stats = _summarize_cluster_counts(label_counts)

    audio_to_cluster = {
        audio_path: int(cluster_id)
        for audio_path, cluster_id in zip(audio_paths, assignments)
    }

    if args.dry_run:
        logger.info("[dry-run] Skipping list rewrite.")
    else:
        for list_path, entries in list_entries_map.items():
            with list_path.open("w", encoding="utf-8") as f:
                for fields in entries:
                    if len(fields) < 2:
                        raise ValueError(
                            "list entry must have speaker field at index 1."
                        )
                    audio_path = fields[0]
                    if audio_path not in audio_to_cluster:
                        raise ValueError(
                            f"Audio path is missing in cluster mapping: {audio_path}"
                        )
                    fields = fields.copy()
                    fields[1] = f"char_{audio_to_cluster[audio_path]:04d}"
                    f.write("|".join(fields) + "\n")

    mapping = {
        "cluster_count": int(cluster_count),
        "total_samples": len(audio_paths),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "cluster_stats": cluster_stats,
        "target_cluster_size": DEFAULT_TARGET_CLUSTER_SIZE,
        "kmeans_iters": DEFAULT_KMEANS_ITERS,
        "kmeans_restarts": DEFAULT_KMEANS_RESTARTS,
        "random_seed": DEFAULT_RANDOM_SEED,
        "max_cluster_candidates": DEFAULT_MAX_CLUSTER_CANDIDATES,
        "single_speaker_similarity_stats": sim_stats,
        "single_speaker_thresholds": {
            "centroid_mean": SINGLE_SPEAKER_MEAN_SIM_THRESHOLD,
            "centroid_p10": SINGLE_SPEAKER_P10_SIM_THRESHOLD,
            "pairwise_median": SINGLE_SPEAKER_PAIRWISE_MEDIAN_THRESHOLD,
            "pairwise_p10": SINGLE_SPEAKER_PAIRWISE_P10_THRESHOLD,
        },
        "single_speaker_override": {
            "silhouette_score_k2": silhouette_score_k2,
            "min_cluster_ratio_k2": min_cluster_ratio_k2,
            "silhouette_threshold": SILHOUETTE_SPLIT_THRESHOLD,
            "min_cluster_ratio_threshold": MIN_CLUSTER_RATIO_FOR_SPLIT,
        },
        "pairwise_sample_size": PAIRWISE_SAMPLE_SIZE,
        "elbow_distances_by_k": {str(k): float(v) for k, v in distances_by_k.items()},
        "elbow_relative_improvement_threshold": ELBOW_RELATIVE_IMPROVEMENT_THRESHOLD,
        "elbow_min_abs_improvement": ELBOW_MIN_ABS_IMPROVEMENT,
        "single_speaker_detected": is_single_speaker,
    }
    mapping_path = dataset_dir / "clusters.json"
    plot_path = dataset_dir / "clusters.png"
    tsne_plot_path = dataset_dir / "clusters_tsne.png"

    mapping["dry_run"] = bool(args.dry_run)
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    _plot_cluster_histogram(label_counts, plot_path)
    _plot_cluster_scatter_tsne(
        embedding_matrix,
        assignments,
        tsne_plot_path,
        DEFAULT_RANDOM_SEED,
    )

    _update_config_json(dataset_dir / "config.json", cluster_count, args.dry_run)

    logger.info("Clustered list updated.")


if __name__ == "__main__":
    main()
