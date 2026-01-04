"""
TODO:
importが重いので、WebUI全般が重くなっている。どうにかしたい。
"""

import json
import shutil
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from umap import UMAP

from default_style import save_styles_by_dirs
from style_bert_vits2.constants import DEFAULT_STYLE, GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.utils.paths import get_paths_config
from style_bert_vits2.utils.style_strength import (
    apply_style_strength,
    load_style_strength,
)


paths_config = get_paths_config()
dataset_root = paths_config.dataset_root
assets_root = paths_config.assets_root
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CLUSTER_NUM = 10
MAX_AUDIO_NUM = 10

tsne = TSNE(n_components=2, random_state=42, metric="cosine")
umap = UMAP(n_components=2, random_state=42, metric="cosine", n_jobs=1, min_dist=0.0)

wav_files: list[Path] = []
x = np.array([])
x_reduced = None
y_pred = np.array([])
mean = np.array([])
centroids = []


def load_style_strength_table(model_name: str) -> tuple[list[tuple[str, int]], str]:
    """スタイル強度調整用にスタイル一覧を取得する。"""

    style_entries, message, is_success = load_style_strength(model_name, assets_root)
    if is_success is False:
        return [], message
    return style_entries, message


def apply_style_strength_table(model_name: str, rows: list[list[Any]] | None) -> str:
    """入力値に基づいてスタイルベクトルを再スケーリングする。"""

    _, message = apply_style_strength(model_name, assets_root, rows)
    return message


def update_preview_weight_slider(desired_max: float, current_value: float) -> gr.Slider:
    """調整後の最大値に合わせて試聴用スライダーの上限と値を更新する。"""

    new_value = current_value if current_value <= desired_max else desired_max
    return gr.Slider(maximum=desired_max, value=new_value)


def run_style_strength_tts(
    model_name: str,
    text: str,
    style_name: str,
    style_weight: float,
) -> tuple[str, tuple[int, NDArray[Any]] | None]:
    """スタイル強度調整タブ内で音声を試聴するための簡易 TTS を実行する。"""

    if model_name.strip() == "":
        return "モデル名を入力してください。", None
    if style_name == "":
        return "スタイルを選択してください。", None

    model_path = assets_root / model_name / f"{model_name}.safetensors"
    config_path = assets_root / model_name / "config.json"
    style_vec_path = assets_root / model_name / "style_vectors.npy"

    if not model_path.exists():
        return f"{model_path} が存在しません。", None
    if not config_path.exists():
        return f"{config_path} が存在しません。", None
    if not style_vec_path.exists():
        return f"{style_vec_path} が存在しません。", None

    # 試聴のたびに新しいモデルを作成するため、メモリリークを防ぐために
    # 推論完了後に必ずアンロードする
    model = TTSModel(model_path, config_path, style_vec_path, device)
    try:
        result = model.infer(text, style=style_name, style_weight=style_weight)
        return "Success: 音声を生成しました。", result
    finally:
        model.unload()


def build_style_dropdown_update(entries: list[tuple[str, int]]) -> gr.Dropdown:
    """試聴スタイル選択用ドロップダウンを更新する。"""

    choices = [name for name, _ in entries]
    default_value = None
    for name, style_id in entries:
        if style_id != 0:
            default_value = name
            break
    if default_value is None and choices:
        default_value = choices[0]
    return gr.Dropdown(choices=choices, value=default_value)


def load(model_name: str, reduction_method: str):
    global wav_files, x, x_reduced, mean
    wavs_dir = dataset_root / model_name / "wavs"
    style_vector_files = [f for f in wavs_dir.rglob("*.npy") if f.is_file()]
    # foo.wav.npy -> foo.wav
    wav_files = [f.with_suffix("") for f in style_vector_files]
    logger.info(f"Found {len(style_vector_files)} style vectors in {wavs_dir}")
    style_vectors = [np.load(f) for f in style_vector_files]
    x = np.array(style_vectors)
    mean = np.mean(x, axis=0)
    if reduction_method == "t-SNE":
        x_reduced = tsne.fit_transform(x)
    elif reduction_method == "UMAP":
        x_reduced = umap.fit_transform(x)
    else:
        raise ValueError("Invalid reduction method")
    x_reduced = np.asarray(x_reduced)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1])
    return plt


def do_clustering(n_clusters=4, method="KMeans"):
    global centroids, x_reduced, y_pred
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x)
    elif method == "KMeans after reduction":
        assert x_reduced is not None
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x_reduced)
    elif method == "Agglomerative after reduction":
        assert x_reduced is not None
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x_reduced)
    else:
        raise ValueError("Invalid method")

    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))

    return y_pred, centroids


def do_dbscan(eps=2.5, min_samples=15):
    global centroids, x_reduced, y_pred
    model = DBSCAN(eps=eps, min_samples=min_samples)
    assert x_reduced is not None
    y_pred = model.fit_predict(x_reduced)
    n_clusters = max(y_pred) + 1
    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))
    return y_pred, centroids


def representative_wav_files(cluster_id, num_files=1):
    # y_predの中でcluster_indexに関するメドイドを探す
    cluster_indices = np.where(y_pred == cluster_id)[0]
    cluster_vectors = x[cluster_indices]
    # クラスタ内の全ベクトル間の距離を計算
    distances = pdist(cluster_vectors)
    distance_matrix = squareform(distances)

    # 各ベクトルと他の全ベクトルとの平均距離を計算
    mean_distances = distance_matrix.mean(axis=1)

    # 平均距離が最も小さい順にnum_files個のインデックスを取得
    closest_indices = np.argsort(mean_distances)[:num_files]

    return cluster_indices[closest_indices]


def do_dbscan_gradio(eps=2.5, min_samples=15):
    global x_reduced, centroids

    y_pred, centroids = do_dbscan(eps, min_samples)

    assert x_reduced is not None

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(max(y_pred) + 1):
        plt.scatter(
            x_reduced[y_pred == i, 0],
            x_reduced[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    # Noise cluster (-1) is black
    plt.scatter(
        x_reduced[y_pred == -1, 0],
        x_reduced[y_pred == -1, 1],
        color="black",
        label="Noise",
    )
    plt.legend()

    n_clusters = int(max(y_pred) + 1)

    if n_clusters > MAX_CLUSTER_NUM:
        # raise ValueError(f"The number of clusters is too large: {n_clusters}")
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            f"クラスタ数が多すぎます、パラメータを変えてみてください。: {n_clusters}",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    elif n_clusters == 0:
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            "クラスタが数が0です。パラメータを変えてみてください。",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    return [plt, gr.Slider(maximum=n_clusters, value=1), n_clusters] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def representative_wav_files_gradio(cluster_id, num_files=1):
    cluster_id = cluster_id - 1  # UIでは1から始まるので0からにする
    closest_indices = representative_wav_files(cluster_id, num_files)
    actual_num_files = len(closest_indices)  # ファイル数が少ないときのため
    return [
        gr.Audio(wav_files[i], visible=True, label=str(wav_files[i]))
        for i in closest_indices
    ] + [gr.update(visible=False)] * (MAX_AUDIO_NUM - actual_num_files)


def do_clustering_gradio(n_clusters=4, method="KMeans"):
    global x_reduced, centroids
    y_pred, centroids = do_clustering(n_clusters, method)

    assert x_reduced is not None
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(n_clusters):
        plt.scatter(
            x_reduced[y_pred == i, 0],
            x_reduced[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    plt.legend()

    return [plt, gr.Slider(maximum=n_clusters, value=1)] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def save_style_vectors_from_clustering(model_name: str, style_names_str: str):
    """centerとcentroidsを保存する"""
    result_dir = assets_root / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    style_vectors = np.stack([mean] + centroids)
    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    np.save(style_vector_path, style_vectors)
    logger.success(f"Saved style vectors to {style_vector_path}")

    # config.jsonの更新
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path}が存在しません。"
    style_names = [name.strip() for name in style_names_str.split(",")]
    style_name_list = [DEFAULT_STYLE] + style_names
    if len(style_name_list) != len(centroids) + 1:
        return f"スタイルの数が合いません。`,`で正しく{len(centroids)}個に区切られているか確認してください: {style_names_str}"
    if len(set(style_names)) != len(style_names):
        return "スタイル名が重複しています。"

    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.success(f"Updated {config_path}")
    return f"成功!\n{style_vector_path}に保存し{config_path}を更新しました。"


def save_style_vectors_from_files(
    model_name: str, audio_files_str: str, style_names_str: str
):
    """音声ファイルからスタイルベクトルを作成して保存する"""
    global mean
    if len(x) == 0:
        return "Error: スタイルベクトルを読み込んでください。"
    mean = np.mean(x, axis=0)

    result_dir = assets_root / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    audio_files = [name.strip() for name in audio_files_str.split(",")]
    style_names = [name.strip() for name in style_names_str.split(",")]
    if len(audio_files) != len(style_names):
        return f"音声ファイルとスタイル名の数が合いません。`,`で正しく{len(style_names)}個に区切られているか確認してください: {audio_files_str}と{style_names_str}"
    style_name_list = [DEFAULT_STYLE] + style_names
    if len(set(style_names)) != len(style_names):
        return "スタイル名が重複しています。"
    style_vectors = [mean]

    wavs_dir = dataset_root / model_name / "wavs"
    for audio_file in audio_files:
        path = wavs_dir / audio_file
        if not path.exists():
            return f"{path}が存在しません。"
        style_vectors.append(np.load(f"{path}.npy"))
    style_vectors = np.stack(style_vectors)
    assert len(style_name_list) == len(style_vectors)
    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    np.save(style_vector_path, style_vectors)

    # config.jsonの更新
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path}が存在しません。"
    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")

    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    return f"成功!\n{style_vector_path}に保存し{config_path}を更新しました。"


def save_style_vectors_by_dirs(model_name: str, audio_dir_str: str):
    if model_name == "":
        return "モデル名を入力してください。"
    if audio_dir_str == "":
        return "音声ファイルが入っているディレクトリを入力してください。"

    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import cpu_count

    from tqdm import tqdm

    from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
    from style_gen import save_style_vector

    # First generate style vectors for each audio file

    audio_dir = Path(audio_dir_str)
    audio_suffixes = [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"]
    audio_files = [f for f in audio_dir.rglob("*") if f.suffix in audio_suffixes]

    def process(file: Path):
        # f: `test.wav` -> search `test.wav.npy`
        if (file.with_name(file.name + ".npy")).exists():
            return file, None
        try:
            save_style_vector(str(file))
        except Exception as e:
            return file, e
        return file, None

    with ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
        _ = list(
            tqdm(
                executor.map(
                    process,
                    audio_files,
                ),
                total=len(audio_files),
                file=SAFE_STDOUT,
                desc="Generating style vectors",
                dynamic_ncols=True,
            )
        )

    result_dir = assets_root / model_name
    config_path = result_dir / "config.json"
    if not config_path.exists():
        return f"{config_path}が存在しません。"
    logger.info(f"Backup {config_path} to {config_path}.bak")
    shutil.copy(config_path, f"{config_path}.bak")

    style_vector_path = result_dir / "style_vectors.npy"
    if style_vector_path.exists():
        logger.info(f"Backup {style_vector_path} to {style_vector_path}.bak")
        shutil.copy(style_vector_path, f"{style_vector_path}.bak")
    save_styles_by_dirs(
        wav_dir=audio_dir,
        output_dir=result_dir,
        config_path=config_path,
        config_output_path=config_path,
    )
    return f"成功!\n{result_dir}にスタイルベクトルを保存しました。"


how_to_md = f"""
Style-Bert-VITS2でこまかくスタイルを指定して音声合成するには、モデルごとにスタイルベクトルのファイル`style_vectors.npy`を作成する必要があります。

ただし、学習の過程では自動的に、平均スタイル「{DEFAULT_STYLE}」と、（**Ver 2.5.0以降からは**）音声をサブフォルダに分けていた場合はそのサブフォルダごとのスタイルが保存されています。

## 方法

- 方法0: 音声を作りたいスタイルごとのサブフォルダに分け、そのフォルダごとにスタイルベクトルを作成
- 方法1: 音声ファイルを自動でスタイル別に分け、その各スタイルの平均を取って保存
- 方法2: スタイルを代表する音声ファイルを手動で選んで、その音声のスタイルベクトルを保存
- 方法3: 自分でもっと頑張ってこだわって作る（JVNVコーパスなど、もともとスタイルラベル等が利用可能な場合はこれがよいかも）
"""

method0 = """
音声をスタイルごとにサブフォルダを作り、その中に音声ファイルを入れてください。

**注意**

- Ver 2.5.0以降では、`inputs/`フォルダや`raw/`フォルダにサブディレクトリに分けて音声ファイルを入れるだけで、スタイルベクトルが自動で作成されるので、この手順は不要です。
- それ未満のバージョンで学習したモデルに新しくスタイルベクトルをつけたい場合や、学習に使ったのとは別の音声でスタイルベクトルを作成したい場合に使います。
- 学習との整合性のため、もし**現在学習中や、今後学習する予定がある場合は**、音声ファイルは、`Data/{モデル名}/wavs`フォルダではなく**新しい別のディレクトリに保存してください**。

例:

```bash
audio_dir
├── style1
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── style2
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```
"""

method1 = f"""
学習の時に取り出したスタイルベクトルを読み込んで、可視化を見ながらスタイルを分けていきます。

手順:
1. 図を眺める
2. スタイル数を決める（平均スタイルを除く）
3. スタイル分けを行って結果を確認
4. スタイルの名前を決めて保存


詳細: スタイルベクトル(256次元)たちを適当なアルゴリズムでクラスタリングして、各クラスタの中心のベクトル（と全体の平均ベクトル）を保存します。

平均スタイル（{DEFAULT_STYLE}）は自動的に保存されます。
"""

dbscan_md = """
DBSCANという方法でスタイル分けを行います。
こちらの方が方法1よりも特徴がはっきり出るもののみを取り出せ、よいスタイルベクトルが作れるかもしれません。
ただし事前にスタイル数は指定できません。

パラメータ：
- eps: この値より近い点同士をどんどん繋げて同じスタイル分類とする。小さいほどスタイル数が増え、大きいほどスタイル数が減る傾向。
- min_samples: ある点をスタイルの核となる点とみなすために必要な近傍の点の数。小さいほどスタイル数が増え、大きいほどスタイル数が減る傾向。

UMAPの場合はepsは0.3くらい、t-SNEの場合は2.5くらいがいいかもしれません。min_samplesはデータ数に依存するのでいろいろ試してみてください。

詳細：
https://ja.wikipedia.org/wiki/DBSCAN
"""


def create_style_vectors_app():
    with gr.Blocks(theme=GRADIO_THEME) as app:
        with gr.Accordion("使い方", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(placeholder="your_model_name", label="モデル名")
        with gr.Tab("方法0: サブフォルダごとにスタイルベクトルを作成"):
            gr.Markdown(method0)
            audio_dir = gr.Textbox(
                placeholder="path/to/audio_dir",
                label="音声が入っているフォルダ",
                info="音声ファイルをスタイルごとにサブフォルダに分けて保存してください。",
            )
            method0_btn = gr.Button("スタイルベクトルを作成", variant="primary")
            method0_info = gr.Textbox(label="結果")
            method0_btn.click(
                save_style_vectors_by_dirs,
                inputs=[model_name, audio_dir],
                outputs=[method0_info],
            )
        with gr.Tab("その他の方法"):
            with gr.Row():
                reduction_method = gr.Radio(
                    choices=["UMAP", "t-SNE"],
                    label="次元削減方法",
                    info="v 1.3以前はt-SNEでしたがUMAPのほうがよい可能性もあります。",
                    value="UMAP",
                )
                load_button = gr.Button("スタイルベクトルを読み込む", variant="primary")
            output = gr.Plot(label="音声スタイルの可視化")
            load_button.click(
                load, inputs=[model_name, reduction_method], outputs=[output]
            )
            with gr.Tab("方法1: スタイル分けを自動で行う"):
                with gr.Tab("スタイル分け1"):
                    n_clusters = gr.Slider(
                        minimum=2,
                        maximum=10,
                        step=1,
                        value=4,
                        label="作るスタイルの数（平均スタイルを除く）",
                        info="上の図を見ながらスタイルの数を試行錯誤してください。",
                    )
                    c_method = gr.Radio(
                        choices=[
                            "Agglomerative after reduction",
                            "KMeans after reduction",
                            "Agglomerative",
                            "KMeans",
                        ],
                        label="アルゴリズム",
                        info="分類する（クラスタリング）アルゴリズムを選択します。いろいろ試してみてください。",
                        value="Agglomerative after reduction",
                    )
                    c_button = gr.Button("スタイル分けを実行")
                with gr.Tab("スタイル分け2: DBSCAN"):
                    gr.Markdown(dbscan_md)
                    eps = gr.Slider(
                        minimum=0.1,
                        maximum=10,
                        step=0.01,
                        value=0.3,
                        label="eps",
                    )
                    min_samples = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=15,
                        label="min_samples",
                    )
                    with gr.Row():
                        dbscan_button = gr.Button("スタイル分けを実行")
                        num_styles_result = gr.Textbox(label="スタイル数")
                gr.Markdown("スタイル分けの結果")
                gr.Markdown(
                    "注意: もともと256次元なものをを2次元に落としているので、正確なベクトルの位置関係ではありません。"
                )
                with gr.Row():
                    gr_plot = gr.Plot()
                    with gr.Column():
                        with gr.Row():
                            cluster_index = gr.Slider(
                                minimum=1,
                                maximum=MAX_CLUSTER_NUM,
                                step=1,
                                value=1,
                                label="スタイル番号",
                                info="選択したスタイルの代表音声を表示します。",
                            )
                            num_files = gr.Slider(
                                minimum=1,
                                maximum=MAX_AUDIO_NUM,
                                step=1,
                                value=5,
                                label="代表音声の数をいくつ表示するか",
                            )
                            get_audios_button = gr.Button("代表音声を取得")
                        with gr.Row():
                            audio_list = []
                            for i in range(MAX_AUDIO_NUM):
                                audio_list.append(
                                    gr.Audio(visible=False, show_label=True)
                                )
                    c_button.click(
                        do_clustering_gradio,
                        inputs=[n_clusters, c_method],
                        outputs=[gr_plot, cluster_index] + audio_list,
                    )
                    dbscan_button.click(
                        do_dbscan_gradio,
                        inputs=[eps, min_samples],
                        outputs=[gr_plot, cluster_index, num_styles_result]
                        + audio_list,
                    )
                    get_audios_button.click(
                        representative_wav_files_gradio,
                        inputs=[cluster_index, num_files],
                        outputs=audio_list,
                    )
                gr.Markdown("結果が良さそうなら、これを保存します。")
                style_names = gr.Textbox(
                    "Angry, Sad, Happy",
                    label="スタイルの名前",
                    info=f"スタイルの名前を`,`で区切って入力してください（日本語可）。例: `Angry, Sad, Happy`や`怒り, 悲しみ, 喜び`など。平均音声は{DEFAULT_STYLE}として自動的に保存されます。",
                )
                with gr.Row():
                    save_button1 = gr.Button(
                        "スタイルベクトルを保存", variant="primary"
                    )
                    info2 = gr.Textbox(label="保存結果")

                save_button1.click(
                    save_style_vectors_from_clustering,
                    inputs=[model_name, style_names],
                    outputs=[info2],
                )
            with gr.Tab("方法2: 手動でスタイルを選ぶ"):
                gr.Markdown(
                    "下のテキスト欄に、各スタイルの代表音声のファイル名を`,`区切りで、その横に対応するスタイル名を`,`区切りで入力してください。"
                )
                gr.Markdown("例: `angry.wav, sad.wav, happy.wav`と`Angry, Sad, Happy`")
                gr.Markdown(
                    f"注意: {DEFAULT_STYLE}スタイルは自動的に保存されます、手動では{DEFAULT_STYLE}という名前のスタイルは指定しないでください。"
                )
                with gr.Row():
                    audio_files_text = gr.Textbox(
                        label="音声ファイル名",
                        placeholder="angry.wav, sad.wav, happy.wav",
                    )
                    style_names_text = gr.Textbox(
                        label="スタイル名", placeholder="Angry, Sad, Happy"
                    )
                with gr.Row():
                    save_button2 = gr.Button(
                        "スタイルベクトルを保存", variant="primary"
                    )
                    info2 = gr.Textbox(label="保存結果")
                    save_button2.click(
                        save_style_vectors_from_files,
                        inputs=[model_name, audio_files_text, style_names_text],
                        outputs=[info2],
                    )
        with gr.Tab("スタイル強度調整"):
            gr.Markdown(
                "各スタイルごとに、「スタイルの強さ」パラメーターの効き方を統一します。<br>"
                " まず共通の目標値（例: 10）を決め、試聴時の各スタイルのスタイル強度を徐々に上げていき、耳で聞いて「これ以上上げると音声が不自然になる」と感じた値を入力してください。<br>"
                "結果を保存するとバックアップを作成の上で、既存の `style_vectors.npy` が上書きされます。"
            )

            load_strength_button = gr.Button("スタイル一覧を読み込む")
            desired_max_slider = gr.Slider(
                label="共通の目標値（例: 10）",
                minimum=1.0,
                maximum=50.0,
                step=0.5,
                value=10.0,
            )

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        preview_style_dropdown = gr.Dropdown(
                            label="試聴するスタイル", choices=[]
                        )
                        preview_weight_slider = gr.Slider(
                            label="試聴時のスタイル強度",
                            minimum=0.0,
                            maximum=50.0,
                            step=0.5,
                            value=5.0,
                        )
                    preview_text = gr.Textbox(
                        label="試聴用テキスト",
                        value="これはテスト音声です。これから調整前のスタイル強度の効き方を確認します。\nあなたがそんなこと言うなんて、私はとっても嬉しい。\nあなたがそんなこと言うなんて、私はとっても怒ってる。\nあなたがそんなこと言うなんて、私はとっても驚いてる。\nあなたがそんなこと言うなんて、私はとっても辛い。\n試聴時の各スタイルのスタイル強度を徐々に上げていき、耳で聞いて「これ以上上げると音声が不自然になる」と感じた値を入力してください。",
                        lines=6,
                    )
                    preview_button = gr.Button(
                        "調整前のスタイル強度で音声を試聴", variant="primary"
                    )
                with gr.Column(scale=1):
                    preview_info = gr.Textbox(label="情報")
                    preview_audio = gr.Audio(label="試聴結果")

            adjustable_style_names: list[str] = []
            adjustable_sliders: list[gr.Slider] = []
            apply_strength_button = gr.Button(
                "スタイルを保存",
                variant="primary",
            )
            style_strength_info = gr.Textbox(label="結果")

            style_strength_entries_state = gr.State([])

            @gr.render(inputs=[style_strength_entries_state, desired_max_slider])
            def render_style_strength_controls(
                style_entries: list[tuple[str, int]], desired_max_value: float
            ) -> None:
                nonlocal adjustable_style_names, adjustable_sliders
                adjustable_style_names = []
                adjustable_sliders = []
                if not style_entries:
                    gr.Markdown("スタイル一覧を読み込んでください。")
                    return
                for style_name, style_id in style_entries:
                    if style_id == 0:
                        gr.Markdown(
                            f"- {style_name} は平均スタイルのため調整対象外です。"
                        )
                        continue
                    slider = gr.Slider(
                        label=f"『{style_name}』の「これ以上上げると音声が不自然になる」と感じたスタイル強度の値",
                        minimum=0.5,
                        maximum=50.0,
                        step=0.5,
                        value=float(desired_max_value),
                    )
                    adjustable_style_names.append(style_name)
                    adjustable_sliders.append(slider)
                if not adjustable_sliders:
                    gr.Markdown("調整対象のスタイルが存在しません。")

                def _apply_strength(data: dict[gr.components.Component, Any]) -> str:
                    if not adjustable_sliders:
                        return "スタイル一覧を読み込んでください。"
                    desired_max = data[desired_max_slider]
                    rows: list[list[Any]] = []
                    for style_name, slider in zip(
                        adjustable_style_names, adjustable_sliders
                    ):
                        rows.append([style_name, data[slider], desired_max])
                    return apply_style_strength_table(data[model_name], rows)

                if adjustable_sliders:
                    apply_strength_button.click(
                        _apply_strength,
                        inputs=set(
                            [model_name, desired_max_slider] + adjustable_sliders
                        ),
                        outputs=[style_strength_info],
                    )

            load_strength_button.click(
                load_style_strength_table,
                inputs=[model_name],
                outputs=[style_strength_entries_state, style_strength_info],
            ).then(
                build_style_dropdown_update,
                inputs=[style_strength_entries_state],
                outputs=[preview_style_dropdown],
            )

            desired_max_slider.change(
                update_preview_weight_slider,
                inputs=[desired_max_slider, preview_weight_slider],
                outputs=[preview_weight_slider],
            )

            preview_button.click(
                run_style_strength_tts,
                inputs=[
                    model_name,
                    preview_text,
                    preview_style_dropdown,
                    preview_weight_slider,
                ],
                outputs=[preview_info, preview_audio],
            )

    return app


if __name__ == "__main__":
    app = create_style_vectors_app()
    app.launch(inbrowser=True)
