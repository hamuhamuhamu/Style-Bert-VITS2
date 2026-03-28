"""
TensorBoard への書き込みや可視化など、学習スクリプトでのみ使用される関数を提供する。
"""

import glob
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from scipy.io.wavfile import read

from style_bert_vits2.logging import logger


if TYPE_CHECKING:
    # tensorboard はライブラリとしてインストールされている場合は依存関係に含まれないため、
    # 型チェック時のみインポートする
    from torch.utils.tensorboard import SummaryWriter


__is_matplotlib_imported = False


def summarize(
    writer: "SummaryWriter",
    global_step: int,
    scalars: dict[str, float | torch.Tensor] = {},
    histograms: dict[str, Any] = {},
    images: dict[str, Any] = {},
    audios: dict[str, Any] = {},
    audio_sampling_rate: int = 22050,
) -> None:
    """
    指定されたデータを TensorBoard にまとめて追加する

    Args:
        writer (SummaryWriter): TensorBoard への書き込みを行うオブジェクト
        global_step (int): グローバルステップ数
        scalars (dict[str, float | torch.Tensor]): スカラー値の辞書
        histograms (dict[str, Any]): ヒストグラムの辞書
        images (dict[str, Any]): 画像データの辞書
        audios (dict[str, Any]): 音声データの辞書
        audio_sampling_rate (int): 音声データのサンプリングレート
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def is_resuming(dir_path: str | Path) -> bool:
    """
    指定されたディレクトリパスに再開可能なモデルが存在するかどうかを返す

    Args:
        dir_path: チェックするディレクトリのパス

    Returns:
        bool: 再開可能なモデルが存在するかどうか
    """
    # JP-ExtraバージョンではDURがなくWDがあったり変わるため、Gのみで判断する
    g_list = glob.glob(os.path.join(dir_path, "G_*.pth"))
    # d_list = glob.glob(os.path.join(dir_path, "D_*.pth"))
    # dur_list = glob.glob(os.path.join(dir_path, "DUR_*.pth"))
    return len(g_list) > 0


def plot_spectrogram_to_numpy(spectrogram: NDArray[Any]) -> NDArray[Any]:
    """
    指定されたスペクトログラムを画像データに変換する

    Args:
        spectrogram (NDArray[Any]): スペクトログラム

    Returns:
        NDArray[Any]: 画像データ
    """

    global __is_matplotlib_imported
    if not __is_matplotlib_imported:
        import matplotlib

        matplotlib.use("Agg")
        __is_matplotlib_imported = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import matplotlib_fontja  # pyright: ignore[reportUnusedImport]

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(
    alignment: NDArray[Any], info: str | None = None
) -> NDArray[Any]:
    """
    指定されたアライメントを画像データに変換する

    Args:
        alignment (NDArray[Any]): アライメント
        info (str | None): 画像に追加する情報

    Returns:
        NDArray[Any]: 画像データ
    """

    global __is_matplotlib_imported
    if not __is_matplotlib_imported:
        import matplotlib

        matplotlib.use("Agg")
        __is_matplotlib_imported = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import matplotlib_fontja  # pyright: ignore[reportUnusedImport]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path: str | Path) -> tuple[torch.FloatTensor, int]:
    """
    指定された音声ファイルを読み込み、PyTorch のテンソルに変換して返す。

    既存の WAV 学習データとの互換性を維持するため、`.wav` / `.WAV` は従来どおり
    `scipy.io.wavfile.read()` で raw PCM スケールのまま読み込む。
    それ以外の形式は `soundfile` で float32 にデコードして返す。
    呼び出し側では、値域を見て追加の正規化が必要かどうかを判定する。

    Args:
        full_path (str | Path): 音声ファイルのパス

    Returns:
        tuple[torch.FloatTensor, int]: 音声データのテンソルとサンプリングレート
    """

    full_path_obj = Path(full_path)

    # WAV は従来どおり raw PCM スケールで読み込み、既存学習データとの互換性を維持する
    if full_path_obj.suffix.lower() == ".wav":
        sampling_rate, data = read(full_path_obj)
        audio = np.asarray(data, dtype=np.float32)
    else:
        # 非 WAV は soundfile でデコードし、OGG Vorbis / FLAC 等に対応する
        ## soundfile は float32 [-1, 1] を返すため、呼び出し側で追加正規化は原則不要
        data, sampling_rate = sf.read(full_path_obj, dtype="float32")
        audio = np.asarray(data, dtype=np.float32)

    # マルチチャンネル音声を黙って通すと学習データが壊れるため、明示的にエラーにする
    if audio.ndim > 1:
        raise ValueError(
            f"Multi-channel audio is not supported (expected mono). "
            f"channels: {audio.shape[1]}, file: {full_path}"
        )
    return torch.FloatTensor(audio), sampling_rate


def load_filepaths_and_text(filename: str | Path, split: str = "|") -> list[list[str]]:
    """
    指定されたファイルからファイルパスとテキストを読み込む

    Args:
        filename (str | Path): ファイルのパス
        split (str): ファイルの区切り文字 (デフォルト: "|")

    Returns:
        list[list[str]]: ファイルパスとテキストのリスト
    """

    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_logger(
    model_dir_path: str | Path, filename: str = "train.log"
) -> logging.Logger:
    """
    ロガーを取得する

    Args:
        model_dir_path (str | Path): ログを保存するディレクトリのパス
        filename (str): ログファイルの名前 (デフォルト: "train.log")

    Returns:
        logging.Logger: ロガー
    """

    file_logger = logging.getLogger(os.path.basename(model_dir_path))
    file_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    h = logging.FileHandler(os.path.join(model_dir_path, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    file_logger.addHandler(h)
    return file_logger


def get_steps(model_path: str | Path) -> int | None:
    """
    モデルのパスからイテレーション回数を取得する

    Args:
        model_path (str | Path): モデルのパス

    Returns:
        int | None: イテレーション回数
    """

    model_path_str = str(model_path)
    matches = re.findall(r"\d+", model_path_str)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def check_git_hash(model_dir_path: str | Path) -> None:
    """
    モデルのディレクトリに .git ディレクトリが存在する場合、ハッシュ値を比較する

    Args:
        model_dir_path (str | Path): モデルのディレクトリのパス
    """

    # training/ ディレクトリの親（プロジェクトルート）を取得
    source_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(
            f"{source_dir} is not a git repository, therefore hash value comparison will be ignored."
        )
        return

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        cur_hash = result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as ex:
        logger.warning(f"Failed to get git hash: {ex}")
        return

    path = os.path.join(model_dir_path, "githash")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved_hash = f.read()
        if saved_hash != cur_hash:
            logger.warning(
                f"git hash values are different. {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)"
            )
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cur_hash)
