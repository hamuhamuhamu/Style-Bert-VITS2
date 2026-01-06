from __future__ import annotations

import argparse
import csv
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import utmosv2
from scipy.io import wavfile
from tqdm import tqdm
from utmosv2 import UTMOSv2Model

from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.utils.paths import get_paths_config


MOS_RESULT_DIR = Path(__file__).parent / "mos_results"
TEMP_AUDIO_DIR = MOS_RESULT_DIR / "mos_audio"
UTMOS_BATCH_SIZE = 4
UTMOS_NUM_WORKERS = 0

TEST_TEXTS = [
    # JVNVコーパスのテキスト
    # https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus
    # CC BY-SA 4.0
    "ああ？どうしてこんなに荒々しい態度をとるんだ？落ち着いて話を聞けばいいのに。",
    "いや、あんな醜い人間を見るのは本当に嫌だ。",
    "うわ、不景気の影響で失業してしまうかもしれない。どうしよう、心配で眠れない。",
    "今日の山登りは最高だった！山頂で見た景色は言葉に表せないほど美しかった！あはは、絶頂の喜びが胸に溢れるよ！",
    "あーあ、昨日の事故で大切な車が全損になっちゃった。もうどうしようもないよ。",
    "ああ、彼は本当に速い！ダッシュの速さは尋常じゃない！",
    # 以下 app.py の説明文章
    "音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。",
    "この分野の最新の研究成果を使うと、より自然で表現豊かな音声の生成が可能である。深層学習の応用により、感情やアクセントを含む声質の微妙な変化も再現することが出来る。",
    # 追加テキスト
    "おはようございます！現在時刻は7時30分です。今日の東京の気温は18度で、天気は雨です。10時からミーティング、午後3時に歯医者の予約があります。今日も素敵な一日になりますように。",
    "やった〜！テストでようやく満点取れた〜！めちゃくちゃ嬉しい…。　そうそう、さっき読んでたこの漫画がめっちゃ面白くてさ〜！見てよこれ！",
    "ごめんね、今ちょっと風邪気味なんだよね…。それでもよければ会いたいけど、どう？　…………そっか…。コロナ流行ってるもんね。じゃまた今度にしようか。…元気になったらぜひご飯でも！",
    "イーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。",
    "小笠原近海で台風５号が発生しました。今後、北上し、関東から東北の太平洋側に沿って北上した後、北海道付近に到達する可能性が大きくなっています。",
    "もし関東へ上陸すれば６年ぶり、東北に上陸すれば２年連続、北海道へ上陸すれば９年ぶりとなります。この台風の進路の特徴とともに、詳しくみていきましょう。",
    "濁流は、メロスの叫びをせせら笑う如く、ますます激しく躍り狂う。浪は浪を呑み、捲き、煽り立て、そうして時は、刻一刻と消えて行く。今はメロスも覚悟した。泳ぎ切るより他に無い。ああ、神々も照覧あれ！",
    "万博協会が13日に発表した、12日（土）の大阪・関西万博の一般来場者数は速報値ベースで約16万4000人、パビリオンなどの関係者を含めた総来場者数は約18万2000人で、1日あたりの来場者数が“過去3番目”となりました。",
    "12日は、航空自衛隊の「ブルーインパルス」による展示飛行が行われ、多くの人が会場に詰めかけ、歓声を上げました。",
    "13日も午後2時40分ごろに関西空港を離陸後、大阪市の通天閣や吹田市の万博記念公園などの上空を通過した上で、午後3時ごろから15分程度、会場上空などで展示飛行が予定されています。",
    "サンプリングレート 44100Hz で出力されるクリアな音質は、IVR やカスタマーサポート、音声アシスタントなど、お客様と直接コミュニケーションをとる重要な場面でも、安心してご利用いただける品質です。",
    "感情表現の強さやテンポの調整も自由自在。例えば、重要なお知らせでは感情表現を抑えた信頼感のあるトーン、プロモーションでは明るく親しみやすいトーンといった具合に、お客様のブランドイメージに合わせた音声コミュニケーションを設計できます。",
    "血圧は 118/76、脈拍は 72 です。念のため、胸の痛みや息切れが続く場合は早めに受診してください。",
    "円相場は 1 ドル 150 円台まで円安が進みました。金利差の拡大を背景に、輸入コストの上昇が懸念されています。",
    "次の駅は新宿です。お降りの方は足元にご注意ください。",
    "ログの出力が増えたので、レベルを INFO から WARN に変更して挙動を確認します。",
    "この製品は 12V で動作し、消費電力は最大 18W です。連続使用は 30 分までにしてください。",
]


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
        help="Model folder name (folder name under model_assets/).",
    )
    parser.add_argument("--device", "-d", type=str, default="cuda")
    return parser.parse_args()


def _resolve_model_files(model_name: str) -> list[Path]:
    """
    指定モデルの .safetensors ファイル一覧を取得する。

    Args:
        model_name (str): モデル名

    Returns:
        list[Path]: チェックポイントのパス一覧
    """

    model_path = get_paths_config().assets_root / model_name
    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if len(safetensors_files) == 0:
        raise ValueError(f"No .safetensors files found in {model_path}.")
    return safetensors_files


def _extract_step_count(file_name: str) -> int | None:
    """
    ファイル名からステップ数を抽出する。

    Args:
        file_name (str): ファイル名

    Returns:
        int | None: 抽出したステップ数
    """

    match = re.search(r"_s(\d+)\.safetensors$", file_name)
    if match is not None:
        return int(match.group(1))
    return None


def _create_tts_model(model_file: Path, device: str) -> TTSModel:
    """
    TTS モデルを読み込む。

    Args:
        model_file (Path): チェックポイントのパス
        device (str): 推論デバイス

    Returns:
        TTSModel: 読み込んだ TTS モデル
    """

    return TTSModel(
        model_path=model_file,
        config_path=model_file.parent / "config.json",
        style_vec_path=model_file.parent / "style_vectors.npy",
        device=device,
    )


def _prepare_audio_dir(model_file: Path) -> Path:
    """
    MOS 推論用の一時音声ディレクトリを作成する。

    Args:
        model_file (Path): チェックポイントのパス

    Returns:
        Path: 一時音声ディレクトリのパス
    """

    audio_dir = TEMP_AUDIO_DIR / model_file.stem
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def _clear_audio_dir(audio_dir: Path) -> None:
    """
    MOS 推論用の一時音声ディレクトリを初期化する。

    Args:
        audio_dir (Path): 一時音声ディレクトリ
    """

    for audio_path in audio_dir.glob("*.wav"):
        audio_path.unlink()


def _generate_audio_files(
    tts_model: TTSModel,
    texts: list[str],
    audio_dir: Path,
) -> list[Path]:
    """
    テキストから音声を生成し、MOS 推論用の wav ファイルとして保存する。

    Args:
        tts_model (TTSModel): TTS モデル
        texts (list[str]): 評価に使うテキスト
        audio_dir (Path): 保存先ディレクトリ

    Returns:
        list[Path]: 生成した wav ファイルのパス
    """

    audio_paths: list[Path] = []
    for index, text in enumerate(texts):
        sampling_rate, audio = tts_model.infer(text)
        audio = audio.astype("float32")
        audio = TTSModel.convert_to_16_bit_wav(audio)
        output_path = audio_dir / f"{index:04d}.wav"
        wavfile.write(output_path, sampling_rate, audio)
        audio_paths.append(output_path)
    return audio_paths


def _predict_mos_from_audio_dir(
    utmos_model: UTMOSv2Model,
    audio_dir: Path,
    audio_paths: list[Path],
    device: torch.device,
) -> list[float]:
    """
    保存済み wav から MOS を推定する。

    Args:
        utmos_model (UTMOSv2Model): UTMOSv2 モデル
        audio_dir (Path): wav 保存先ディレクトリ
        audio_paths (list[Path]): wav ファイルのパス
        device (torch.device): UTMOSv2 推論デバイス

    Returns:
        list[float]: MOS の一覧
    """

    results = utmos_model.predict(
        input_dir=audio_dir,
        device=device,
        num_workers=UTMOS_NUM_WORKERS,
        batch_size=UTMOS_BATCH_SIZE,
        remove_silent_section=True,
        verbose=False,
    )
    result_map: dict[str, float] = {}
    for item in results:
        file_path_value = item["file_path"]
        predicted_mos_value = item["predicted_mos"]
        if not isinstance(file_path_value, str):
            raise ValueError("Unexpected UTMOSv2 result format.")
        result_map[Path(file_path_value).name] = float(predicted_mos_value)

    scores: list[float] = []
    for audio_path in audio_paths:
        file_name = audio_path.name
        is_missing = file_name not in result_map
        if is_missing is True:
            raise ValueError(f"MOS result not found for {file_name}.")
        scores.append(result_map[file_name])
    return scores


def _evaluate_model(
    tts_model: TTSModel,
    utmos_model: UTMOSv2Model,
    model_label: str,
    texts: list[str],
    device: torch.device,
    audio_dir: Path,
) -> list[float]:
    """
    指定テキスト群の MOS を推定する。

    Args:
        tts_model (TTSModel): TTS モデル
        utmos_model (UTMOSv2Model): UTMOSv2 モデル
        model_label (str): ログに出すモデル識別子
        texts (list[str]): 評価に使うテキスト
        device (torch.device): UTMOSv2 推論デバイス
        audio_dir (Path): MOS 推論用の一時音声ディレクトリ

    Returns:
        list[float]: MOS の一覧
    """

    with torch.inference_mode():
        _clear_audio_dir(audio_dir)
        audio_paths = _generate_audio_files(tts_model, texts, audio_dir)
        scores = _predict_mos_from_audio_dir(
            utmos_model,
            audio_dir,
            audio_paths,
            device,
        )
        for index, score in enumerate(scores):
            logger.info(f"checkpoint: {model_label} text_index: {index} score: {score}")
        return scores


def _append_mean_scores(
    results: list[tuple[str, int, list[float]]],
) -> list[tuple[str, int, list[float]]]:
    """
    MOS の平均値を追加する。

    Args:
        results (list[tuple[str, int, list[float]]]): MOS 推論結果

    Returns:
        list[tuple[str, int, list[float]]]: 平均を追加した結果
    """

    results_with_mean: list[tuple[str, int, list[float]]] = []
    for model_file, step_count, scores in results:
        mean_score = float(np.mean(scores))
        results_with_mean.append((model_file, step_count, scores + [mean_score]))
    return results_with_mean


def _write_results_csv(
    output_path: Path,
    results: list[tuple[str, int, list[float]]],
    texts: list[str],
) -> None:
    """
    MOS 推論結果を CSV に書き出す。

    Args:
        output_path (Path): 出力先
        results (list[tuple[str, int, list[float]]]): MOS 推論結果
        texts (list[str]): 評価に使ったテキスト
    """

    with open(output_path, "w", encoding="utf_8_sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model_path", "step"] + texts + ["mean"])
        for model_file, step_count, scores in results:
            writer.writerow([model_file, step_count] + scores)


def _save_plot(
    output_path: Path,
    results: list[tuple[str, int, list[float]]],
) -> None:
    """
    MOS の推移を可視化したグラフを保存する。

    Args:
        output_path (Path): 出力先
        results (list[tuple[str, int, list[float]]]): MOS 推論結果
    """

    steps: list[int] = []
    mos_values: list[list[float]] = []
    for _, step_count, scores in results:
        steps.append(step_count)
        mos_values.append(scores)

    data_frame = pd.DataFrame(mos_values, index=pd.Index(steps))
    data_frame = data_frame.sort_index()

    plt.figure(figsize=(10, 5))

    for column_index in range(len(data_frame.columns) - 1):
        plt.plot(
            data_frame.index,
            data_frame.iloc[:, column_index],
            label=f"MOS{column_index + 1}",
        )

    plt.plot(
        data_frame.index,
        data_frame.iloc[:, -1],
        label="Mean",
        color="black",
        linewidth=2,
    )

    plt.title("TTS Model Naturalness MOS")
    plt.xlabel("Step Count")
    plt.ylabel("MOS")

    max_step = int(max(data_frame.index))
    tick_positions = np.arange(0, max_step + 1000, 2000)
    plt.xticks(
        ticks=tick_positions,
        labels=[f"{int(step / 1000)}" for step in tick_positions],
    )

    plt.grid(True, axis="x")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def main() -> None:
    """
    UTMOSv2 を使って複数チェックポイントの MOS を評価する。
    """

    warnings.filterwarnings("ignore")

    args = _parse_args()
    model_name: str = args.model
    tts_device: str = args.device
    utmos_device = torch.device(args.device)

    MOS_RESULT_DIR.mkdir(exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    model_files = _resolve_model_files(model_name)
    logger.info(f"There are {len(model_files)} models.")

    logger.info("Loading UTMOSv2 model.")
    utmos_model = utmosv2.create_model(pretrained=True, device=utmos_device)

    results: list[tuple[str, int, list[float]]] = []

    for model_file in tqdm(model_files, dynamic_ncols=True):
        step_count = _extract_step_count(model_file.name)
        if step_count is None:
            logger.warning(f"Step count not found in {model_file.name}, so skip it.")
            continue

        tts_model = _create_tts_model(model_file, tts_device)
        audio_dir = _prepare_audio_dir(model_file)
        scores = _evaluate_model(
            tts_model,
            utmos_model,
            model_file.name,
            TEST_TEXTS,
            utmos_device,
            audio_dir,
        )
        results.append((model_file.name, step_count, scores))
        del tts_model

    if len(results) == 0:
        raise ValueError("No models were evaluated.")

    logger.success("All models have been evaluated:")

    results_with_mean = _append_mean_scores(results)
    results_with_mean = sorted(
        results_with_mean,
        key=lambda item: item[2][-1],
        reverse=True,
    )

    for model_file, step_count, scores in results_with_mean:
        logger.info(f"{model_file}: {scores[-1]}")

    csv_path = MOS_RESULT_DIR / f"mos_{model_name}.csv"
    _write_results_csv(csv_path, results_with_mean, TEST_TEXTS)
    logger.info(f"{csv_path.name} has been saved.")

    png_path = MOS_RESULT_DIR / f"mos_{model_name}.png"
    _save_plot(png_path, results_with_mean)


if __name__ == "__main__":
    main()
