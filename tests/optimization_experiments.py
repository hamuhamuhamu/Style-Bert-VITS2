"""
Usage: .venv/bin/python -m tests.optimization_experiments [--device cuda:0] [--model-name 10-hinataanna] [--num-runs 3]

Style-Bert-VITS2 の推論高速化施策を比較検証するためのスクリプト。

- 既存モデルの追加学習は行わず、推論グラフのみを変更する
- それぞれの施策について、変更前 / 変更後で速度・VRAM 使用量・波形レベルの差分を計測する
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import attentions, models, models_jp_extra
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


SEED_BASELINE = 1234


@dataclass
class InferenceMetrics:
    """単一回の推論で計測された指標を保持するデータクラス。

    Attributes:
        elapsed_sec (float): 推論に要した経過時間 (秒)
        max_alloc_bytes (int): 推論中に割り当てられた CUDA メモリのピーク値 (バイト)
        max_reserved_bytes (int): 推論中に予約された CUDA メモリのピーク値 (バイト)
        max_abs_diff (float | None): 変更前後の波形の最大絶対誤差 (比較対象がある場合のみ)
        mean_abs_diff (float | None): 変更前後の波形の平均絶対誤差 (比較対象がある場合のみ)
    """

    elapsed_sec: float
    max_alloc_bytes: int
    max_reserved_bytes: int
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None


def set_global_seed(seed: int) -> None:
    """乱数のシードを固定し、推論をできるだけ決定的にする。

    Args:
        seed (int): 使用するシード値
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() is True:
        torch.cuda.manual_seed_all(seed)
    # CuDNN の決定論的挙動を優先することで、波形比較時のばらつきを抑える
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_target_model(
    model_root_dir: str,
    model_name: str | None,
    device: str,
) -> tuple[TTSModelHolder, TTSModel]:
    """検証に使用する TTSModel を選択しロードする。

    Args:
        model_root_dir (str): モデル資産が格納されているディレクトリパス
        model_name (str | None): 使用したいモデル名。None の場合は JP-Extra モデルの先頭を選択
        device (str): 使用するデバイス (例: \"cuda:0\")

    Returns:
        tuple[TTSModelHolder, TTSModel]: モデルホルダーとロード済み TTSModel
    """

    holder = TTSModelHolder(
        BASE_DIR / model_root_dir,
        device=device,
        onnx_providers=[
            (
                "CPUExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        ],
        use_fp16=True,
    )
    if len(holder.models_info) == 0:
        raise RuntimeError("No models found under model_assets directory.")

    if model_name is None:
        # JP-Extra モデルを優先して選択
        target_info = None
        for info in holder.models_info:
            tmp_model = holder.get_model(
                info.name,
                holder.model_files_dict[info.name][0].as_posix(),
            )
            if str(tmp_model.hyper_parameters.version).endswith("JP-Extra"):
                target_info = info
                break
        if target_info is None:
            target_info = holder.models_info[0]
    else:
        # 指定された名前と一致するモデルを探す
        target_info = None
        for info in holder.models_info:
            if info.name == model_name:
                target_info = info
                break
        if target_info is None:
            raise ValueError(f"Model `{model_name}` is not found in model_assets.")

    model_files = holder.model_files_dict[target_info.name]
    if len(model_files) == 0:
        raise RuntimeError(f"No model files found for `{target_info.name}`.")

    model = holder.get_model(target_info.name, model_files[0].as_posix())
    model.load()
    logger.info(
        f"Using model `{target_info.name}` (version: {model.hyper_parameters.version})"
    )
    return holder, model


def choose_fixed_text() -> str:
    """検証用に、そこそこの長さを持つ固定テキストを返す。

    Returns:
        str: 音声合成に使用するテキスト
    """

    return (
        "音声合成は、機械学習を活用してテキストから人の声を生成する技術です。"
        "このテキストは、モデルの推論速度とメモリ消費を比較するためのベンチマークとして使用されます。"
    )


def get_default_style_and_speaker(model: TTSModel) -> tuple[str, int]:
    """モデルから既定のスタイル名と話者 ID を取得する。

    Args:
        model (TTSModel): 対象の音声合成モデル

    Returns:
        tuple[str, int]: (スタイル名, 話者 ID)
    """

    style_names = list(model.style2id.keys())
    if len(style_names) == 0:
        raise RuntimeError("No styles found in the target model.")
    speakers = list(model.spk2id.values())
    if len(speakers) == 0:
        raise RuntimeError("No speakers found in the target model.")
    return style_names[0], speakers[0]


def run_single_inference(
    model: TTSModel,
    text: str,
    style_name: str,
    speaker_id: int,
    device: str,
) -> tuple[InferenceMetrics, NDArray[Any]]:
    """単発の推論を実行し、時間・メモリ・波形を収集する。

    Args:
        model (TTSModel): 使用する TTSModel
        text (str): 合成するテキスト
        style_name (str): 使用するスタイル名
        speaker_id (int): 使用する話者 ID
        device (str): 使用するデバイス (例: \"cuda:0\")

    Returns:
        tuple[InferenceMetrics, NDArray[Any]]: 計測結果と生成波形
    """

    if torch.cuda.is_available() is True and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    sample_rate, audio = model.infer(
        text=text,
        language=Languages.JP,
        speaker_id=speaker_id,
        sdp_ratio=0.4,
        noise=0.6,
        noise_w=0.8,
        length=1.0,
        style=style_name,
        style_weight=2.0,
    )
    if torch.cuda.is_available() is True and device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if torch.cuda.is_available() is True and device.startswith("cuda"):
        max_alloc = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
    else:
        max_alloc = 0
        max_reserved = 0

    # wavfile.write と同様に 16bit に正規化してから差分を計算する方が比較しやすい
    audio_int16 = model.convert_to_16_bit_wav(audio)
    _ = sample_rate  # 型チェック用の未使用抑制

    metrics = InferenceMetrics(
        elapsed_sec=elapsed,
        max_alloc_bytes=max_alloc,
        max_reserved_bytes=max_reserved,
    )
    return metrics, audio_int16


def summarize_metrics(
    name: str,
    metrics_list: list[InferenceMetrics],
) -> None:
    """複数回の計測結果を集計し、標準出力に要約を表示する。

    Args:
        name (str): 施策名または条件名
        metrics_list (list[InferenceMetrics]): 計測結果のリスト
    """

    elapsed = np.array([m.elapsed_sec for m in metrics_list], dtype=np.float64)
    alloc = np.array([m.max_alloc_bytes for m in metrics_list], dtype=np.float64)
    reserved = np.array([m.max_reserved_bytes for m in metrics_list], dtype=np.float64)

    logger.info(
        f"[{name}] elapsed_sec: mean: {elapsed.mean():.4f}s, std: {elapsed.std():.4f}s"
    )
    logger.info(
        f"[{name}] max_alloc_bytes: mean: {alloc.mean():.0f}, min: {alloc.min():.0f}, max: {alloc.max():.0f}"
    )
    logger.info(
        f"[{name}] max_reserved_bytes: mean: {reserved.mean():.0f}, min: {reserved.min():.0f}, max: {reserved.max():.0f}"
    )

    diffs = [m for m in metrics_list if m.max_abs_diff is not None]
    if len(diffs) > 0:
        max_abs = np.array(
            [m.max_abs_diff for m in diffs if m.max_abs_diff is not None],
            dtype=np.float64,
        )
        mean_abs = np.array(
            [m.mean_abs_diff for m in diffs if m.mean_abs_diff is not None],
            dtype=np.float64,
        )
        logger.info(
            f"[{name}] waveform max_abs_diff: mean: {max_abs.mean():.6f}, "
            f"max: {max_abs.max():.6f}"
        )
        logger.info(
            f"[{name}] waveform mean_abs_diff: mean: {mean_abs.mean():.6f}, "
            f"max: {mean_abs.max():.6f}"
        )


def compute_waveform_diff(
    base: NDArray[Any],
    other: NDArray[Any],
) -> tuple[float, float]:
    """2 つの波形の差分 (最大絶対値・平均絶対値) を計算する。

    Args:
        base (NDArray[Any]): 基準となる波形
        other (NDArray[Any]): 比較対象の波形

    Returns:
        tuple[float, float]: (max_abs_diff, mean_abs_diff)
    """

    min_len = min(base.shape[0], other.shape[0])
    if min_len == 0:
        return 0.0, 0.0
    diff = base[:min_len].astype(np.float64) - other[:min_len].astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    return max_abs, mean_abs


def run_experiment_attention_fp16(
    model: TTSModel,
    device: str,
    num_runs: int,
) -> None:
    """Encoder の Multi-Head Attention をより積極的に FP16 化する施策の比較実験を行う。

    - ベースライン: 既存実装 (相対位置エンコーディング部分のみ FP16)
    - 施策: MultiHeadAttention.attention 全体を autocast ブロック内で実行

    Args:
        model (TTSModel): 検証対象のモデル
        device (str): 使用するデバイス
        num_runs (int): 各条件での繰り返し回数
    """

    logger.info("Running experiment: attention full FP16 under autocast")

    # 検証対象の MultiHeadAttention.attention を保存しておく
    original_attention = attentions.MultiHeadAttention.attention

    def attention_full_amp(
        self: attentions.MultiHeadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_fp16: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MultiHeadAttention.attention を autocast で全体 FP16 化した版。"""

        if use_fp16 is True and query.is_cuda:
            original_dtype = query.dtype
            device_type = query.device.type
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=True,
            ):
                # 元の attention 実装を FP16 環境下で呼び出す
                output, attn = original_attention(
                    self,
                    query,
                    key,
                    value,
                    mask=mask,
                    use_fp16=False,
                )
            return output.to(dtype=original_dtype), attn.to(dtype=original_dtype)
        return original_attention(self, query, key, value, mask=mask, use_fp16=use_fp16)

    text = choose_fixed_text()
    style_name, speaker_id = get_default_style_and_speaker(model)

    # 事前に 1 回だけウォームアップして BERT ロードなどの一過性オーバーヘッドを除外する
    set_global_seed(SEED_BASELINE - 1)
    _warmup_metrics, _warmup_wave = run_single_inference(
        model,
        text,
        style_name,
        speaker_id,
        device,
    )
    _ = _warmup_metrics
    _ = _warmup_wave

    # ベースライン
    baseline_metrics: list[InferenceMetrics] = []
    baseline_waves: list[NDArray[Any]] = []
    for i in range(num_runs):
        set_global_seed(SEED_BASELINE + i)
        metrics, wave = run_single_inference(
            model, text, style_name, speaker_id, device
        )
        baseline_metrics.append(metrics)
        baseline_waves.append(wave)
    summarize_metrics("attention_baseline", baseline_metrics)

    # 施策適用
    attentions.MultiHeadAttention.attention = attention_full_amp  # type: ignore[assignment]
    optimized_metrics: list[InferenceMetrics] = []
    try:
        for i in range(num_runs):
            set_global_seed(SEED_BASELINE + i)
            metrics, wave = run_single_inference(
                model,
                text,
                style_name,
                speaker_id,
                device,
            )
            max_abs_diff, mean_abs_diff = compute_waveform_diff(
                baseline_waves[i],
                wave,
            )
            metrics.max_abs_diff = max_abs_diff
            metrics.mean_abs_diff = mean_abs_diff
            optimized_metrics.append(metrics)
        summarize_metrics("attention_full_fp16", optimized_metrics)
    except Exception as ex:
        # Flow 内の spline などが数値的に不安定になる場合があるため、
        # その場合は施策をスキップした扱いにしてログだけ残す
        logger.warning(
            "Attention full FP16 experiment failed due to exception. "
            "This optimization seems numerically unstable for this model.",
            exc_info=ex,
        )
    finally:
        # 元に戻す
        attentions.MultiHeadAttention.attention = original_attention  # type: ignore[assignment]


def run_experiment_attention_cache(
    model: TTSModel,
    device: str,
    num_runs: int,
) -> None:
    """MultiHeadAttention.forward における self.attn のキャッシュ有無を比較する実験を行う。

    - ベースライン: 既存実装 (最後の注意重みを self.attn に保存)
    - 施策: 推論時は self.attn を保存せず、その場限りで捨てる

    Args:
        model (TTSModel): 検証対象のモデル
        device (str): 使用するデバイス
        num_runs (int): 各条件での繰り返し回数
    """

    logger.info("Running experiment: attention cache (self.attn) on/off")

    original_forward = attentions.MultiHeadAttention.forward

    def forward_no_cache(
        self: attentions.MultiHeadAttention,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_fp16: bool = False,
    ) -> torch.Tensor:
        """推論時に self.attn を保持しない MultiHeadAttention.forward。"""

        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        # self.attn を保持せず、その場限りで捨てる
        output, _ = self.attention(q, k, v, mask=attn_mask, use_fp16=use_fp16)
        output = self.conv_o(output)
        return output

    text = choose_fixed_text()
    style_name, speaker_id = get_default_style_and_speaker(model)

    # 事前に 1 回だけウォームアップして BERT ロードなどの一過性オーバーヘッドを除外する
    set_global_seed(SEED_BASELINE + 99)
    _warmup_metrics, _warmup_wave = run_single_inference(
        model,
        text,
        style_name,
        speaker_id,
        device,
    )
    _ = _warmup_metrics
    _ = _warmup_wave

    # ベースライン
    baseline_metrics: list[InferenceMetrics] = []
    baseline_waves: list[NDArray[Any]] = []
    for i in range(num_runs):
        set_global_seed(SEED_BASELINE + 100 + i)
        metrics, wave = run_single_inference(
            model, text, style_name, speaker_id, device
        )
        baseline_metrics.append(metrics)
        baseline_waves.append(wave)
    summarize_metrics("attn_cache_baseline", baseline_metrics)

    # 施策適用
    attentions.MultiHeadAttention.forward = forward_no_cache  # type: ignore[assignment]
    optimized_metrics: list[InferenceMetrics] = []
    for i in range(num_runs):
        set_global_seed(SEED_BASELINE + 100 + i)
        metrics, wave = run_single_inference(
            model, text, style_name, speaker_id, device
        )
        max_abs_diff, mean_abs_diff = compute_waveform_diff(baseline_waves[i], wave)
        metrics.max_abs_diff = max_abs_diff
        metrics.mean_abs_diff = mean_abs_diff
        optimized_metrics.append(metrics)
    summarize_metrics("attn_cache_disabled", optimized_metrics)

    # 元に戻す
    attentions.MultiHeadAttention.forward = original_forward  # type: ignore[assignment]


def patch_flow_chunk_forward() -> tuple[
    Any,
    Any,
]:
    """TransformerCouplingBlock.forward のチャンクサイズをインスタンス属性から取得するようにパッチする。

    Returns:
        tuple[Any, Any]: (models.TransformerCouplingBlock の元の forward, models_jp_extra 版の元の forward)
    """

    original_forward_base = models.TransformerCouplingBlock.forward
    original_forward_extra = models_jp_extra.TransformerCouplingBlock.forward

    def forward_with_custom_chunk(
        self: models.TransformerCouplingBlock,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """チャンクサイズをインスタンスごとの属性から取得する forward 実装。"""

        seq_len = x.size(2)
        chunk_size = getattr(self, "_chunk_size", 1024)
        if seq_len > chunk_size:
            return self._chunked_forward(x, x_mask, g, reverse, chunk_size)  # pyright: ignore[reportPrivateUsage]
        return self._standard_forward(x, x_mask, g, reverse)  # pyright: ignore[reportPrivateUsage]

    models.TransformerCouplingBlock.forward = forward_with_custom_chunk  # type: ignore[assignment]
    models_jp_extra.TransformerCouplingBlock.forward = forward_with_custom_chunk  # type: ignore[assignment]
    return original_forward_base, original_forward_extra


def set_flow_chunk_size(net_g: torch.nn.Module, chunk_size: int) -> None:
    """ネットワーク内の Flow に対してチャンクサイズを設定する。

    Args:
        net_g (torch.nn.Module): SynthesizerTrn / SynthesizerTrnJPExtra インスタンス
        chunk_size (int): 使用するチャンクサイズ
    """

    if not hasattr(net_g, "flow"):
        return
    flow_module = getattr(net_g, "flow")
    if isinstance(
        flow_module,
        (models.TransformerCouplingBlock, models_jp_extra.TransformerCouplingBlock),
    ):
        setattr(flow_module, "_chunk_size", chunk_size)


def run_experiment_flow_chunk(
    model: TTSModel,
    device: str,
    num_runs: int,
) -> None:
    """Flow (TransformerCouplingBlock) のチャンクサイズを変更する施策の比較実験を行う。

    - ベースライン: CHUNK_SIZE=1024
    - 施策: CHUNK_SIZE を 768 / 512 などに変更し、VRAM と速度のトレードオフを見る

    Args:
        model (TTSModel): 検証対象のモデル
        device (str): 使用するデバイス
        num_runs (int): 各条件での繰り返し回数
    """

    logger.info("Running experiment: flow chunk size tuning")

    net_g = model.net_g
    if net_g is None:
        raise RuntimeError("net_g is not loaded in the target model.")

    original_forward_base, original_forward_extra = patch_flow_chunk_forward()

    text = choose_fixed_text()
    style_name, speaker_id = get_default_style_and_speaker(model)

    # 事前に 1 回だけウォームアップして BERT ロードなどの一過性オーバーヘッドを除外する
    set_global_seed(SEED_BASELINE + 199)
    _warmup_metrics, _warmup_wave = run_single_inference(
        model,
        text,
        style_name,
        speaker_id,
        device,
    )
    _ = _warmup_metrics
    _ = _warmup_wave

    # 比較するチャンクサイズの候補
    chunk_sizes = [1024, 768, 512]
    baseline_wave: NDArray[Any] | None = None

    for idx, chunk_size in enumerate(chunk_sizes):
        set_flow_chunk_size(net_g, chunk_size)
        metrics_list: list[InferenceMetrics] = []
        waves: list[NDArray[Any]] = []
        for i in range(num_runs):
            set_global_seed(SEED_BASELINE + 200 + i)
            metrics, wave = run_single_inference(
                model,
                text,
                style_name,
                speaker_id,
                device,
            )
            metrics_list.append(metrics)
            waves.append(wave)

        # 最初のチャンクサイズを基準として波形差分を計算する
        if idx == 0:
            baseline_wave = waves[0]
        if baseline_wave is not None:
            for wave_idx, m in enumerate(metrics_list):
                max_abs_diff, mean_abs_diff = compute_waveform_diff(
                    baseline_wave,
                    waves[wave_idx],
                )
                m.max_abs_diff = max_abs_diff
                m.mean_abs_diff = mean_abs_diff

        summarize_metrics(f"flow_chunk_{chunk_size}", metrics_list)

    # 元に戻す
    models.TransformerCouplingBlock.forward = original_forward_base  # type: ignore[assignment]
    models_jp_extra.TransformerCouplingBlock.forward = original_forward_extra  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする。

    Returns:
        argparse.Namespace: パース済み引数
    """

    parser = argparse.ArgumentParser(
        description="Run optimization experiments for Style-Bert-VITS2 models.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Target model name under model_assets. If omitted, the first JP-Extra model is used.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per condition.",
    )
    parser.add_argument(
        "--skip-attention-fp16",
        action="store_true",
        help="Skip attention full FP16 experiment.",
    )
    parser.add_argument(
        "--skip-attention-cache",
        action="store_true",
        help="Skip attention cache (self.attn) experiment.",
    )
    parser.add_argument(
        "--skip-flow-chunk",
        action="store_true",
        help="Skip flow chunk size experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """メインエントリポイント。指定された条件で各実験を実行する。"""

    args = parse_args()

    holder, model = select_target_model(
        model_root_dir="model_assets",
        model_name=args.model_name,
        device=args.device,
    )
    _ = holder  # 型チェック用の未使用抑制

    try:
        if args.skip_attention_fp16 is False:
            run_experiment_attention_fp16(model, args.device, args.num_runs)

        if args.skip_attention_cache is False:
            run_experiment_attention_cache(model, args.device, args.num_runs)

        if args.skip_flow_chunk is False:
            run_experiment_flow_chunk(model, args.device, args.num_runs)
    finally:
        # モデルを明示的にアンロードしておく
        model.unload()


if __name__ == "__main__":
    main()
