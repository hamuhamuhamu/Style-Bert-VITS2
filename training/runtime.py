import copy
from dataclasses import dataclass
from typing import Any, cast

import torch

from style_bert_vits2.logging import logger


@dataclass(frozen=True)
class TrainRuntimeConfig:
    """
    学習スクリプトでのみ使用されるランタイム設定。

    Args:
        model_name: config.json 由来のモデル名
        model_dir: チェックポイントの保存先ディレクトリ
        out_dir: 推論用モデルの出力先ディレクトリ
        dataset_path: データセットのルートディレクトリ
        keep_ckpts: 保持するチェックポイント数
        repo_id: Hugging Face へのバックアップ用リポジトリ ID
        speedup: 速度優先モードの有効化フラグ
        spec_cache: Spectrogram キャッシュの有効化フラグ
    """

    model_name: str
    model_dir: str
    out_dir: str
    dataset_path: str
    keep_ckpts: int
    repo_id: str | None
    speedup: bool
    spec_cache: bool = True


class GradientMonitor:
    """
    勾配ノルムを監視し、異常検出時に学習率を自動調整するクラス。

    Generator の勾配ノルム (grad_norm_g) の指数移動平均 (EMA) を追跡し、
    以下の条件で学習率を自動的に減少させる。

    1. EMA 閾値超過: smoothed（EMA）値が閾値を超え続けている場合
       - TensorBoard の smoothed グラフが高止まりしている状態に対応
       - 瞬間的なスパイクは EMA に吸収されるためスルーされる
    2. 急激なスパイク検出: 現在の grad_norm_g が EMA を大幅に上回った場合
       - 学習の破綻を防ぐための安全弁

    機械学習の専門知識がないユーザーでも、勾配関連の問題に対して
    手動介入なしで安定した学習を行えるようにすることが目的。
    """

    def __init__(
        self,
        ema_alpha: float = 0.4,  # EMA 平滑化係数 (smoothing=0.6 相当)
        spike_threshold: float = 3.0,  # スパイク検出: 現在値 > 3倍 EMA でトリガー
        ema_high_threshold: float = 400.0,  # EMA (smoothed) がこの値を超えたら介入
        lr_decay_factor: float = 0.5,  # トリガー時に学習率を 50% に減少
        min_lr: float = 1e-6,  # 最小学習率
        cooldown_steps: int = 2000,  # 連続調整間のステップ数
        warmup_steps: int = 500,  # 学習初期の猶予ステップ数
    ):
        """
        GradientMonitor を初期化する。

        Args:
            ema_alpha: EMA 計算の平滑化係数 (0 < alpha <= 1)。
                値が大きいほど直近の観測値を重視する。
                0.4 は TensorBoard の smoothing=0.6 に相当。
            spike_threshold: スパイク検出の倍率。
                現在の grad_norm > spike_threshold * EMA の場合にスパイクと判定。
            ema_high_threshold: EMA (smoothed 値) が「高い」と判定する閾値。
                EMA がこの値を超えたら学習率調整の対象となる。
            lr_decay_factor: 調整時に学習率に乗じる係数。
            min_lr: 過度な減少を防ぐための最小学習率。
            cooldown_steps: 連続した学習率調整間の最小ステップ数。
            warmup_steps: 学習初期に介入しない猶予ステップ数。
                序盤は勾配が不安定になりやすいため。
        """

        self.ema_alpha = ema_alpha
        self.spike_threshold = spike_threshold
        self.ema_high_threshold = ema_high_threshold
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.cooldown_steps = cooldown_steps
        self.warmup_steps = warmup_steps

        # 内部状態
        self.ema: float | None = None
        self.steps_since_adjustment: int = 0
        self.total_steps: int = 0
        self.adjustment_history: list[tuple[int, float, float, str]] = []
        self.is_enabled: bool = True

    def update(
        self,
        grad_norm_g: float,
        optim_g: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.LRScheduler,
        global_step: int,
    ) -> bool:
        """
        新しい勾配ノルムの観測値でモニターを更新し、必要に応じて学習率を調整する。

        Args:
            grad_norm_g: 現在の Generator 勾配ノルム値。
            optim_g: 学習率を調整する対象の Generator オプティマイザ。
            scheduler_g: Generator の学習率スケジューラ。
                調整時に base_lrs も更新することでエポック境界での上書きを防ぐ。
            global_step: ログ出力用の現在のグローバル学習ステップ。

        Returns:
            学習率が調整された場合は True、そうでなければ False。
        """

        if not self.is_enabled:
            return False

        self.total_steps += 1
        self.steps_since_adjustment += 1

        # grad_norm が無効値 (NaN または Inf) の場合はスキップ
        if not torch.isfinite(torch.tensor(grad_norm_g)):
            logger.warning(
                f"[GradientMonitor] Invalid grad_norm_g detected at step {global_step}: {grad_norm_g}. "
                f"Skipping update. Training may be unstable."
            )
            return False

        # 初回ステップで EMA を初期化
        if self.ema is None:
            self.ema = grad_norm_g
            return False

        # EMA を更新
        # EMA 更新式: new_ema = alpha * current + (1 - alpha) * old_ema
        old_ema = self.ema
        self.ema = self.ema_alpha * grad_norm_g + (1 - self.ema_alpha) * self.ema

        # warmup 期間中は介入しない（学習初期は勾配が不安定になりやすいため）
        if self.total_steps < self.warmup_steps:
            return False

        # クールダウン期間中は介入しない
        is_in_cooldown = self.steps_since_adjustment < self.cooldown_steps
        if is_in_cooldown:
            return False

        # 学習率調整が必要かどうかを判定
        is_adjustment_needed = False
        adjustment_reason = ""

        # 1. EMA（smoothed 値）が閾値を超えている場合
        # これは「TensorBoard の smoothed グラフが高止まりしている状態」に対応
        # 瞬間的なスパイクは EMA に吸収されるためスルーされる
        if self.ema > self.ema_high_threshold:
            is_adjustment_needed = True
            adjustment_reason = (
                f"EMA (smoothed) exceeded threshold: EMA: {self.ema:.2f} > {self.ema_high_threshold:.0f}. "
                f"Current grad_norm_g: {grad_norm_g:.2f}."
            )

        # 2. 急激なスパイク検出（安全弁）
        # 現在の grad_norm が EMA を大幅に上回った場合
        # これは学習の破綻を防ぐための安全弁
        elif grad_norm_g > self.spike_threshold * old_ema and old_ema > 0:
            is_adjustment_needed = True
            adjustment_reason = (
                f"Spike detected: grad_norm_g: {grad_norm_g:.2f} is {grad_norm_g / old_ema:.1f}x "
                f"higher than EMA: {old_ema:.2f}."
            )

        if is_adjustment_needed:
            old_lr = optim_g.param_groups[0]["lr"]
            new_lr = max(old_lr * self.lr_decay_factor, self.min_lr)

            if new_lr < old_lr:
                for param_group in optim_g.param_groups:
                    param_group["lr"] = new_lr
                # scheduler の base_lrs も更新して次の step() で上書きされないようにする
                if hasattr(scheduler_g, "base_lrs"):
                    scheduler_g.base_lrs = [new_lr for _ in scheduler_g.base_lrs]

                self.adjustment_history.append(
                    (global_step, old_lr, new_lr, adjustment_reason)
                )
                self.steps_since_adjustment = 0

                logger.warning(
                    f"[GradientMonitor] Learning rate adjusted at step {global_step}: "
                    f"{old_lr:.6f} -> {new_lr:.6f}. Reason: {adjustment_reason}"
                )
                return True

        return False

    def state_dict(self) -> dict[str, Any]:
        """
        内部状態を辞書として取得する。

        Returns:
            GradientMonitor の状態辞書
        """

        return {
            "ema": self.ema,
            "steps_since_adjustment": self.steps_since_adjustment,
            "total_steps": self.total_steps,
            "adjustment_history": self.adjustment_history,
            "is_enabled": self.is_enabled,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        状態辞書から内部状態を復元する。

        Args:
            state_dict: 以前のチェックポイントからの状態変数を含む辞書。
        """

        self.ema = state_dict.get("ema", None)
        self.steps_since_adjustment = state_dict.get("steps_since_adjustment", 0)
        self.total_steps = state_dict.get("total_steps", 0)
        self.adjustment_history = state_dict.get("adjustment_history", [])
        self.is_enabled = state_dict.get("is_enabled", True)


class EMAModel:
    """
    モデル重みの指数移動平均 (Exponential Moving Average) を管理するクラス。

    学習中のモデル重みは勾配更新により振動するが、EMA はその平滑化されたバージョンを保持する。
    推論時に EMA 重みを使用することで、より安定した出力が得られる傾向がある。

    使い方:
        1. 学習開始時に EMAModel を初期化
        2. 各オプティマイザステップ後に update() を呼び出し
        3. チェックポイント保存時に get_ema_model() で EMA モデルを取得して保存

    Args:
        model: EMA を適用する対象モデル (通常は Generator)
        decay: EMA の減衰率 (0.999 が一般的)。
            値が大きいほど過去の重みを重視し、より滑らかになる。
        device: EMA モデルを配置するデバイス
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,  # 減衰率: 0.999 が一般的な値
        device: torch.device | None = None,
    ):
        self.decay = decay
        self.device = device

        # モデルの深いコピーを作成して EMA 用の重みを保持
        # DDP の場合は .module を使用して内部モデルを取得
        if hasattr(model, "module"):
            self.ema_model = cast(torch.nn.Module, copy.deepcopy(model.module))
        else:
            self.ema_model = copy.deepcopy(model)

        # EMA モデルは学習しないので勾配計算を無効化
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        if device is not None:
            self.ema_model.to(device)

    @torch.no_grad()  # type: ignore[misc]
    def update(self, model: torch.nn.Module) -> None:
        """
        現在のモデル重みで EMA を更新する。

        各オプティマイザステップ後に呼び出すこと。
        EMA 更新式: ema_weight = decay * ema_weight + (1 - decay) * current_weight

        Args:
            model: 現在の学習中モデル
        """

        # DDP の場合は .module を使用
        source_model = cast(
            torch.nn.Module, model.module if hasattr(model, "module") else model
        )

        for ema_param, param in zip(
            self.ema_model.parameters(),
            source_model.parameters(),
        ):
            # EMA 更新: ema = decay * ema + (1 - decay) * current
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def get_ema_model(self) -> torch.nn.Module:
        """
        EMA モデルを取得する。

        チェックポイント保存時や推論時に使用。

        Returns:
            EMA 重みを持つモデル
        """

        return self.ema_model

    def state_dict(self) -> dict[str, Any]:
        """
        EMA モデルの状態辞書を取得する。

        Returns:
            EMA モデルの状態辞書
        """

        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        EMA モデルの状態を復元する。

        Args:
            state_dict: 以前のチェックポイントからの状態変数を含む辞書。
        """

        self.ema_model.load_state_dict(state_dict)
