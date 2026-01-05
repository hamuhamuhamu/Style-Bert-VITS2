import torch
import torchaudio
from transformers import AutoModel


def feature_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    識別器の特徴マップ間の L1 損失を計算する。

    Args:
        fmap_r (list[list[torch.Tensor]]): 実データの特徴マップリスト
        fmap_g (list[list[torch.Tensor]]): 生成データの特徴マップリスト

    Returns:
        torch.Tensor: 特徴損失
    """

    loss = torch.tensor(0.0, device=fmap_r[0][0].device)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[float], list[float]]:
    """
    識別器の損失を計算する。

    Args:
        disc_real_outputs (list[torch.Tensor]): 実データに対する識別器出力
        disc_generated_outputs (list[torch.Tensor]): 生成データに対する識別器出力

    Returns:
        tuple[torch.Tensor, list[float], list[float]]: 合計損失、実データ損失のリスト、生成データ損失のリスト
    """

    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    r_losses: list[float] = []
    g_losses: list[float] = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    生成器の敵対的損失を計算する。

    Args:
        disc_outputs (list[torch.Tensor]): 生成データに対する識初器出力

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]]: 合計損失、各識別器の損失リスト
    """

    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    gen_losses: list[torch.Tensor] = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    KL ダイバージェンス損失を計算する。

    Args:
        z_p (torch.Tensor): 事前分布からのサンプル [b, h, t_t]
        logs_q (torch.Tensor): 事後分布の対数標準偏差 [b, h, t_t]
        m_p (torch.Tensor): 事前分布の平均 [b, h, t_t]
        logs_p (torch.Tensor): 事前分布の対数標準偏差 [b, h, t_t]
        z_mask (torch.Tensor): マスク

    Returns:
        torch.Tensor: KL 損失
    """

    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


class WavLMLoss(torch.nn.Module):
    """
    WavLM を使用した知覚的音声損失。
    WavLM の隠れ層の特徴量を使用して、生成音声と実音声の知覚的な類似性を測定する。
    """

    def __init__(
        self,
        model: str,
        wd: torch.nn.Module,
        model_sr: int,
        slm_sr: int = 16000,
    ):
        """
        WavLMLoss を初期化する。

        Args:
            model (str): WavLM モデルのパスまたは HuggingFace モデル ID
            wd (torch.nn.Module): 波形識別器モジュール
            model_sr (int): モデルのサンプリングレート
            slm_sr (int): SLM のサンプリングレート（デフォルト: 16000）
        """

        super().__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(
        self,
        wav: torch.Tensor,
        y_rec: torch.Tensor,
    ) -> torch.Tensor:
        """
        特徴損失を計算する。

        Args:
            wav (torch.Tensor): 元の音声波形
            y_rec (torch.Tensor): 再構成された音声波形

        Returns:
            torch.Tensor: 特徴損失
        """

        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states

        floss = torch.tensor(0.0, device=wav.device)
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec: torch.Tensor) -> torch.Tensor:
        """
        生成器の損失を計算する。

        Args:
            y_rec (torch.Tensor): 再構成された音声波形

        Returns:
            torch.Tensor: 生成器損失
        """

        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(
        self,
        wav: torch.Tensor,
        y_rec: torch.Tensor,
    ) -> torch.Tensor:
        """
        識別器の損失を計算する。

        Args:
            wav (torch.Tensor): 元の音声波形
            y_rec (torch.Tensor): 再構成された音声波形

        Returns:
            torch.Tensor: 識別器損失
        """

        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        識別器のフォワードパスのみを実行する。

        Args:
            wav (torch.Tensor): 音声波形

        Returns:
            torch.Tensor: 識別器の出力
        """

        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs
