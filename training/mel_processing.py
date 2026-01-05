import warnings

import torch
from librosa.filters import mel as librosa_mel_fn


# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore")
MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(
    x: torch.Tensor,
    C: float = 1.0,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    """
    ダイナミックレンジ圧縮を適用する。

    Args:
        x (torch.Tensor): 入力テンソル
        C (float): 圧縮係数
        clip_val (float): クリッピング値

    Returns:
        torch.Tensor: 圧縮されたテンソル
    """

    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(
    x: torch.Tensor,
    C: float = 1.0,
) -> torch.Tensor:
    """
    ダイナミックレンジ圧縮を解除する。

    Args:
        x (torch.Tensor): 圧縮されたテンソル
        C (float): 圧縮に使用された係数

    Returns:
        torch.Tensor: 解凍されたテンソル
    """

    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """
    スペクトル正規化を適用する。

    Args:
        magnitudes (torch.Tensor): スペクトログラムのマグニチュード

    Returns:
        torch.Tensor: 正規化されたスペクトログラム
    """

    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """
    スペクトル正規化を解除する。

    Args:
        magnitudes (torch.Tensor): 正規化されたスペクトログラム

    Returns:
        torch.Tensor: 元のスペクトログラム
    """

    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis: dict[str, torch.Tensor] = {}
hann_window: dict[str, torch.Tensor] = {}


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """
    入力波形から線形スペクトログラムを計算する。

    Args:
        y (torch.Tensor): 入力波形 (バッチ, サンプル)
        n_fft (int): FFT サイズ
        sampling_rate (int): サンプリングレート
        hop_size (int): ホップサイズ
        win_size (int): ウィンドウサイズ
        center (bool): センタリングするかどうか

    Returns:
        torch.Tensor: スペクトログラム
    """

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    """
    線形スペクトログラムを Mel スペクトログラムに変換する。

    Args:
        spec (torch.Tensor): 線形スペクトログラム
        n_fft (int): FFT サイズ
        num_mels (int): Mel バンド数
        sampling_rate (int): サンプリングレート
        fmin (float): 最小周波数
        fmax (float | None): 最大周波数

    Returns:
        torch.Tensor: Mel スペクトログラム
    """

    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
    center: bool = False,
) -> torch.Tensor:
    """
    入力波形から Mel スペクトログラムを計算する。

    Args:
        y (torch.Tensor): 入力波形 (バッチ, サンプル)
        n_fft (int): FFT サイズ
        num_mels (int): Mel バンド数
        sampling_rate (int): サンプリングレート
        hop_size (int): ホップサイズ
        win_size (int): ウィンドウサイズ
        fmin (float): 最小周波数
        fmax (float | None): 最大周波数
        center (bool): センタリングするかどうか

    Returns:
        torch.Tensor: Mel スペクトログラム
    """

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
