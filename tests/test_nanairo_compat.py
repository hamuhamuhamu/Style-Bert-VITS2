"""
Nanairo / JP-Extra 互換性の回帰テスト。
Nanairo は JP-Extra ベースのため、設定正規化と checkpoint 部分ロードの両方が壊れていないことを検証する。
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.utils.safetensors import load_safetensors


class DummyTextEncoder(torch.nn.Module):
    """
    TextEncoder 互換の最小埋め込みモジュール。

    Notes:
        実際の部分ロード対象キー `enc_p.emb.weight` を再現するために使う。
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 4) -> None:
        """
        DummyTextEncoder を初期化する。

        Args:
            vocab_size (int): embedding の語彙数
            embedding_dim (int): embedding 次元数
        """

        super().__init__()

        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)


class DummyEmbeddingModel(torch.nn.Module):
    """checkpoint 互換テスト用の最小モデル。"""

    def __init__(self, vocab_size: int, embedding_dim: int = 4) -> None:
        """
        DummyEmbeddingModel を初期化する。

        Args:
            vocab_size (int): embedding の語彙数
            embedding_dim (int): embedding 次元数
        """

        super().__init__()

        self.enc_p = DummyTextEncoder(
            vocab_size=vocab_size, embedding_dim=embedding_dim
        )


def test_hyper_parameters_force_use_jp_extra_for_nanairo() -> None:
    """use_nanairo が有効な場合は use_jp_extra を自動補正する。"""

    hyper_parameters = HyperParameters.model_validate(
        {
            "model_name": "Dummy",
            "version": "2.7.0",
            "data": {
                "use_nanairo": True,
                "use_jp_extra": False,
            },
        },
    )

    assert hyper_parameters.data.use_nanairo is True
    assert hyper_parameters.data.use_jp_extra is True
    assert hyper_parameters.is_nanairo_like_model() is True
    assert hyper_parameters.is_jp_extra_like_model() is True


def test_hyper_parameters_version_suffix_nanairo_forces_data_use_nanairo() -> None:
    """version 文字列が Nanairo を示しているのに data.use_nanairo が有効でないと、bert_gen / DataLoader / 推論で語彙や NLP の経路が不一致になるため補正する。"""

    hyper_parameters = HyperParameters.model_validate(
        {
            "model_name": "Dummy",
            "version": "2.9.0-Nanairo",
            "data": {
                "use_jp_extra": True,
                "use_nanairo": False,
            },
        },
    )

    assert hyper_parameters.data.use_nanairo is True
    assert hyper_parameters.is_nanairo_like_model() is True


def test_load_safetensors_partially_copies_embedding_rows_only_when_opted_in(
    tmp_path: Path,
) -> None:
    """`.safetensors` checkpoint では明示 opt-in 時のみ共有語彙ぶんを部分ロードする。"""

    source_model = DummyEmbeddingModel(vocab_size=5)
    target_model = DummyEmbeddingModel(vocab_size=8)
    with torch.no_grad():
        source_model.enc_p.emb.weight.copy_(
            torch.arange(20, dtype=torch.float32).view(5, 4),
        )
        target_model.enc_p.emb.weight.fill_(-1.0)

    checkpoint_path = tmp_path / "dummy_nanairo_expandable.safetensors"
    save_file(source_model.state_dict(), str(checkpoint_path))

    load_safetensors(
        checkpoint_path=checkpoint_path,
        model=target_model,
        device="cpu",
        allow_partial_load_embedding_keys=("enc_p.emb.weight",),
    )

    assert torch.equal(
        target_model.enc_p.emb.weight[:5],
        source_model.enc_p.emb.weight,
    )
    assert torch.all(target_model.enc_p.emb.weight[5:] == -1.0)


def test_load_safetensors_without_opt_in_preserves_previous_shape_mismatch_error(
    tmp_path: Path,
) -> None:
    """`.safetensors` checkpoint では opt-in しなければ shape mismatch を従来どおり失敗させる。"""

    source_model = DummyEmbeddingModel(vocab_size=5)
    target_model = DummyEmbeddingModel(vocab_size=8)
    with torch.no_grad():
        source_model.enc_p.emb.weight.copy_(
            torch.arange(20, dtype=torch.float32).view(5, 4),
        )
        target_model.enc_p.emb.weight.fill_(-1.0)

    checkpoint_path = tmp_path / "dummy_nanairo_expandable_no_opt_in.safetensors"
    save_file(source_model.state_dict(), str(checkpoint_path))

    with pytest.raises(RuntimeError):
        load_safetensors(
            checkpoint_path=checkpoint_path,
            model=target_model,
            device="cpu",
        )


def test_load_safetensors_reports_unexpected_checkpoint_keys(tmp_path: Path) -> None:
    """`.safetensors` checkpoint の余分な key は warning として報告する。"""

    source_model = DummyEmbeddingModel(vocab_size=5)
    checkpoint_path = tmp_path / "dummy_with_unexpected_key.safetensors"
    checkpoint_state_dict = source_model.state_dict()
    checkpoint_state_dict["unexpected.weight"] = torch.ones(1)
    save_file(checkpoint_state_dict, str(checkpoint_path))

    captured_messages: list[str] = []
    logger_sink_id = logger.add(captured_messages.append, format="{message}")
    try:
        load_safetensors(
            checkpoint_path=checkpoint_path,
            model=DummyEmbeddingModel(vocab_size=5),
            device="cpu",
        )
    finally:
        logger.remove(logger_sink_id)

    captured_text = "\n".join(captured_messages)
    assert "Unexpected key: unexpected.weight" in captured_text
