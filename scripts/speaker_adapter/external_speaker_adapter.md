# Nanairo External Speaker Adapter

## g の生成と注入位置

Nanairo (models_nanairo.py) では、g はグローバル話者条件として扱われます。形状は常に `[B, gin_channels, 1]` です。通常は `nn.Embedding` から `g = emb_g(sid).unsqueeze(-1)` を作り、`n_speakers <= 0` の場合は `ReferenceEncoder` から生成します。Nanairo ではこれに加えて `ExternalSpeakerAdapter` を用意し、外部 speaker embedding から g を生成できます。

g の注入箇所は JP-Extra と同一です。`SynthesizerTrn.forward` では次のモジュールに g が渡されます。

- TextEncoder (`enc_p`) の encoder ブロック
- PosteriorEncoder (`enc_q`)
- Flow (`flow`)
- StochasticDurationPredictor (`sdp`)
- DurationPredictor (`dp`)
- Generator (`dec`)

推論時も `infer_input_feature` で同様に g を生成し、上記と同じ経路に注入します。Nanairo では `external_spk_emb` が指定されている場合に限り Adapter 由来の g を優先し、`g_adjust` が指定された場合は g に加算します。style 経路とは直接の合流はありませんが、学習上の相互作用はあり得るため完全非干渉とは言い切りません。

## 追加した設定キー

config_nanairo.json で以下のキーを追加しています。

- `model.use_external_speaker_adapter`: ExternalSpeakerAdapter を有効化します。
- `model.external_speaker_embedding_dim`: 外部 embedding 次元です。
- `model.external_speaker_adapter_hidden_dim`: Adapter の中間次元です。
- `data.use_external_speaker_embedding`: 学習時に外部 embedding を読み込むかどうかを指定します。
- `train.train_external_speaker_adapter_only`: Adapter のみを更新するモードです。
- `train.disable_discriminators_for_adapter`: Adapter-only 時に Discriminator を無効化します。

## 外部 speaker embedding の用意

学習・推論の両方で外部 speaker embedding を使う前提です。各音声ファイル `xxx.wav` に対して、`xxx.wav.spk.npy` のような sidecar を用意してください。埋め込みは 1 次元ベクトルとして保存する想定です。学習時は同一話者の別クリップを参照する仕様です。

## external speaker embedding の生成

Char モデルを固定で使う前提のスクリプトとして `external_spk_emb_gen.py` を追加しています。`--model` でモデル名を指定すると、train.list と val.list から自動的に音声ファイルを読み込み、.spk.npy を生成します。

```
.venv/bin/python external_spk_emb_gen.py --model YourModel --skip_existing
```

## Char embedding による自動クラスタリング

話者ラベルを機械的に付け直すためのスクリプトとして `scripts/speaker_adapter/cluster_char_embeddings.py` を追加しています。元の話者ラベルは無視され、Char embedding に基づいた自動クラスタが生成されます。

```
.venv/bin/python scripts/speaker_adapter/cluster_char_embeddings.py \
  --model YourModel \
  --dry_run \
  --skip_existing
```

## Adapter-only 学習

Adapter のみを更新し、Generator 本体は凍結されます。以下は一例です。

```
.venv/bin/python train_ms_nanairo.py --model YourModel
```

`train_external_speaker_adapter_only` が true の場合は Adapter 以外を自動的に凍結します。また `disable_discriminators_for_adapter` が true の場合は Discriminator の更新と損失を無効化し、mel/kl/duration を中心に学習します。

## g ダンプと軸抽出

### g ダンプ

```
.venv/bin/python scripts/speaker_adapter/dump_g.py \
  --model YourModel \
  --checkpoint G_0.safetensors \
  --output_npz outputs/g_dump.npz \
  --output_meta outputs/g_dump.jsonl
```

### 軸抽出

```
.venv/bin/python scripts/speaker_adapter/extract_axes.py \
  --model YourModel \
  --g_npz outputs/g_dump.npz \
  --meta_jsonl outputs/g_dump.jsonl \
  --output_dir outputs/axes
```

### 方向操作デモ

```
.venv/bin/python scripts/speaker_adapter/apply_axes_demo.py \
  --model YourModel \
  --checkpoint G_0.safetensors \
  --text 'こんにちは' \
  --external_embedding Data/YourModel/example.wav.spk.npy \
  --axes_npz outputs/axes/axes.npz \
  --axes_json outputs/axes/axes.json \
  --axis_name f0_mean \
  --alphas -2,-1,0,1,2 \
  --output_dir outputs/demo
```

## cosine 評価の例

```
.venv/bin/python scripts/speaker_adapter/eval_speaker_similarity.py \
  --pairs_jsonl outputs/eval_pairs.jsonl \
  --output_json outputs/eval_result.json
```

`eval_pairs.jsonl` は以下の形式を想定しています。外部 embedding は `*.spk.npy` として事前に用意してください。

```
{"ref_audio_path":"path/to/ref.wav","gen_audio_path":"path/to/gen.wav","speaker":"spk01"}
```
