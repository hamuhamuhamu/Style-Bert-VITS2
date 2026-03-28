# Nanairo Speaker Adapter

## 概要

Nanairo の speaker path は、`speaker_embedding` から `g` を直接全面置換する方式ではなく、**既存の `g` 分布を基準点とする residual adapter** として実装しています。  
現在の構成は `SpeakerControlEncoder + SpeakerAdapter + g_neutral + gated delta` の 4 要素です。

```text
speaker_embedding (.spk.npy, 192 dim)
  -> SpeakerControlEncoder
  -> ctrl
  -> SpeakerAdapter
  -> delta
  -> gated_delta = sigmoid(gate_logit) * delta
  -> g = g_neutral + gated_delta
```

## 2 モジュール構成

### SpeakerControlEncoder

- 入力は anime-speaker-embedding 由来の 192 次元 speaker embedding
- 役割は、話者性制御に必要な情報を低次元の control subspace `ctrl` に圧縮すること
- 既定では `speaker_adapter_bottleneck_dim = 96`

### SpeakerAdapter

- `ctrl` から `g` 空間への差分 `delta` を生成する
- `delta` は gate 適用前の生の差分であり、そのまま `g` に加算しない
- 実際に `g` へ入るのは `sigmoid(gate_logit) * delta` である

## g_neutral + gated delta

`g_neutral` は、speaker path の基準点です。  
初期実装では、対象多話者モデルの `emb_g.weight` 平均を第一候補として使います。

最終的な `g` は次式で決まります。

```text
g = g_neutral + sigmoid(gate_logit) * delta
```

この構成により、次の性質を狙います。

- 初期状態で speaker path が過剰に強くならない
- `speaker_embedding` から絶対座標を一発で回帰しない
- 既存の `emb_g` 分布を壊しにくい

## g の注入位置

生成された `g` は、JP-Extra 系と同じく以下のモジュールに渡されます。

- TextEncoder (`enc_p`)
- PosteriorEncoder (`enc_q`)
- Flow (`flow`)
- StochasticDurationPredictor (`sdp`)
- DurationPredictor (`dp`)
- Generator (`dec`)

推論時も同様で、`speaker_embedding` が指定された場合に限り、Adapter 由来の `g` を優先して使います。  
`g_adjust` が指定された場合は、その後段で `g` に加算されます。

## 新しい config キー

### model

- `model.use_speaker_adapter`
  - speaker path を有効化する
- `model.speaker_adapter_input_dim`
  - speaker embedding の入力次元
- `model.speaker_adapter_bottleneck_dim`
  - control subspace の次元

### data

- `data.use_speaker_embedding`
  - 学習時に `.spk.npy` を読み込む

### train

- `train.train_speaker_adapter_only`
  - `SpeakerControlEncoder` と `SpeakerAdapter` のみを学習する
- `train.disable_discriminators_for_adapter`
  - Adapter-only 学習時に Discriminator を止める
- `train.c_teacher`
  - `L_teacher` の重み
- `train.c_delta_l2`
  - delta L2 penalty の重み

## 補助損失

### L_teacher

Adapter-only 学習時のみ有効です。  
forward の戻り値に含まれる `g` を `emb_g(sid)` に近づけ、speaker path が既存の多話者 `g` 分布から大きく外れないようにします。

```text
g_target = emb_g(sid)
g_pred = g.squeeze(-1)
L_teacher = mse(g_pred, g_target)
```

初期値は以下です。

- `c_teacher = 0.1`

強すぎると単なる `emb_g` 回帰になりやすいため、小さめの係数から始めます。

### delta L2 penalty

Adapter が出す生の差分 `delta` のノルムを抑える正則化です。  
gate 適用前の `delta` を再計算し、その二乗平均を損失へ加えます。

```text
delta = speaker_adapter.net(ctrl)
L_delta_l2 = mean(delta^2)
```

初期値は以下です。

- `c_delta_l2 = 0.01`

これにより、`g_neutral` からの逸脱を緩やかに保ちつつ学習できます。

## Phase 1 の学習手順

### 1. `.spk.npy` を生成する

各音声ファイル `xxx.wav` に対して `xxx.wav.spk.npy` を用意します。

```bash
uv run python speaker_embedding_gen.py --model YourModel --skip_existing
```

### 2. 必要ならクラスタリングで話者ラベルを見直す

Char embedding ベースで自動クラスタリングできます。

```bash
uv run python scripts/speaker_adapter/cluster_char_embeddings.py \
  --model YourModel \
  --dry_run \
  --skip_existing
```

### 3. Adapter-only 学習を実行する

`config_nanairo.json` では、少なくとも以下を有効にします。

- `train.train_speaker_adapter_only: true`
- `train.disable_discriminators_for_adapter: true`
- `data.use_speaker_embedding: true`
- `model.use_speaker_adapter: true`
- `train.c_teacher: 0.1`
- `train.c_delta_l2: 0.01`

学習実行例です。

```bash
uv run python train_ms_nanairo.py --model YourModel
```

Adapter-only 学習時は VITS2 本体を凍結し、`SpeakerControlEncoder` と `SpeakerAdapter` のみ更新します。

### 4. TensorBoard で監視する

学習中は最低限、以下を監視します。

- `loss/teacher`
- `loss/delta_l2`
- `g/adapter_gate`
- `g/adapter_delta_norm_mean`
- `g/adapter_gated_delta_norm_mean`
- 未学習話者に対するコサイン類似度

## 推論 API

推論では `speaker_embedding` キーワード引数を使います。  
`.spk.npy` を NumPy として読み込み、そのまま渡す想定です。

```python
sr, audio = tts_model.infer(
    text='こんにちは',
    language=Languages.JP,
    speaker_id=0,
    speaker_embedding=np.load('Data/YourModel/example.wav.spk.npy'),
)
```

## g ダンプと評価

### g ダンプ

```bash
uv run python scripts/speaker_adapter/dump_g.py \
  --model YourModel \
  --checkpoint G_0.safetensors \
  --output_npz outputs/g_dump.npz \
  --output_meta outputs/g_dump.jsonl
```

### 軸抽出

```bash
uv run python scripts/speaker_adapter/extract_axes.py \
  --model YourModel \
  --g_npz outputs/g_dump.npz \
  --meta_jsonl outputs/g_dump.jsonl \
  --output_dir outputs/axes
```

### 方向操作デモ

```bash
uv run python scripts/speaker_adapter/apply_axes_demo.py \
  --model YourModel \
  --checkpoint G_0.safetensors \
  --text 'こんにちは' \
  --speaker_embedding Data/YourModel/example.wav.spk.npy \
  --axes_npz outputs/axes/axes.npz \
  --axes_json outputs/axes/axes.json \
  --axis_name f0_mean \
  --alphas -2,-1,0,1,2 \
  --output_dir outputs/demo
```

### cosine 評価

```bash
uv run python scripts/speaker_adapter/eval_speaker_similarity.py \
  --pairs_jsonl outputs/eval_pairs.jsonl \
  --output_json outputs/eval_result.json
```

`eval_pairs.jsonl` の想定形式です。

```json
{"ref_audio_path":"path/to/ref.wav","gen_audio_path":"path/to/gen.wav","speaker":"spk01"}
```
