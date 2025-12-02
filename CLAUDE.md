# AGENTS.md

## Python 環境セットアップ

このプロジェクトでは `.venv` 仮想環境を使用することを前提としています：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

以降のコマンドは全て仮想環境内で実行するか、 `.venv/bin/python` を使用してください。

## 開発コマンド

### 初期化・セットアップ
```bash
.venv/bin/python initialize.py  # 必要なモデルとデフォルト TTS モデルのダウンロード
```

### テスト実行
```bash
# PyTorchでの推論テスト
hatch run test:test  # CPU 推論テスト
hatch run test:test-cuda  # CUDA 推論テスト

# ONNX推論テスト  
hatch run test-onnx:test  # CPU 推論テスト（ONNX）
hatch run test-onnx:test-cuda  # CUDA 推論テスト（ONNX）
hatch run test-onnx:test-directml  # DirectML 推論テスト

# 正規化テスト
hatch run test-normalizer:test
```

### リンティング・フォーマット
```bash
hatch run style:check  # ruff でのコードチェック
hatch run style:fmt   # ruff でのフォーマット
```

### 学習・推論
```bash
# 学習用スクリプト（基本的にはメンテナンスしていない）
.venv/bin/python train_ms.py  # マルチスピーカー学習
.venv/bin/python train_ms_jp_extra.py  # JP-Extra 学習

# 推論
.venv/bin/python app.py  # WebUI 起動（旧版）
.venv/bin/python server_editor.py --inbrowser  # エディター起動
.venv/bin/python server_fastapi.py  # API サーバー起動

# 前処理
.venv/bin/python preprocess_all.py  # 全前処理実行
.venv/bin/python bert_gen.py  # BERT特徴生成
.venv/bin/python style_gen.py  # スタイルベクトル生成
```

## プロジェクト構造

### アーキテクチャ概要
Style-Bert-VITS2 は、Bert-VITS2 をベースとした日本語対応の音声合成システム。感情や発話スタイルを制御可能な音声合成を実現。

**このフォークの特徴**：
- 推論コードのメンテナンスと性能改善が主な焦点
- 学習コードは基本的に手を加えていない
- JP-Extra モデル（`models_jp_extra.py`）が現在のメイン
- 従来の多言語対応モデル（`models.py`）も並存

### 主要コンポーネント

**style_bert_vits2/**: 推論専用ライブラリ（リファクタリング済み）
- `tts_model.py`: TTS モデルのエントリーポイント
- `models/`: 音声合成モデル
  - `models_jp_extra.py`: **現在のメインモデル**（JP-Extra 版）
  - `models.py`: 従来の多言語対応モデル
  - `infer.py`, `infer_onnx.py`: 推論エンジン
- `nlp/`: 言語処理（日本語、中国語、英語対応）
- `voice.py`: 音声調整機能

**学習・前処理スクリプト**（ルートディレクトリ）:
- `train_ms.py`, `train_ms_jp_extra.py`: 学習スクリプト
- `preprocess_*.py`: データ前処理
- `bert_gen.py`: BERT 特徴抽出
- `style_gen.py`: スタイルベクトル生成

**WebUI・API**:
- `app.py`: 旧版 WebUI
- `server_editor.py`: エディター機能
- `server_fastapi.py`: RESTful API

### 依存関係の構造

**pyproject.toml** の依存関係は推論部分のみを定義：
- 基本依存関係：推論に必要な最小限のライブラリ
- `torch` オプション：PyTorch推論に必要
- `onnxruntime` オプション：ONNX推論に必要（Torch依存なし）

**推論方式の選択**：
- ONNX推論：Torch依存を除去可能、軽量
- PyTorch推論：別途Torchのインストールが必要（`torch` オプションを有効にする）

### 設定ファイル
- `config.yml`: メイン設定ファイル（学習、推論、サーバー設定）
- `pyproject.toml`: Python 環境設定、依存関係、テスト設定
- `Data/{model_name}/config.json`: モデル固有設定

### モデル・データ構造
```
model_assets/
├── {model_name}/
│   ├── config.json
│   ├── {model}_file.safetensors
│   └── style_vectors.npy
Data/
├── {model_name}/
│   ├── raw/        # 元音声ファイル
│   ├── wavs/       # 前処理済み音声
│   ├── train.list  # 学習リスト
│   └── val.list    # 検証リスト
```

## 開発時の注意点

### フォークの方針
- **推論コードの最適化がメイン**：メモリ効率、速度改善
- **学習コードは基本ノーメンテ**：必要最小限の修正のみ
- **JP-Extra が主軸**：日本語特化の改良版を重視

### 言語対応
日本語、中国語、英語の多言語対応。各言語ごとに異なる BERT モデルと前処理パイプラインを使用。

### モデル形式
- safetensors 形式を推奨（PyTorch の pth からの移行）
- ONNX 推論もサポート（`infer_onnx.py`）

### GPU・CPU対応
学習には CUDA 必須。推論は CPU でも可能（`--device cpu`）。

### 設定管理
`config.yml` で主に学習関連の設定が管理されている。  
しかしこのフォークでは積極的に維持されておらず、またライブラリコンポーネントからも参照されない。

### テスト環境
hatch を使用した環境管理。PyTorch と ONNX、異なるデバイス（CPU/CUDA/DirectML）でのテストをサポート。
