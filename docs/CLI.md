# CLI

## 0. Install and global paths settings

```bash
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
python -m venv venv
venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Then download the necessary models and the default TTS model, and set the global paths.
```bash
python initialize.py [--skip_default_models] [--only_infer] [--dataset_root <path>] [--assets_root <path>]
```

Optional:
- `--skip_default_models`: Skip downloading the default voice models (use this if you only have to train your own models).
- `--only_infer`: Skip downloading models needed for training (SLM model and pretrained models). Use this if you only need inference.
- `--dataset_root`: Default: `Data`. Root directory of the training dataset. The training dataset of `{model_name}` should be placed in `{dataset_root}/{model_name}`.
- `--assets_root`: Default: `model_assets`. Root directory of the model assets (for inference). In training, the model assets will be saved to `{assets_root}/{model_name}`, and in inference, we load all the models from `{assets_root}`.


## 1. Dataset preparation

### 1.1. Slice audio files

The following audio formats are supported: ".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a".
```bash
python slice.py --model <model_name> [-i <input_dir>] [--min_sec <min_sec>] [-M <max_sec>] [-s <min_silence_dur_ms>] [-t] [--num_processes <num_processes>]
```

Required:
- `--model`, `-m`: Name of the speaker (to be used as the name of the trained model).

Optional:
- `--input_dir`, `-i`: Path to the directory containing the audio files to slice (default: `inputs`).
- `--min_sec`: Minimum duration of the sliced audio files in seconds (default: 2).
- `--max_sec`, `-M`: Maximum duration of the sliced audio files in seconds (default: 12).
- `--min_silence_dur_ms`, `-s`: Silence above this duration (ms) is considered as a split point (default: 700).
- `--time_suffix`, `-t`: Make the filename end with -start_ms-end_ms when saving wav.
- `--num_processes`: Number of processes to use (default: 3).

### 1.2. Transcribe audio files

```bash
python transcribe.py --model <model_name> [--initial_prompt <prompt>] [--language <lang>] [--whisper-model <model>] [--device <device>] [--compute_type <type>] [--use_hf_whisper] [--hf_repo_id <repo_id>] [--batch_size <size>] [--num_beams <beams>] [--no_repeat_ngram_size <size>]
```

Required:
- `--model`, `-m`: Name of the speaker (to be used as the name of the trained model).

Optional:
- `--initial_prompt`: Initial prompt to use for the transcription (default value is specific to Japanese).
- `--device`: `cuda` or `cpu` (default: `cuda`).
- `--language`: `ja`, `en`, or `zh` (default: `ja`).
- `--whisper-model`: Whisper model (default: `large-v3`). Only used if not `--use_hf_whisper`.
- `--compute_type`: Compute type (default: `bfloat16`). Only used if not `--use_hf_whisper`.
- `--use_hf_whisper`: Use Hugging Face's whisper model instead of default faster-whisper (HF whisper is faster but requires more VRAM).
- `--hf_repo_id`: Hugging Face repository ID for Whisper model. Required if `--use_hf_whisper` is used.
- `--batch_size`: Batch size (default: 16). Only used if `--use_hf_whisper`.
- `--num_beams`: Beam size (default: 1).
- `--no_repeat_ngram_size`: N-gram size for no repeat (default: 10).

## 2. Preprocess

```bash
python preprocess_all.py --model <model_name> [--use_jp_extra] [-b <batch_size>] [-e <epochs>] [-s <save_every_steps>] [--num_processes <num_processes>] [--normalize] [--trim] [--val_per_lang <val_per_lang>] [--log_interval <log_interval>] [--freeze_EN_bert] [--freeze_JP_bert] [--freeze_ZH_bert] [--freeze_style] [--freeze_decoder] [--yomi_error <yomi_error>]
```

Required:
- `--model`, `-m`: Name of the speaker (to be used as the name of the trained model).

Optional:
- `--batch_size`, `-b`: Batch size (default: 2).
- `--epochs`, `-e`: Number of epochs (default: 100).
- `--save_every_steps`, `-s`: Save every steps (default: 1000).
- `--num_processes`: Number of processes (default: half of the number of CPU cores).
- `--normalize`: Loudness normalize audio.
- `--trim`: Trim silence.
- `--freeze_EN_bert`: Freeze English BERT.
- `--freeze_JP_bert`: Freeze Japanese BERT.
- `--freeze_ZH_bert`: Freeze Chinese BERT.
- `--freeze_style`: Freeze style vector.
- `--freeze_decoder`: Freeze decoder.
- `--use_jp_extra`: Use JP-Extra model.
- `--val_per_lang`: Validation data per language (default: 0).
- `--log_interval`: Log interval (default: 200).
- `--yomi_error`: How to handle yomi errors (default: `raise`: raise an error after preprocessing all texts, `skip`: skip the texts with errors, `use`: use the texts with errors by ignoring unknown characters).

## 3. Train

Training settings are automatically loaded from the above process.

If NOT using JP-Extra model:
```bash
python train_ms.py --model <model_name> [--pretrained_model_dir <dir>] [--assets_root <path>] [--keep_ckpts <num>] [--skip_default_style] [--no_progress_bar] [--speedup] [--repo_id <username>/<repo_name>] [--not_use_custom_batch_sampler]
```

If using JP-Extra model:
```bash
python train_ms_jp_extra.py --model <model_name> [--pretrained_model_dir <dir>] [--assets_root <path>] [--keep_ckpts <num>] [--skip_default_style] [--no_progress_bar] [--speedup] [--repo_id <username>/<repo_name>] [--not_use_custom_batch_sampler] [--use_ema] [--ema_decay <decay>] [--gradient_accumulation_steps <steps>]
```

Required:
- `--model`, `-m`: Name of the speaker (to be used as the name of the trained model).

Optional (common to both):
- `--pretrained_model_dir`: Directory that contains G_0.safetensors / D_0.safetensors for initialization. If omitted, model_dir is used.
- `--assets_root`: Root directory of model assets needed for inference (default: `model_assets`).
- `--keep_ckpts`: Number of checkpoints to keep. Set to 0 to keep all (default: 1).
- `--skip_default_style`: Skip saving default style config and mean vector. Use this if you want to resume training (since the default style vector has been already made).
- `--no_progress_bar`: Do not show the progress bar while training.
- `--speedup`: Speed up training by disabling logging and evaluation.
- `--repo_id`: Hugging Face repository ID to upload the trained model to. You should have logged in using `huggingface-cli login` before running this command.
- `--not_use_custom_batch_sampler`: Don't use custom batch sampler for training, which was used in the version < 2.5.

Optional (JP-Extra only):
- `--use_ema`: Use Exponential Moving Average (EMA) for model weights. EMA weights are smoother and often produce more stable inference results.
- `--ema_decay`: EMA decay rate. Higher values (e.g., 0.9999) produce smoother weights (default: 0.999).
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before updating weights. Effective batch size = batch_size * gradient_accumulation_steps (default: 1, no accumulation).
