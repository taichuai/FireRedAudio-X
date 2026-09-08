# FireRedAudio-X

基于 [FireRedAudio](https://github.com/FireRedTeam/FireRedAudio) 的**非官方扩展**，聚焦训练工具和推理加速。

## 新增功能

- **加速推理**：vLLM / SGLang 后端，支持 ASR、音频理解、thinking、多音频及 JSONL 批量请求。
- **单设备 SFT**：ASR / 理解分支和连续 latent TTS，支持验证、断点恢复与完整模型导出。
- **辅助工具**：批量语音生成、副语言评估脚本和推理回归测试。

训练用法与支持范围见 [训练说明](TRAINING_REPRODUCTION.md)。副语言评估仍为实验性工具，失败样本统计待完善。

## 安装

原生推理和训练使用 Python 3.10、CUDA 12.8 PyTorch wheel，需要 FFmpeg。

```bash
uv sync --frozen

# 可选加速扩展，编译需 CUDA Toolkit 和 C++ 工具链
uv sync --frozen --extra accel --extra accel-build
```

vLLM / SGLang 按需选择，分别使用独立的 Python 3.12 环境：

```bash
# vLLM
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python --torch-backend cu128 -r requirements-vllm.txt

# SGLang
uv venv --python 3.12 .venv-sglang
uv pip install --python .venv-sglang/bin/python --torch-backend cu128 -r requirements-sglang.txt
```

音频客户端使用原生环境，文本服务使用对应后端环境；两个后端可以同时保留。
独立环境用 `uv pip` 管理，不执行主项目的 `uv sync`。

## 脚本入口

| 功能 | 入口 |
|---|---|
| 原生 ASR、理解、TTS、语音编辑、音色设计 | [inference.py](inference.py) |
| Gradio 演示（上游提供） | [app.py](app.py) |
| 加速推理与服务启动 | [加速推理说明](ACCELERATED_INFERENCE_CN.md) |
| ASR / 理解训练 | [train_fireredaudio_understanding.py](scripts/train_fireredaudio_understanding.py) |
| TTS 训练 | [train_fireredaudio_tts.py](scripts/train_fireredaudio_tts.py) |
| 合并导出训练权重 | [export_fireredaudio_checkpoint.py](scripts/export_fireredaudio_checkpoint.py) |
| 批量语音生成 | [run_instruct_tts_demo.py](scripts/run_instruct_tts_demo.py) |
| 副语言评估（实验性） | [eval_paralanguage.py](scripts/eval_paralanguage.py) |

各脚本参数可通过 `--help` 查看。以下命令在仓库根目录执行，GPU 编号按实际设备调整。

## 原生推理

下载官方权重：

```bash
uv run hf download FireRedTeam/FireRedAudio --local-dir pretrained_models/
```

### ASR 与音频理解

```bash
# 语音识别
CUDA_VISIBLE_DEVICES=0 uv run python inference.py \
  --task asr --model pretrained_models/FireRedAudio \
  --audio assets/examples/asr_zh_fleurs.wav

# 音频理解，启用 thinking
CUDA_VISIBLE_DEVICES=0 uv run python inference.py \
  --task understand --model pretrained_models/FireRedAudio \
  --audio assets/examples/assets_mmau_test.wav \
  --prompt 'Describe the audio in detail.' \
  --enable-thinking --max-new-tokens 2048
```

### TTS

通过参考音频和对应转写克隆音色，生成目标文本的语音：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference.py \
  --task tts --model pretrained_models/FireRedAudio \
  --vae-decoder pretrained_models/RedAE_decoder/model.pt \
  --prompt-audio assets/examples/tts_zh_prompt.wav \
  --prompt-text '同时，他强调微调要科学有序。' \
  --target-text '这是一个语音合成示例。' --language zh --output tts.wav
```

语音编辑和音色设计见 [官方用法](https://github.com/FireRedTeam/FireRedAudio#quick-start-)。
Gradio 演示可用以下命令启动：

```bash
uv pip install --python .venv/bin/python gradio
uv run --no-sync python app.py \
  --model_path pretrained_models/FireRedAudio \
  --vae_decoder_path pretrained_models/RedAE_decoder/model.pt \
  --host 127.0.0.1 --port 7860
```

## 加速推理

安装所选后端后，先导出文本主干：

```bash
uv run python -m fireredaudio.accelerated.prepare \
  --source pretrained_models/FireRedAudio \
  --output pretrained_models/FireRedAudio-backbone
```

启动一个服务，以下两个命令任选其一：

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8010 GPU_MEMORY_UTILIZATION=0.6 \
  bash scripts/serve_accelerated.sh vllm

CUDA_VISIBLE_DEVICES=0 PORT=8011 GPU_MEMORY_UTILIZATION=0.6 \
  bash scripts/serve_accelerated.sh sglang
```

在另一个终端运行音频客户端。示例将客户端放在另一张 GPU；共用 GPU 时需为音频编码器留出显存。

```bash
CUDA_VISIBLE_DEVICES=1 uv run python inference_accelerated.py \
  --backend vllm --base-url http://127.0.0.1:8010 \
  --task asr --audio assets/examples/asr_zh_fleurs.wav

# JSONL 批量请求
CUDA_VISIBLE_DEVICES=1 uv run python inference_accelerated.py \
  --backend vllm --base-url http://127.0.0.1:8010 \
  --input-jsonl assets/examples/accelerated_requests.jsonl \
  --concurrency 4 --max-new-tokens 2048 --output results.jsonl
```

使用 SGLang 时，将客户端改为 `--backend sglang --base-url http://127.0.0.1:8011`。
当前加速范围为 ASR / 音频理解，TTS 暂未接入。更多选项见 [加速推理说明](ACCELERATED_INFERENCE_CN.md)。

## SFT 训练

训练清单为 JSONL，音频相对路径以清单所在目录为基准，文本必须与完整音频对应。

```jsonl
{"task":"asr","audio":"speech.wav","text":"转写结果"}
{"task":"understand","audio":"audio.wav","prompt":"发生了什么？","text":"回答"}
```

TTS 使用以下格式，参考音频与参考文本为可选的成对字段：

```jsonl
{"audio":"target.wav","text":"目标转写","language":"zh"}
{"audio":"target.wav","text":"目标转写","prompt_audio":"ref.wav","prompt_text":"参考转写","language":"zh"}
```

### ASR / 理解与 TTS

准备训练集与独立验证集后执行：

```bash
# 默认训练 Audio Adapter
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fireredaudio_understanding.py \
  --model pretrained_models/FireRedAudio \
  --manifest train.jsonl --eval-manifest valid.jsonl \
  --steps 1000 --learning-rate 1e-5 --warmup-steps 50 \
  --gradient-accumulation 4 --output-dir checkpoints/asr

# 默认训练 Patch Encoder + DiT，RedAE 保持冻结
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_fireredaudio_tts.py \
  --model pretrained_models/FireRedAudio \
  --manifest tts_train.jsonl --eval-manifest tts_valid.jsonl \
  --steps 1000 --learning-rate 1e-5 --warmup-steps 50 \
  --gradient-accumulation 4 --output-dir checkpoints/tts
```

只检查训练是否能运行，可使用随仓库提供的样例：

```bash
uv run python scripts/train_fireredaudio_understanding.py \
  --manifest assets/examples/understanding_train_example.jsonl --steps 1 --no-save

uv run python scripts/train_fireredaudio_tts.py \
  --manifest assets/examples/tts_train_example.jsonl --steps 1 --no-save
```

### 恢复与导出

恢复时使用原训练命令，并追加 `--resume checkpoints/asr/latest.pt`（TTS 对应其输出目录）。
`--steps` 表示原计划总步数，恢复时保持训练设置及数据内容一致。

训练检查点为增量权重，推理前合并为完整模型：

```bash
uv run python scripts/export_fireredaudio_checkpoint.py \
  --model pretrained_models/FireRedAudio \
  --checkpoint checkpoints/asr/latest.pt --output finetuned-model

uv run python inference.py --task asr \
  --model finetuned-model --audio assets/examples/asr_zh_fleurs.wav
```

TTS 检查点使用同一导出工具，随后将原生 TTS 命令的 `--model` 指向导出目录。
加速推理还需从该目录重新导出文本主干。

当前支持单设备 SFT，不含 DDP / FSDP / ZeRO。可训练参数和优化器状态为 FP32，CUDA 计算使用 BF16；
默认启用梯度检查点及分块损失。更多训练参数和验证范围见 [训练说明](TRAINING_REPRODUCTION.md)。

## 参考

- [FireRedAudio 论文](https://arxiv.org/abs/2608.24168)与[官方实现](https://github.com/FireRedTeam/FireRedAudio)。
- 推理适配参考：[FireRedAudio-VLLM](https://github.com/duduke321/FireRedAudio-VLLM)。

代码沿用 [Apache-2.0](LICENSE)，保留上游版权声明。
