# SFT 训练

两个入口共用单设备训练循环，分别支持 ASR / 音频理解和连续 latent TTS。
使用 [README](README.md) 中的原生环境；CUDA 计算采用 BF16 autocast，可训练参数及 AdamW 状态保留 FP32。

## 数据

每行一个 JSON 对象，音频相对路径以 manifest 所在目录为基准。转写必须对应完整音频。

```jsonl
{"task":"asr","audio":"speech.wav","text":"转写结果"}
{"task":"understand","audio":["a.wav","b.wav"],"prompt":"比较两个说话人。","text":"回答"}
{"task":"understand","audio":"audio.wav","prompt":"为什么？","reasoning":"推理过程","text":"回答"}
```

TTS 的目标文本、参考文本必须分别匹配各自音频；参考音频与参考文本应成对提供。

```jsonl
{"audio":"target.wav","text":"目标转写","language":"zh"}
{"audio":"target.wav","text":"目标转写","prompt_audio":"ref.wav","prompt_text":"参考转写","language":"zh"}
```

`--max-audio-seconds` 默认 30 秒，限制一个样本内所有音频的总时长；
`--max-sequence-length` 默认 4096。超限会报错，不截断音频或文本。
训练前检查数据字段和文件；读取时检查空音频、非有限值和序列预算。
ASR 可以用空转写表示无语音样本，TTS 要求非空文本。

## 训练入口

ASR / 理解分支默认只训练 Audio Adapter：

```bash
uv run python scripts/train_fireredaudio_understanding.py \
  --model /path/to/FireRedAudio --manifest train.jsonl --eval-manifest valid.jsonl \
  --steps 1000 --learning-rate 1e-5 --warmup-steps 50 \
  --gradient-accumulation 4 --output-dir checkpoints/asr
```

TTS 默认训练 Patch Encoder 和 DiT：

```bash
uv run python scripts/train_fireredaudio_tts.py \
  --model /path/to/FireRedAudio --manifest tts_train.jsonl --eval-manifest tts_valid.jsonl \
  --steps 1000 --learning-rate 1e-5 --warmup-steps 50 \
  --gradient-accumulation 4 --output-dir checkpoints/tts
```

示例学习率仅作为起点，应按数据量和验证损失调整。使用 `--steps 1 --no-save` 可做短训练检查；
`--no-save` 不写检查点或指标文件。

| 参数 | 含义 |
|---|---|
| `--train-modules` | 理解：`backbone,audio_encoder,audio_adapter`；TTS：`backbone,patch_encoder,dit` |
| `--learning-rate` / `--backbone-learning-rate` | 音频模块 / LLM 学习率，默认 2e-4 / 3e-5 |
| `--scheduler` | `linear`（默认）或 `constant`，均支持 warmup |
| `--save-every` / `--eval-every` | 默认每 100 个优化器步骤保存 / 验证，最后一步也执行 |
| `--loss-chunk-size` / `--flow-chunk-size` | 文本监督位置 / DiT patch 分块大小，默认 64 / 32 |
| `--no-gradient-checkpointing` | 关闭默认启用的 backbone / encoder 梯度检查点 |
| `--attention` | 默认 `sdpa`；可选 `eager` 或已安装的 `flash_attention_2` |

RedAE 始终冻结。数据按固定种子逐轮打乱，验证使用固定随机状态且不影响训练随机序列。
梯度累积按整个窗口的有效 token 权重和音频 patch 数分别归一化；
非有限损失或梯度会在更新参数前报错。

文本 CE 只计算有效监督位置并分块重算，DiT 也分块计算和重算，以降低显存峰值。
骨干检查点即使在冻结骨干的 eval 模式下仍可工作，让梯度回传到音频模块。
输入波形与特征留在 CPU，逐个样本送入 GPU。一个微批次仍是一条对话。

## 保存与恢复

检查点按 `step-00000100.pt` 保存，`latest.pt` 指向最新检查点，写入采用原子替换。
保存内容包括可训练参数、优化器、学习率调度器、采样顺序和 Python / NumPy / Torch 随机状态。

恢复时，在原训练命令后添加 `--resume checkpoints/asr/latest.pt`。
`--steps` 是计划的总步数，恢复时保持不变；基础权重、训练清单、验证清单、
训练超参数和 Torch / Transformers 版本也必须匹配。文件位置可以改变，但内容应保持一致。
校验会读取基础权重计算 SHA-256，启动时有额外 I/O。

支持新格式检查点；旧训练原型的无版本 `latest.pt` 不支持直接恢复。
若要在已完成的训练后继续新的计划，可先导出完整模型，再以它为基础开启新训练。

## 导出与推理

训练检查点是增量权重。合并导出后即可使用原有推理入口：

```bash
uv run python scripts/export_fireredaudio_checkpoint.py \
  --model /path/to/FireRedAudio --checkpoint checkpoints/asr/latest.pt \
  --output /path/to/finetuned-model

uv run python inference.py --task asr \
  --model /path/to/finetuned-model --audio speech.wav
```

导出目录必须不存在，基础权重必须与训练时一致。导出保留全部冻结模块、tokenizer 和 processor，
并按原权重精度保存（发布模型为 BF16）。TTS 推理仍需要另行提供 RedAE Decoder。
用于 vLLM / SGLang 时，再从这个完整模型导出文本主干，并让音频客户端使用同一个完整模型。

## 验证与边界

```bash
uv run python -m unittest discover -s tests -p test_training.py -v
```

测试覆盖分块损失和梯度、累积归一化、CFG、参考音频掩码、因果条件、
冻结模块、保存恢复、验证随机状态及合并后模型加载。
CPU 测试检查恢复训练逐参数一致；GPU BF16 与加速内核可能产生数值差异，不承诺逐位重现。

当前是单设备 SFT，不支持 DDP / FSDP / ZeRO 或对话 packing。全量 LLM 的 FP32 参数和优化器状态
需要显著更多显存；默认局部模块训练不等于完整复现论文的多阶段训练。
短样本训练验证不能代替独立验证集、转写准确率和语音生成质量评估。

目标函数参考 [FireRedAudio 论文](https://arxiv.org/abs/2608.24168)第 2.4、2.5 节；
AdamW、均匀 flow-time 采样和此处的训练调度是工程实现选择。
