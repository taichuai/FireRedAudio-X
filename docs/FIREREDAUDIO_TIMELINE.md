# 长音频转写、说话人归属与切片

`run_fireredaudio_long_audio.py` 将整段音频送入 FireRedAudio **识别一次**，生成时间戳、
`spk_N`、逐字文本和副语言事件，再调用 `fireredaudio_timeline.py` 完成对齐与切片。

说话人编号由 FireRedAudio 生成，不是独立声纹聚类的结果；Qwen ForcedAligner 只细化时间，不改文本或说话人。
完整转写、准确说话人归属和切片质量仍需复核，脚本不会把“模型正常停止”视为无漏字的证明。

## 安装

原生环境增加音频读写依赖，长音频推荐安装 FlashAttention：

```bash
uv sync --frozen --extra timeline --extra accel --extra accel-build
```

Qwen 对齐器依赖 Transformers 4.57.6，与 FireRedAudio 的环境不同，单独安装：

```bash
uv venv --python 3.12 .venv-aligner
uv pip install --python .venv-aligner/bin/python --torch-backend cu128 -r requirements-aligner.txt
```

模型权重可自动下载，也可传本地目录。对应模型为
[Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)，
接口使用 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 `Qwen3ForcedAligner`。

## 一次完成识别与切片

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/run_fireredaudio_long_audio.py \
  --audio recording.wav --model pretrained_models/FireRedAudio \
  --aligner-python .venv-aligner/bin/python --output results/recording
```

默认使用原生 FireRedAudio，识别后释放模型，再在独立 Python 进程中对齐，可以复用同一张 GPU。
`--device` 和 `--aligner-device` 均默认 `cuda:0`，受 `CUDA_VISIBLE_DEVICES` 控制。

已有 vLLM / SGLang 服务时，可以选择 `--backend vllm` / `--backend sglang` 和 `--base-url`；
音频前端仍在原生环境运行。服务的 `MAX_MODEL_LEN` 必须与客户端 `--max-model-len` 一致。
例如先按 [加速说明](../ACCELERATED_INFERENCE_CN.md) 导出主干，再启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 MAX_MODEL_LEN=32768 PORT=8010 \
  bash scripts/serve_accelerated.sh vllm

# 另一个终端，音频编码与对齐使用另一张 GPU
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/run_fireredaudio_long_audio.py \
  --audio recording.wav --backend vllm --base-url http://127.0.0.1:8010 \
  --max-model-len 32768 --max-new-tokens 8192 \
  --aligner-python .venv-aligner/bin/python --output results/recording-vllm
```

未安装 Qwen 对齐器时可使用 `--coarse-only`。这仅按粗时间戳切片，所有片段会标为待复核。

## 重用识别文本

如果只需重做对齐，无需再次运行 FireRedAudio：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-aligner/bin/python scripts/fireredaudio_timeline.py \
  --audio recording.wav --firered-text results/recording/raw_response.txt \
  --aligner-model Qwen/Qwen3-ForcedAligner-0.6B --output results/realigned
```

输入文本格式：

```text
[00:12.340-00:15.120] spk_2: 这里是转写文本。
[00:16.000-00:17.000] <叹气声>
```

也可以给主入口传 `--firered-text` 跳过识别。空文本、无法解析的行、反向时间戳、
明显超出音频范围的时间戳都会报错，不会静默丢掉这些行。

## 对齐与边界

- 按时间顺序把完整语句组织为默认 90–150 秒目标窗口，尾窗口或受长度限制的窗口可以短于 90 秒。
  每次 Qwen 调用使用窗口内**全部文本与连续音频**，上下文 padding 也计入 150 秒限制。
- 对齐词按文本字符位置映射回原句。文本不匹配、词跨句、句首尾零时长或非单调时间戳会标记失败，
  仅失败句使用邻句上下文重试一次，仍失败则保留原始时间供复核。少量内部零时长词不影响已有的
  句首尾边界，但会标记待复核且不用于长句拆分。超过单窗口限制的原始长句不会送入超长对齐请求。
- 成功对齐的长句按词边界拆分；相邻同说话人片段在间隔不超过 800ms、总长不超过 15 秒时合并。
  无可靠词边界的超长段不强行按比例切文本，而是保留并标为 `segment_over_max_review`。
- 默认头尾最多补 150ms / 500ms。只在句间空隙重新分配 padding，不从可能重叠的语音中间硬切。
  重叠语音仍保留并标记 `overlap_review`。对齐导致原始语句顺序冲突时，回退到粗时间戳并标记复核。
- 独立声音事件使用粗时间戳；包含副语言标签的句子保留粗覆盖范围，标记事件边界待复核。
  纯文本强制对齐不能可靠定位咳嗽、叹气等非词语事件。

因此，`segments.jsonl` 中可能包含待复核的超长或重叠片段；只有无已知风险的片段才进入
`candidates.jsonl`，也仍需抽检。脚本不保证未知模型错误会全部被检测。

## 输出与参数

主入口输出 `raw_response.txt`、`recognition.json`，切片位于 `clips/`；
独立时间线工具直接在指定目录写切片。

| 文件 | 内容 |
|---|---|
| `segments.jsonl`、编号 WAV | 全部片段、说话人、文本、词级时间和风险 |
| `alignment.json` | 每次实际对齐请求的窗口范围、文本、词时间及状态 |
| `review_required.jsonl` / `candidates.jsonl` | 已知风险片段 / 无已知风险候选 |
| `uncovered_energy.json` | 未被切片覆盖的高能量区域，仅用于发现潜在遗漏，不能当作 VAD |
| `transcript.txt`、`review.html` | 文本清单与试听页面 |
| `run.json` | 时长、窗口、片段和风险统计 |

输出目录必须不存在。识别达到输出 token 上限时，会保存原始回答并停止，不对不完整转写继续切片。
`--max-new-tokens` 默认 16384、`--max-model-len` 默认 65536；
输入 embedding 与输出预算之和必须落在上下文内。长音频输出可能需要更多 token。
`--max-audio-seconds` 最多 3600 秒是输入保护上限，不代表已验证一小时音频的完整识别质量或显存需求。

`--window-min-seconds` / `--window-max-seconds` 控制对齐窗口，最大可设为 180 秒；
`--max-segment-seconds`、`--merge-gap-ms` 和 `--head-padding-ms` / `--tail-padding-ms` 控制片段边界。
默认提示词针对中文，`--prompt-file` 可覆盖，`--language` 设置对齐语言。

## 测试

```bash
uv run --extra timeline python -m unittest discover -s tests -p test_timeline.py -v
```

测试覆盖整段识别只调用一次、联合窗口、文本归属、长句拆分、重叠保护、无效时间戳、
截断拒绝以及导出 WAV 时长与清单一致性。

功能验证：A100 上对一段约 295 秒多人音频进行一次原生 FireRedAudio 识别，得到 88 条记录及
6 个模型生成的说话人编号；4 个共享窗口加失败句重试后导出 73 个切片，最长 13 秒。
原文顺序保留，WAV 非空且时长与清单一致。47 个片段被标为待复核；此验证没有人工参考时间轴，
不代表已测得边界误差、说话人错误率或转写准确率，也未验证一小时音频。
