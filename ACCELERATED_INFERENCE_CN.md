# vLLM / SGLang 加速推理

支持 ASR、音频理解、thinking、多音频和批量请求。音频前端复用 FireRedAudio 的编码器与提示词，
通过 embedding 接口调用独立的 Qwen3.5 文本服务。TTS、语音编辑和音色设计继续使用原生推理。

## 环境

安装命令见 [README](README.md#安装)。音频客户端使用 `.venv`；
两个可选服务分别使用 `.venv-vllm`（vLLM 0.18.0）和 `.venv-sglang`（SGLang 0.5.9）。
Python 3.12 环境可能提示不符合主项目的 Python 3.10 要求，使用 `uv pip` 独立安装即可。

## 导出主干

```bash
uv run python -m fireredaudio.accelerated.prepare \
  --source pretrained_models/FireRedAudio \
  --output pretrained_models/FireRedAudio-backbone
```

导出约占 17.9 GB，输出目录必须不存在。SGLang 子目录通过相对软链接复用权重。
音频前端与导出的主干必须来自同一 checkpoint；微调后需重新导出。

## 启动服务

选择一个后端，GPU 编号按实际设备调整：

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8010 GPU_MEMORY_UTILIZATION=0.6 \
  bash scripts/serve_accelerated.sh vllm
```

或：

```bash
CUDA_VISIBLE_DEVICES=0 PORT=8011 GPU_MEMORY_UTILIZATION=0.6 \
  bash scripts/serve_accelerated.sh sglang
```

可设置 `MODEL_PATH`、`ENGINE_PYTHON`、`FIREREDAUDIO_HOST`、`PORT`、
`TENSOR_PARALLEL_SIZE`、`MAX_MODEL_LEN`（默认 4096）、`MAX_NUM_SEQS`（默认 8）、
`GPU_MEMORY_UTILIZATION`（默认 0.75）。额外框架参数放在脚本末尾。

同时运行时使用不同端口并分配足够显存。客户端与服务共用 GPU 时，须为音频编码器留出空间。
首次启动和请求可能触发较长的内核编译。

vLLM 启动器显式使用 `--mamba-ssm-cache-dtype float32`。FireRedAudio 的 Qwen3.5
配置要求线性注意力 SSM 状态为 FP32；导出的文本包装器使用自定义架构名，vLLM 0.18
不会自动应用上游多模态架构的 dtype 校正规则。模型权重和其余 cache 仍为 BF16。

## 推理

在另一个终端使用原生环境运行客户端：

```bash
CUDA_VISIBLE_DEVICES=1 uv run python inference_accelerated.py \
  --backend vllm --base-url http://127.0.0.1:8010 \
  --task asr --audio assets/examples/asr_zh_fleurs.wav
```

音频理解与 thinking：

```bash
CUDA_VISIBLE_DEVICES=1 uv run python inference_accelerated.py \
  --backend sglang --base-url http://127.0.0.1:8011 \
  --task understand --audio assets/examples/assets_mmau_test.wav \
  --prompt 'Describe the audio in detail.' --enable-thinking --max-new-tokens 2048
```

`--audio a.wav b.wav` 表示同一问题的多个音频。JSONL 每行一个独立请求，路径相对当前工作目录：

```jsonl
{"id":"asr-1","task":"asr","audio":"speech.wav"}
{"id":"qa-1","task":"understand","audio":["a.wav","b.wav"],"prompt":"Compare the speakers."}
```

```bash
CUDA_VISIBLE_DEVICES=1 uv run python inference_accelerated.py \
  --backend vllm --base-url http://127.0.0.1:8010 \
  --input-jsonl assets/examples/accelerated_requests.jsonl \
  --concurrency 4 --max-new-tokens 2048 --output results.jsonl
```

输出包含完整 `text`、解析后的 `answer` / `reasoning`、结束原因和用量。
thinking 默认上限 1024 token，其他任务默认 300；`finish_reason=length` 表示截断。
客户端超时默认 300 秒，可通过 `--timeout` 调整。

## 限制与验证

- 当前验证范围为 A100、BF16、TP=1；尚未完成多卡、长音频和系统性吞吐或准确率测试。
- ASR 默认 greedy，原生入口默认 4-beam，输出不保证一致。
- 默认关闭 prefix/radix cache 和 CUDA graph；SGLang 还关闭 chunked prefill。
- 客户端 `--max-model-len` 应与服务配置一致。默认 SDPA 对长音频显存开销较大，
  可安装兼容的 FlashAttention 后使用 `--attention flash_attention_2`。
- 音频逐条编码，文本请求并发调度；SGLang 的 JSON embedding 传输对长输入开销较大。
  批处理的队首等待和单条失败处理仍待完善。
- SGLang 随机种子由服务端 `SEED` 设置；客户端 `--seed` 用于 vLLM。

测试覆盖权重导出、传输格式，以及相同三轴 MRoPE 与普通 RoPE 的数值一致性：

```bash
uv run python -m unittest discover -s tests -v
.venv-vllm/bin/python -m unittest discover -s tests -p test_accelerated_vllm.py -v
```

接口依据：[vLLM prompt embeddings](https://docs.vllm.ai/en/v0.18.0/features/prompt_embeds/)、
[SGLang GenerateReqInput](https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/srt/managers/io_struct.py)。
