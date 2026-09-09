# FireRedAudio speaker timeline tool

`scripts/fireredaudio_timeline.py` converts FireRedAudio's chronological output
into speaker-attributed WAV clips. FireRed remains the source of transcript text,
interjections, and `spk_N`; the optional Qwen ForcedAligner only refines timing.
It does not perform speaker clustering.

The input text must contain lines such as:

```text
[00:12.340-00:15.120] spk_2: 这里是转写文本。
```

The tool merges adjacent same-speaker entries with a gap up to 800ms, keeps each
merged span below 15 seconds, organizes the entries into 90-150 second alignment
windows, and adds bounded head/tail padding. Different speakers receive a quiet
boundary guard and are never merged. Every output line and WAV is chronological.

The aligner model is optional. Pass a Hugging Face model name or a local model
directory; no machine-specific path is embedded in the script:

```bash
uv run python scripts/fireredaudio_timeline.py \
  --audio recording.wav \
  --firered-text firered_output.txt \
  --output results/timeline \
  --aligner-model Qwen/Qwen3-ForcedAligner-0.6B
```

Without `--aligner-model`, the tool exports FireRed's coarse time with the same
speaker merge and boundary guards. With the aligner, word boundaries are refined
within the large windows. The output contains `segments.jsonl`, `transcript.txt`,
`review.html`, `run.json`, and numbered WAV files. Each segment includes `risks`
for alignment failures, coarse-only timing, speaker boundary guards, or fallback
intervals. A risk flag means the clip needs listening review; it is not an automatic
quality score.

If a single merged FireRed group exceeds the maximum window, it is retained and
marked \`owner_window_over_max\` instead of being silently sent as an oversized
request. The public script applies the optional aligner to FireRed groups within
their owner windows and records window and risk metadata.

For production data, keep the original FireRed response and audio next to the
exported output. The tool does not overwrite input files.
