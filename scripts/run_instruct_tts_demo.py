#!/usr/bin/env python3
"""Batch Instruct TTS and related generation demos with one model load."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import FireRedAudioInference, set_seed


def resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                case = json.loads(line)
                case.setdefault("line", line_number)
                cases.append(case)
    if not cases:
        raise ValueError(f"empty case file: {path}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("assets/examples/instruct_tts_cases.jsonl"))
    parser.add_argument("--model", default="pretrained_models/FireRedAudio")
    parser.add_argument("--vae-decoder", default="pretrained_models/RedAE_decoder/model.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs/instruct_tts"))
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-new-audio-steps", type=int, default=160)
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--inference-cfg", type=float, default=2.0)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    case_base = args.cases.resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = FireRedAudioInference(
        model_path=args.model,
        vae_decoder_path=args.vae_decoder,
        device=args.device,
    )
    common = {
        "max_new_audio_steps": args.max_new_audio_steps,
        "n_timesteps": args.n_timesteps,
        "inference_cfg": args.inference_cfg,
    }
    results = []
    playlist = []
    for index, case in enumerate(cases, 1):
        case_id = case.get("id", f"case_{index:02d}")
        output_path = args.output_dir / f"{case_id}.wav"
        set_seed(args.seed + index)
        result = {"id": case_id, "task": case.get("task"), "output": str(output_path)}
        print(f"[{index:02d}/{len(cases)}] {case_id} ({case.get('task')})", flush=True)
        try:
            task = case["task"]
            if task == "voice_design":
                output = engine.voice_design(case["instruction"], case["text"], **common)
            elif task == "tts":
                output = engine.tts(
                    prompt_text=case["prompt_text"],
                    prompt_audio=str(resolve(case["prompt_audio"], case_base)),
                    target_text=case["target_text"],
                    language=case.get("language", "zh"),
                    **common,
                )
            elif task == "edit":
                output = engine.edit(
                    str(resolve(case["audio"], case_base)),
                    case["instruction"],
                    edit_type=case.get("edit_type", "semantic"),
                    **common,
                )
            else:
                raise ValueError(f"unsupported task {task!r}")
            torchaudio.save(str(output_path), output.audio.cpu().float(), sample_rate=24000)
            result.update(
                {
                    "ok": True,
                    "seconds": output.audio.shape[-1] / 24000,
                    "text_output": output.text,
                }
            )
            playlist.append(output_path.relative_to(args.output_dir).as_posix())
            print(
                f"    wrote {output_path} ({result['seconds']:.2f}s)"
                + (f" text={output.text!r}" if output.text else ""),
                flush=True,
            )
        except Exception as exc:
            result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"    FAILED: {result['error']}", file=sys.stderr, flush=True)
        results.append(result)

    (args.output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "playlist.m3u").write_text(
        "\n".join(playlist) + "\n", encoding="utf-8"
    )
    summary = {
        "num_cases": len(results),
        "num_success": sum(row["ok"] for row in results),
        "num_failed": sum(not row["ok"] for row in results),
        "seed": args.seed,
        "max_new_audio_steps": args.max_new_audio_steps,
        "n_timesteps": args.n_timesteps,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
