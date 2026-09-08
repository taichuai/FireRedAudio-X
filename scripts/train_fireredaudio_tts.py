#!/usr/bin/env python3
"""FireRedAudio tts supervised fine-tuning."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fireredaudio.training.trainer import main


if __name__ == "__main__":
    main("tts")
