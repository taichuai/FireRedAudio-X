#!/usr/bin/env python3
"""Merge SFT weights into a complete checkpoint accepted by the inference tools."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fireredaudio.training.checkpoints import export_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="original complete base checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    print(export_checkpoint(args.model, args.checkpoint, args.output))


if __name__ == "__main__":
    main()
