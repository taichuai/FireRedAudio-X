"""Start SGLang with visual preprocessing disabled for embedding requests."""

import os
import sys


def main():
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = prepare_server_args(sys.argv[1:])
    # The 0.5.9 CLI exposes only --enable-multimodal, with no false spelling.
    args.enable_multimodal = False
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
