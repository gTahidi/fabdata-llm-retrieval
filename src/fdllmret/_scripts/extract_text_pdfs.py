# %%
from pathlib import Path
import argparse

from fdllm.sysutils import register_models

from ..helpers.extraction import process_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        required=True,
        help="The path to the folder (or folders) of pdfs",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--n-jobs",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "--extract-refs",
        action="store_true",
    )
    parser.add_argument(
        "--download-refs",
        action="store_true",
    ),
    parser.add_argument(
        "--extraction-model",
        required=False,
        type=str,
        default="gpt-4-1106-preview",
    )
    parser.add_argument(
        "--custom-models-config",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        required=False,
        default=10,
    )
    args = parser.parse_args()
    args = args._get_args()
    jsondata, contents = process_folder(**vars(args))
