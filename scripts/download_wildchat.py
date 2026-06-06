#!/usr/bin/env python3
"""
download_wildchat.py
--------------------
Downloads a HuggingFace dataset and saves it to disk.

Usage:
    python scripts/download_wildchat.py
    python scripts/download_wildchat.py --dataset allenai/WildChat-4.8M
    python scripts/download_wildchat.py --dataset allenai/WildChat --split train[:10%]
    python scripts/download_wildchat.py --dataset allenai/WildChat-4.8M --split train[:50000]
"""

import argparse
from pathlib import Path
from datasets import load_dataset


DEFAULT_DATASET   = "allenai/WildChat-4.8M"
DEFAULT_OUTPUT    = "./data"
DEFAULT_SPLIT     = "train"


def main():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset for codebook generation"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to download (default: train). Use train[:10%%] for a quick test."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help=f"Root output directory (default: {DEFAULT_OUTPUT}). Dataset saved under data/<dataset_name>/."
    )
    args = parser.parse_args()

    # derive a clean subdir name from the dataset string
    # e.g. "allenai/WildChat-4.8M" → "data/allenai_WildChat-4.8M"
    dataset_slug = args.dataset.replace("/", "_")
    output_dir   = Path(args.output_dir) / dataset_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset  : {args.dataset}")
    print(f"Split    : {args.split}")
    print(f"Output   : {output_dir}")
    print()
    print("Downloading... (HuggingFace will cache on first run)")
    print()

    ds = load_dataset(args.dataset, split=args.split)

    print(f"Loaded   : {len(ds):,} rows")
    print(f"Columns  : {ds.column_names}")
    print()

    print(f"Saving to {output_dir} ...")
    ds.save_to_disk(str(output_dir))

    print(f"Done.")
    print()
    print("Next step:")
    print(f"  python scripts/build_codebook.py --source arrow --data-dir {output_dir} --tokenizer-hf <model>")


if __name__ == "__main__":
    main()