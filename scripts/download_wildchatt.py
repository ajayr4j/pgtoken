#!/usr/bin/env python3
"""
download_wildchat.py
--------------------
Downloads the WildChat dataset from HuggingFace and saves it to disk.

WildChat (allenai/WildChat) is a 1M+ real user conversation dataset
collected from ChatGPT users. It covers diverse domains — technical,
creative, analytical, conversational — making it a strong general-purpose
corpus for building token frequency codebooks.

Why WildChat for codebook generation:
  - 1M+ real conversations, not synthetic
  - Diverse domains and writing styles
  - Covers the same vocabulary distribution as real LLM usage
  - Freely available on HuggingFace
  - Large enough that frequency rankings stabilize

For domain-specific corpora (medical, legal, financial), substitute
your own text and use build_codebook.py directly.

Usage:
    python scripts/download_wildchat.py
    python scripts/download_wildchat.py --output-dir ./wildchat_data
    python scripts/download_wildchat.py --split train[:10%]   # 10% for testing
"""

import argparse
from pathlib import Path
from datasets import load_dataset


DEFAULT_OUTPUT = "./wildchat_data"


def main():
    parser = argparse.ArgumentParser(
        description="Download WildChat dataset for codebook generation"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help=f"Where to save the dataset (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (default: train). Use train[:10%%] for a quick test."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading WildChat ({args.split})...")
    print("This may take a few minutes on first download.")
    print("Dataset will be cached by HuggingFace for future runs.")
    print()

    ds = load_dataset("allenai/WildChat", split=args.split)

    print(f"Dataset loaded: {len(ds):,} conversations")
    print(f"Columns: {ds.column_names}")
    print(f"Sample entry keys: {list(ds[0].keys())}")
    print()

    print(f"Saving to {output_dir}...")
    ds.save_to_disk(str(output_dir))

    print(f"Done. Dataset saved to: {output_dir}")
    print()
    print("Next step:")
    print(f"  python scripts/build_codebook.py --data-dir {output_dir} --tokenizer cl100k_base")


if __name__ == "__main__":
    main()