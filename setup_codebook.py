#!/usr/bin/env python3
"""
setup_codebook.py
-----------------
Copies your frequency CSV into $PGDATA/pgtoken_codebooks/
where the Postgres extension expects it.

Usage:
    python3 setup_codebook.py \
        --csv token_frequency_cl100k_base_from_varint.csv \
        --name cl100k_base

    # If PGDATA detection fails, pass it explicitly:
    python3 setup_codebook.py \
        --csv token_frequency_cl100k_base_from_varint.csv \
        --pgdata /var/lib/postgresql/16/main
"""

import argparse
import os
import shutil
import subprocess
import sys


def get_pgdata():
    try:
        r = subprocess.run(
            ["psql", "-tAc", "SHOW data_directory"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True)
    parser.add_argument("--name",   default="cl100k_base")
    parser.add_argument("--pgdata", default=None)
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    pgdata = args.pgdata or get_pgdata()
    if not pgdata:
        print("ERROR: Cannot detect PGDATA. Pass --pgdata /your/pgdata/path")
        sys.exit(1)

    codebooks_dir = os.path.join(pgdata, "pgtoken_codebooks")
    os.makedirs(codebooks_dir, exist_ok=True)

    dest = os.path.join(codebooks_dir, f"{args.name}.csv")
    shutil.copy2(csv_path, dest)

    with open(dest) as f:
        rows = sum(1 for _ in f) - 1
    print(f"Installed: {dest}")
    print(f"Codebook '{args.name}': {rows:,} entries")
    print()
    print("Next:")
    print("  cd pgtoken && make && sudo make install")
    print("  psql -c 'CREATE EXTENSION pgtoken;'")
    print(f"  psql -c \"SELECT pgtoken_count(pgtoken_encode(ARRAY[1639,389,257], '{args.name}'));\"")


if __name__ == "__main__":
    main()
