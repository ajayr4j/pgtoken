#!/usr/bin/env python3
"""
build_codebook.py
-----------------
Builds a pgtoken-compatible frequency codebook CSV from a text corpus.

Supports two corpus sources:
  1. HuggingFace Arrow dataset (WildChat or similar)
  2. Postgres table with a text column

Supports two tokenizers:
  1. tiktoken  -- cl100k_base, o200k_base (OpenAI models)
  2. HuggingFace AutoTokenizer -- any model (Qwen, Llama, Mistral, etc.)

Output: a CSV with columns [token_id, frequency_rank]
  rank 0 = most frequent token in the corpus
  All vocabulary tokens included -- unseen tokens ranked last by token_id

Usage:
    # From WildChat dataset, 10 cores
    python scripts/build_codebook.py \\
        --source arrow \\
        --data-dir ./wildchat_data \\
        --tokenizer cl100k_base \\
        --output data/cl100k_base_codebook.csv \\
        --workers 10

    # From WildChat, Qwen tokenizer, 8 cores
    python scripts/build_codebook.py \\
        --source arrow \\
        --data-dir ./wildchat_data \\
        --tokenizer-hf Qwen/Qwen2.5-1.5B-Instruct \\
        --output data/qwen25_codebook.csv \\
        --workers 8

    # From Postgres table, 4 cores
    python scripts/build_codebook.py \\
        --source postgres \\
        --dsn "postgresql://user:pass@localhost/mydb" \\
        --table my_table --column content \\
        --tokenizer cl100k_base \\
        --output data/cl100k_base_codebook.csv \\
        --workers 4

    # Quick test with 100k texts, 4 cores
    python scripts/build_codebook.py \\
        --source arrow \\
        --data-dir ./wildchat_data \\
        --tokenizer cl100k_base \\
        --limit 100000 \\
        --workers 4 \\
        --output data/cl100k_base_codebook.csv
"""

import argparse
import csv
import gc
import json
import math
import multiprocessing
import os
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Optional

STOP_REQUESTED = False


def handle_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nStop requested. Will finish current batch and exit cleanly.")


signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)


# ================================================================
# Tokenizer loader
# ================================================================
def load_tokenizer(tokenizer_name: Optional[str], tokenizer_hf: Optional[str]):
    """Load either a tiktoken or HuggingFace tokenizer. Returns (encode_fn, vocab_size, all_token_ids)."""
    if tokenizer_hf:
        from transformers import AutoTokenizer
        print(f"Loading HuggingFace tokenizer: {tokenizer_hf}")
        tok = AutoTokenizer.from_pretrained(tokenizer_hf)
        vocab = tok.get_vocab()
        all_ids = set(vocab.values())

        def encode(text):
            return tok.encode(text, add_special_tokens=False)

        return encode, tok.vocab_size, all_ids

    elif tokenizer_name:
        import tiktoken
        print(f"Loading tiktoken tokenizer: {tokenizer_name}")
        enc = tiktoken.get_encoding(tokenizer_name)
        all_ids = set(range(enc.n_vocab))

        def encode(text):
            return enc.encode(text, disallowed_special=())

        return encode, enc.n_vocab, all_ids

    else:
        raise ValueError("Provide either --tokenizer or --tokenizer-hf")


# ================================================================
# Corpus readers
# ================================================================
def iter_texts_from_arrow(data_dir: str, limit: Optional[int]):
    """Yield message content strings from a HuggingFace Arrow dataset."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    data_path = Path(data_dir)
    arrow_files = sorted(data_path.glob("data-*.arrow"))
    if not arrow_files:
        arrow_files = sorted(data_path.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {data_dir}")

    print(f"Found {len(arrow_files)} Arrow file(s)")
    yielded = 0

    for arrow_file in arrow_files:
        print(f"  Reading {arrow_file.name}...")
        with pa.memory_map(str(arrow_file), "r") as src:
            try:
                reader = ipc.open_stream(src)
                batches = list(reader)
            except pa.ArrowInvalid:
                src.seek(0)
                reader = ipc.open_file(src)
                batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]

        for batch in batches:
            schema_names = batch.schema.names
            conversations_col = None
            if "conversation" in schema_names:
                conversations_col = batch.column(
                    batch.schema.get_field_index("conversation")
                ).to_pylist()

            if conversations_col is None:
                del batch
                continue

            for conv in conversations_col:
                if conv is None:
                    continue
                if isinstance(conv, str):
                    try:
                        conv = json.loads(conv)
                    except Exception:
                        continue
                if not isinstance(conv, list):
                    continue
                for msg in conv:
                    if not isinstance(msg, dict):
                        continue
                    content = msg.get("content") or ""
                    content = content.replace("\x00", "")
                    if content.strip():
                        yield content
                        yielded += 1
                        if limit and yielded >= limit:
                            return

            del batch
            gc.collect()

        if STOP_REQUESTED:
            print("Stop requested during Arrow reading.")
            return


def iter_texts_from_postgres(dsn: str, table: str, column: str, limit: Optional[int]):
    """Yield text rows from a Postgres table."""
    import psycopg2
    conn = psycopg2.connect(dsn)
    query = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL AND length({column}) > 0"
    if limit:
        query += f" LIMIT {limit}"
    with conn.cursor(name="codebook_cursor") as cur:
        cur.itersize = 2000
        cur.execute(query)
        for row in cur:
            content = (row[0] or "").replace("\x00", "")
            if content.strip():
                yield content
    conn.close()


# ================================================================
# Worker function — runs in each subprocess
# ================================================================

def _worker_init(tokenizer_name, tokenizer_hf):
    """
    Called once per worker process at startup.
    Loads the tokenizer into a module-level global so it isn't
    re-serialised on every task.
    """
    global _worker_encode_fn
    if tokenizer_hf:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_hf)
        _worker_encode_fn = lambda text: tok.encode(text, add_special_tokens=False)
    else:
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer_name)
        _worker_encode_fn = lambda text: enc.encode(text, disallowed_special=())


def _worker_encode_batch(texts: list[str]) -> Counter:
    """Encode a batch of texts and return a Counter of token IDs."""
    counter = Counter()
    for text in texts:
        try:
            counter.update(_worker_encode_fn(text))
        except Exception:
            pass
    return counter


# ================================================================
# Core: count frequencies (parallel)
# ================================================================
BATCH_SIZE = 500   # texts per task submitted to the pool


def count_frequencies_parallel(
    texts,
    tokenizer_name: Optional[str],
    tokenizer_hf: Optional[str],
    workers: int,
) -> tuple[Counter, int]:
    """
    Distribute tokenization across `workers` processes.
    Each worker loads its own tokenizer instance — no GIL contention.
    Results are merged into a single Counter in the main process.
    """
    counter  = Counter()
    n_texts  = 0
    t0       = time.time()

    # initialiser args must be picklable — strings are fine
    init_args = (tokenizer_name, tokenizer_hf)

    with multiprocessing.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=init_args,
    ) as pool:
        batch = []
        futures = []

        for text in texts:
            batch.append(text)
            if len(batch) >= BATCH_SIZE:
                futures.append(pool.apply_async(_worker_encode_batch, (batch,)))
                batch = []

            # collect completed futures periodically to keep memory bounded
            if len(futures) >= workers * 4:
                for f in futures:
                    partial = f.get()
                    counter.update(partial)
                    n_texts += len(partial) // max(1, sum(partial.values()) // max(1, BATCH_SIZE))
                futures = []

                elapsed = time.time() - t0
                total_tokens = sum(counter.values())
                if total_tokens > 0:
                    print(f"  ~{n_texts*BATCH_SIZE:>8,} texts | "
                          f"{total_tokens:>12,} tokens | "
                          f"{total_tokens/elapsed:>10,.0f} tok/s | "
                          f"{workers} workers")

            if STOP_REQUESTED:
                pool.terminate()
                break

        # flush remaining batch
        if batch and not STOP_REQUESTED:
            futures.append(pool.apply_async(_worker_encode_batch, (batch,)))

        # collect all remaining futures
        for f in futures:
            try:
                partial = f.get()
                counter.update(partial)
            except Exception:
                pass

    # n_texts is approximated above; recount from counter for accuracy
    total_tokens = sum(counter.values())
    elapsed = time.time() - t0
    print(f"  Done. {total_tokens:,} tokens in {elapsed:.1f}s "
          f"({total_tokens/elapsed:,.0f} tok/s, {workers} workers)")

    return counter, n_texts


# ================================================================
# Output writers
# ================================================================
def write_codebook_csv(counter: Counter, all_token_ids: set, output_path: Path):
    """
    Write pgtoken-compatible codebook CSV.
    All vocabulary tokens included -- unseen tokens go at the end sorted by token_id.
    rank 0 = most frequent.
    """
    seen_ids   = set(counter.keys())
    unseen_ids = sorted(all_token_ids - seen_ids)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "frequency_rank"])
        rank = 0
        for token_id, _ in counter.most_common():
            writer.writerow([token_id, rank])
            rank += 1
        for token_id in unseen_ids:
            writer.writerow([token_id, rank])
            rank += 1

    print(f"Codebook written: {output_path}")
    print(f"  Seen tokens    : {len(seen_ids):,}")
    print(f"  Unseen tokens  : {len(unseen_ids):,}  (ranked last, still included)")
    print(f"  Total tokens   : {rank:,}")


def write_stats_csv(counter: Counter, n_texts: int, output_path: Path):
    """Write full frequency stats for analysis."""
    total_tokens = sum(counter.values())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "token_id", "frequency", "probability_pct", "self_information_bits"
        ])
        writer.writeheader()
        for rank, (token_id, freq) in enumerate(counter.most_common(), 1):
            p = freq / total_tokens if total_tokens else 0
            writer.writerow({
                "rank":                  rank,
                "token_id":              token_id,
                "frequency":             freq,
                "probability_pct":       round(p * 100, 6),
                "self_information_bits": round(-math.log2(p), 4) if p > 0 else None,
            })
    print(f"Stats CSV written: {output_path}")


# ================================================================
# Entry point
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build pgtoken frequency codebook from a text corpus"
    )
    parser.add_argument("--source", choices=["arrow", "postgres"], required=True,
                        help="Corpus source: arrow (HuggingFace dataset) or postgres")

    # Arrow source
    parser.add_argument("--data-dir", help="Directory containing .arrow files")

    # Postgres source
    parser.add_argument("--dsn",    help="Postgres DSN")
    parser.add_argument("--table",  help="Table name")
    parser.add_argument("--column", default="content", help="Text column name")

    # Tokenizer
    parser.add_argument("--tokenizer",    default="cl100k_base",
                        help="tiktoken encoding name (default: cl100k_base)")
    parser.add_argument("--tokenizer-hf", default=None,
                        help="HuggingFace model name (overrides --tokenizer)")

    # Output
    parser.add_argument("--output", default="data/cl100k_base_codebook.csv",
                        help="Output codebook CSV path")
    parser.add_argument("--output-stats", default=None,
                        help="Optional full stats CSV path")

    # Parallelism
    parser.add_argument("--workers", type=int,
                        default=max(1, os.cpu_count() - 1),
                        help="Number of worker processes for tokenization "
                             "(default: cpu_count - 1). Use 1 for single-threaded.")

    # Limits
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of texts to process (default: all)")

    args = parser.parse_args()

    print("=" * 60)
    print("pgtoken codebook builder")
    print("=" * 60)
    print(f"Source     : {args.source}")
    print(f"Tokenizer  : {args.tokenizer_hf or args.tokenizer}")
    print(f"Output     : {args.output}")
    print(f"Workers    : {args.workers}")
    print(f"Limit      : {args.limit or 'all'}")
    print()

    # load tokenizer in main process to get vocab size and all_token_ids
    _, vocab_size, all_token_ids = load_tokenizer(
        args.tokenizer if not args.tokenizer_hf else None,
        args.tokenizer_hf,
    )
    print(f"Vocab size : {vocab_size:,}")
    print()

    # get corpus iterator
    if args.source == "arrow":
        if not args.data_dir:
            parser.error("--data-dir required for --source arrow")
        texts = iter_texts_from_arrow(args.data_dir, args.limit)
    else:
        if not args.dsn or not args.table:
            parser.error("--dsn and --table required for --source postgres")
        texts = iter_texts_from_postgres(args.dsn, args.table, args.column, args.limit)

    # count
    print(f"Counting token frequencies with {args.workers} worker(s)...")
    t0 = time.time()
    counter, n_texts = count_frequencies_parallel(
        texts,
        args.tokenizer if not args.tokenizer_hf else None,
        args.tokenizer_hf,
        args.workers,
    )
    elapsed = time.time() - t0

    total_tokens = sum(counter.values())
    print()
    print(f"Total tok  : {total_tokens:,}  ({total_tokens/elapsed:,.0f} tok/s)")
    print(f"Unique tok : {len(counter):,} / {vocab_size:,}")
    print()

    # top 10
    print("Top 10 tokens by frequency:")
    for rank, (tid, freq) in enumerate(counter.most_common(10), 1):
        print(f"  {rank:>2}. id={tid:<8} freq={freq:>12,}  ({freq/total_tokens*100:.3f}%)")
    print()

    # write outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_codebook_csv(counter, all_token_ids, output_path)

    if args.output_stats:
        write_stats_csv(counter, n_texts, Path(args.output_stats))

    print()
    print("Next step: install codebook to Postgres")
    print(f"  python setup_codebook.py --csv {args.output} --name {Path(args.output).stem}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()