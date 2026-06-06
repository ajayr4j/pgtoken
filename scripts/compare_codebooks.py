#!/usr/bin/env python3
"""
compare_codebooks.py
--------------------
Compares two pgtoken codebook CSVs to study how token frequency
rankings shifted between corpus versions.

Answers:
  - Which tokens moved from 2-byte to 1-byte encoding (rank 128+ → rank <128)?
  - Which tokens moved from 3-byte to 2-byte (rank 16512+ → rank <16512)?
  - What is the average rank shift across the full vocabulary?
  - How does estimated bytes/token change between codebooks?
  - Top gainers (rank improved most) and losers (rank degraded most)

Usage:
    python scripts/compare_codebooks.py \
        --old data/cl100k_base_codebook.csv \
        --new data/cl100k_base_wildchat4.8M_codebook.csv

    python scripts/compare_codebooks.py \
        --old data/cl100k_base_codebook.csv \
        --new data/cl100k_base_wildchat4.8M_codebook.csv \
        --tokenizer cl100k_base \
        --top 30
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Optional


# varint byte cost thresholds (from pgtoken.c)
def varint_bytes(rank: int) -> int:
    if rank < 128:
        return 1
    elif rank < 16512:
        return 2
    else:
        return 3


def load_codebook(path: Path) -> dict[int, int]:
    """
    Load CSV → {token_id: frequency_rank}.
    Handles two formats:
      Format A (new): token_id, frequency_rank  — 0-based rank
      Format B (old): rank, token_id, frequency, ...  — 1-based rank
    """
    codebook = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        if "frequency_rank" in headers:
            # new format: token_id, frequency_rank
            for row in reader:
                codebook[int(row["token_id"])] = int(row["frequency_rank"])

        elif "rank" in headers and "token_id" in headers:
            # old format: rank, token_id, frequency, ... (1-based → convert to 0-based)
            for row in reader:
                codebook[int(row["token_id"])] = int(row["rank"]) - 1

        else:
            raise ValueError(
                f"Unrecognised codebook format in {path.name}. "
                f"Headers: {headers}"
            )

    return codebook


def tier(rank: int) -> str:
    if rank < 128:
        return "1-byte"
    elif rank < 16512:
        return "2-byte"
    else:
        return "3-byte"


def load_token_text(tokenizer_name: Optional[str], tokenizer_hf: Optional[str]) -> dict[int, str]:
    """Optional: load human-readable token text for display."""
    if tokenizer_hf:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_hf)
            vocab = tok.get_vocab()
            return {v: repr(k) for k, v in vocab.items()}
        except Exception:
            return {}
    elif tokenizer_name:
        try:
            import tiktoken
            enc = tiktoken.get_encoding(tokenizer_name)
            result = {}
            for tid in range(enc.n_vocab):
                try:
                    result[tid] = repr(enc.decode_single_token_bytes(tid).decode("utf-8", errors="replace"))
                except Exception:
                    result[tid] = f"<id:{tid}>"
            return result
        except Exception:
            return {}
    return {}


def print_section(title: str):
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")


def main():
    parser = argparse.ArgumentParser(description="Compare two pgtoken codebook CSVs")
    parser.add_argument("--old",          required=True, help="Path to old codebook CSV")
    parser.add_argument("--new",          required=True, help="Path to new codebook CSV")
    parser.add_argument("--tokenizer",    default=None,  help="tiktoken encoding name for token text display")
    parser.add_argument("--tokenizer-hf", default=None,  help="HuggingFace model name for token text display")
    parser.add_argument("--top",          type=int, default=20, help="Top N movers to show (default: 20)")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)

    print(f"\n{'='*64}")
    print(f"  pgtoken codebook comparison")
    print(f"{'='*64}")
    print(f"  Old : {old_path.name}")
    print(f"  New : {new_path.name}")

    print("\nLoading codebooks...")
    old_cb = load_codebook(old_path)
    new_cb = load_codebook(new_path)

    print(f"  Old codebook : {len(old_cb):,} tokens")
    print(f"  New codebook : {len(new_cb):,} tokens")

    # optional token text
    token_text = load_token_text(args.tokenizer, args.tokenizer_hf)
    has_text = bool(token_text)

    # common tokens
    common_ids = set(old_cb.keys()) & set(new_cb.keys())
    only_old   = set(old_cb.keys()) - set(new_cb.keys())
    only_new   = set(new_cb.keys()) - set(old_cb.keys())

    print(f"  Common tokens: {len(common_ids):,}")
    if only_old:
        print(f"  Only in old  : {len(only_old):,}")
    if only_new:
        print(f"  Only in new  : {len(only_new):,}")

    # ── Tier distribution ──────────────────────────────────────
    print_section("Tier distribution (varint byte cost)")

    def tier_counts(cb):
        t = {"1-byte": 0, "2-byte": 0, "3-byte": 0}
        for rank in cb.values():
            t[tier(rank)] += 1
        return t

    old_tiers = tier_counts(old_cb)
    new_tiers = tier_counts(new_cb)

    print(f"  {'Tier':<12} {'Old':>10} {'New':>10} {'Delta':>10}")
    print(f"  {'─'*44}")
    for t in ["1-byte", "2-byte", "3-byte"]:
        delta = new_tiers[t] - old_tiers[t]
        sign  = "+" if delta >= 0 else ""
        print(f"  {t:<12} {old_tiers[t]:>10,} {new_tiers[t]:>10,} {sign}{delta:>9,}")

    # ── Tier promotions / demotions ────────────────────────────
    print_section("Tier changes (encoding cost shifts)")

    promotions   = {}   # token_id → (old_tier, new_tier, old_rank, new_rank)
    demotions    = {}
    stable       = 0

    for tid in common_ids:
        old_r = old_cb[tid]
        new_r = new_cb[tid]
        old_t = tier(old_r)
        new_t = tier(new_r)
        if old_t == new_t:
            stable += 1
        elif varint_bytes(new_r) < varint_bytes(old_r):
            promotions[tid] = (old_t, new_t, old_r, new_r)
        else:
            demotions[tid]  = (old_t, new_t, old_r, new_r)

    print(f"  Promoted (cheaper encoding) : {len(promotions):,}")
    print(f"  Demoted  (costlier encoding): {len(demotions):,}")
    print(f"  Stable tier                 : {stable:,}")

    # breakdown of promotions
    promo_breakdown = {}
    for old_t, new_t, _, _ in promotions.values():
        key = f"{old_t} → {new_t}"
        promo_breakdown[key] = promo_breakdown.get(key, 0) + 1

    demo_breakdown = {}
    for old_t, new_t, _, _ in demotions.values():
        key = f"{old_t} → {new_t}"
        demo_breakdown[key] = demo_breakdown.get(key, 0) + 1

    if promo_breakdown:
        print("\n  Promotion breakdown:")
        for k, v in sorted(promo_breakdown.items()):
            print(f"    {k:<20} {v:>8,} tokens")

    if demo_breakdown:
        print("\n  Demotion breakdown:")
        for k, v in sorted(demo_breakdown.items()):
            print(f"    {k:<20} {v:>8,} tokens")

    # ── Estimated bytes/token ──────────────────────────────────
    print_section("Estimated bytes per token")

    # weight by rank-based approximation of frequency
    # lower rank = more frequent; use 1/log(rank+2) as frequency proxy
    def estimated_bpt(cb):
        total_weight = 0.0
        total_bytes  = 0.0
        for rank in cb.values():
            weight = 1.0 / math.log2(rank + 2)
            total_weight += weight
            total_bytes  += weight * varint_bytes(rank)
        return total_bytes / total_weight if total_weight else 0

    old_bpt = estimated_bpt(old_cb)
    new_bpt = estimated_bpt(new_cb)
    improvement = (old_bpt - new_bpt) / old_bpt * 100

    print(f"  Old codebook : {old_bpt:.4f} bytes/token (estimated)")
    print(f"  New codebook : {new_bpt:.4f} bytes/token (estimated)")
    print(f"  Improvement  : {improvement:+.2f}%")
    print()
    print("  Note: uses 1/log(rank+2) as frequency proxy. Run against")
    print("  actual token sequences for precise measurement.")

    # ── Rank shift statistics ──────────────────────────────────
    print_section("Rank shift statistics (common tokens)")

    shifts = []
    for tid in common_ids:
        old_r = old_cb[tid]
        new_r = new_cb[tid]
        shifts.append(new_r - old_r)   # negative = improved (lower rank = more frequent)

    shifts.sort()
    n = len(shifts)
    mean_shift    = sum(shifts) / n
    median_shift  = shifts[n // 2]
    improved      = sum(1 for s in shifts if s < 0)
    degraded      = sum(1 for s in shifts if s > 0)
    unchanged     = sum(1 for s in shifts if s == 0)

    print(f"  Mean rank shift   : {mean_shift:+.1f}  (negative = improved)")
    print(f"  Median rank shift : {median_shift:+.1f}")
    print(f"  Improved rank     : {improved:,}  ({improved/n*100:.1f}%)")
    print(f"  Degraded rank     : {degraded:,}  ({degraded/n*100:.1f}%)")
    print(f"  Unchanged         : {unchanged:,}  ({unchanged/n*100:.1f}%)")

    # ── Top movers ─────────────────────────────────────────────
    N = args.top

    top_improved = sorted(common_ids, key=lambda t: new_cb[t] - old_cb[t])[:N]
    top_degraded = sorted(common_ids, key=lambda t: new_cb[t] - old_cb[t], reverse=True)[:N]

    def tok_label(tid):
        return token_text.get(tid, f"<id:{tid}>")

    print_section(f"Top {N} most improved tokens (rank dropped = more frequent in new corpus)")
    header = f"  {'token_id':<10} {'old_rank':>10} {'new_rank':>10} {'shift':>10} {'old_tier':>10} {'new_tier':>10}"
    if has_text:
        header += f"  {'text'}"
    print(header)
    print(f"  {'─'*62}")
    for tid in top_improved:
        old_r = old_cb[tid]
        new_r = new_cb[tid]
        shift = new_r - old_r
        row = f"  {tid:<10} {old_r:>10,} {new_r:>10,} {shift:>+10,} {tier(old_r):>10} {tier(new_r):>10}"
        if has_text:
            row += f"  {tok_label(tid)}"
        print(row)

    print_section(f"Top {N} most degraded tokens (rank rose = less frequent in new corpus)")
    print(header)
    print(f"  {'─'*62}")
    for tid in top_degraded:
        old_r = old_cb[tid]
        new_r = new_cb[tid]
        shift = new_r - old_r
        row = f"  {tid:<10} {old_r:>10,} {new_r:>10,} {shift:>+10,} {tier(old_r):>10} {tier(new_r):>10}"
        if has_text:
            row += f"  {tok_label(tid)}"
        print(row)

    # ── 1-byte zone analysis ───────────────────────────────────
    print_section("1-byte zone analysis (ranks 0–127, highest value tokens)")

    old_1byte = {tid for tid, r in old_cb.items() if r < 128}
    new_1byte = {tid for tid, r in new_cb.items() if r < 128}

    entered_1byte = new_1byte - old_1byte
    left_1byte    = old_1byte - new_1byte
    stayed_1byte  = old_1byte & new_1byte

    print(f"  In 1-byte zone (old)     : {len(old_1byte):,}")
    print(f"  In 1-byte zone (new)     : {len(new_1byte):,}")
    print(f"  Entered 1-byte zone      : {len(entered_1byte):,}")
    print(f"  Left 1-byte zone         : {len(left_1byte):,}")
    print(f"  Stable in 1-byte zone    : {len(stayed_1byte):,}")

    if entered_1byte and has_text:
        sample = sorted(entered_1byte, key=lambda t: new_cb[t])[:10]
        print(f"\n  Sample entering 1-byte zone (by new rank):")
        for tid in sample:
            print(f"    rank {new_cb[tid]:>4}  id={tid:<8}  {tok_label(tid)}")

    if left_1byte and has_text:
        sample = sorted(left_1byte, key=lambda t: old_cb[t])[:10]
        print(f"\n  Sample leaving 1-byte zone (by old rank):")
        for tid in sample:
            print(f"    old_rank {old_cb[tid]:>4}  new_rank {new_cb[tid]:>6}  id={tid:<8}  {tok_label(tid)}")

    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()