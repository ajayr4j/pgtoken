# pgtoken

PostgreSQL extension for rank-varint token storage.
Store LLM token IDs compactly as bytea. No re-tokenization on read.
Works alongside pgvector.

---

## Functions

```sql
pgtoken_encode(token_ids integer[], codebook text) -> bytea
pgtoken_decode(encoded bytea, codebook text)        -> integer[]
pgtoken_count(encoded bytea)                        -> integer   -- O(1), header only
pgtoken_reload_codebooks()                          -> void
```

---

## Writing

**[pgtoken: storing what you already know](https://ajayr4j.substack.com/p/pgtoken-storing-what-you-already)**
Technical internals - varint encoding, codebook design, C extension architecture, use cases.

**[How pgtoken recovers GPU time by fixing what runs before it](https://ajayr4j.substack.com/p/how-pgtoken-recovers-gpu-time-by)**
Benchmark results  83% tokenizer reduction at concurrency=1, 97.5% at concurrency=100, P99 113ms → 0.62ms.


---

## Quick start

```sql
-- Store
INSERT INTO chunks (content, token_ids)
VALUES ('HDFC Q3 results...', pgtoken_encode(ARRAY[39308, 1229, 2632], 'cl100k_base'));

-- Count without decoding
SELECT pgtoken_count(token_ids) FROM chunks;

-- Context window filter (no tokenization)
SELECT id, content FROM chunks
WHERE pgtoken_count(token_ids) <= 1024;

-- Recover IDs
SELECT pgtoken_decode(token_ids, 'cl100k_base') FROM chunks;
```

---

## Repository structure

```
pgtoken/
|-- pgtoken.c                       core C extension
|-- pgtoken--1.0.sql                SQL function definitions
|-- pgtoken.control                 extension metadata
|-- Makefile                        build file
|-- scripts/setup_codebook.py               installs codebook CSV to Postgres data dir
|
|-- data/
|   |-- README.md                            codebook format and generation guide
|   |-- cl100k_base_codebook.csv             pre-built codebook (WildChat 529K, default)
|   +-- cl100k_base_wildchat4.8M_codebook.csv  codebook built from WildChat-4.8M (3.2M)
|
|-- scripts/
|   |-- download_wildchat.py        downloads WildChat dataset from HuggingFace
|   |-- build_codebook.py           builds codebook CSV from any corpus
|   |-- compare_codebooks.py        compares two codebook versions, studies rank shifts
|   +-- setup_codebook.py           installs codebook CSV to Postgres data dir
|
+-- sql/
    +-- pgtoken_test.sql            test queries
```

---

## Installation

### 1. Prerequisites

```bash
# Ubuntu/Debian
sudo apt install postgresql-server-dev-18 build-essential

# macOS
brew install postgresql@18
```

### 2. Install codebook CSV

A pre-built `cl100k_base` codebook is included in `data/`. Install it directly:

```bash
python3 scripts/setup_codebook.py \
    --csv data/cl100k_base_codebook.csv \
    --name cl100k_base
```

This copies the CSV to `$PGDATA/pgtoken_codebooks/cl100k_base.csv`.

If auto-detection fails:
```bash
python3 scripts/setup_codebook.py \
    --csv data/cl100k_base_codebook.csv \
    --pgdata /var/lib/postgresql/18/main
```

**Building your own codebook (optional):**

The pre-built `cl100k_base` codebook (`data/cl100k_base_codebook.csv`) was generated
from [WildChat](https://huggingface.co/datasets/allenai/WildChat) 529K real GPT-3.5
and GPT-4 conversations across 66 languages.

A larger version, [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M),
contains 3.2M conversations (6× larger). We validated the pre-built codebook against
a codebook generated from WildChat-4.8M. The result: **90.2% of tokens stayed in the
same varint byte tier** across both versions. The core frequency distribution is stable.
The 4.8M codebook showed a slight shift toward code and JSON tokens and away from
multilingual tokens making the original WildChat a better default for general-purpose
conversational workloads.

Both CSVs are included in `data/` for reference:

| File | Source corpus | Tokens |
|---|---|---|
| `cl100k_base_codebook.csv` | WildChat (529K conversations) | 98,507 |
| `cl100k_base_wildchat4.8M_codebook.csv` | WildChat-4.8M (3.2M conversations) | 100,277 |

For domain-specific corpora (medical, legal, financial, code-heavy), build your own:

```bash
# Download WildChat-4.8M (~15GB, one-time) or use your own corpus
python3 scripts/download_wildchat.py --dataset allenai/WildChat-4.8M

# Build codebook for tiktoken
python3 scripts/build_codebook.py \
    --source arrow \
    --data-dir data/allenai_WildChat-4.8M \
    --tokenizer cl100k_base \
    --output data/cl100k_base_wildchat4.8M_codebook.csv \
    --workers 10

# Compare against existing codebook to study distribution shift
python3 scripts/compare_codebooks.py \
    --old data/cl100k_base_codebook.csv \
    --new data/cl100k_base_wildchat4.8M_codebook.csv \
    --tokenizer cl100k_base

# Build codebook for HuggingFace tokenizers (Qwen, Llama, etc.)
python3 scripts/build_codebook.py \
    --source arrow \
    --data-dir data/allenai_WildChat-4.8M \
    --tokenizer-hf Qwen/Qwen2.5-1.5B-Instruct \
    --output data/qwen25_codebook.csv \
    --workers 10
```

See `data/README.md` for full details on codebook generation and when to use a custom corpus.

### 3. Build

```bash
cd pgtoken
make
sudo make install
```

### 4. Create extension

```sql
CREATE EXTENSION pgtoken;
```

### 5. Verify

```sql
SELECT pgtoken_count(
    pgtoken_encode(ARRAY[1639, 389, 257], 'cl100k_base')
);
-- returns: 3
```

---

## CSV format

Your codebook CSV must have this header and format:

```
token_id,frequency_rank
1639,0
389,1
257,2
...
```

rank 0 = most frequent token. This is the output format from pgtoken_py.

Check yours before building:
```bash
head -3 data/cl100k_base_codebook.csv
```

---

## Schema pattern with pgvector

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgtoken;

CREATE TABLE chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL,
    content     text NOT NULL,
    token_ids   bytea,
    embedding   vector(1536),
    chunk_order int NOT NULL
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

---

## Varint encoding

```
rank 0-127      -> 1 byte
rank 128-16511  -> 2 bytes
rank 16512+     -> 3 bytes
```

4-byte LE count header prepended. `pgtoken_count()` reads only those 4 bytes.
Average ~1.7 bytes/token vs 4 bytes for raw uint32.

---

## Multiple codebooks

Add more CSVs to `$PGDATA/pgtoken_codebooks/`:
```bash
cp o200k_base_freq.csv $PGDATA/pgtoken_codebooks/o200k_base.csv
```

Then use by name:
```sql
SELECT pgtoken_encode(ids, 'o200k_base') FROM ...;
```

---

## Reload after CSV update

No Postgres restart needed:
```sql
SELECT pgtoken_reload_codebooks();
```

---

## Roadmap

- pgtoken_text() - decode directly to text
- pgtoken_concat() aggregate for prompt assembly
- pgxn.org packaging