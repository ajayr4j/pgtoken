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
|-- setup_codebook.py               installs codebook CSV to Postgres data dir
|
|-- data/
|   |-- README.md                   codebook format and generation guide
|   +-- cl100k_base_codebook.csv    pre-built codebook (WildChat, 100k tokens)
|
|-- scripts/
|   |-- download_wildchat.py        downloads WildChat dataset from HuggingFace
|   +-- build_codebook.py           builds codebook CSV from any corpus
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
python3 setup_codebook.py \
    --csv data/cl100k_base_codebook.csv \
    --name cl100k_base
```

This copies the CSV to `$PGDATA/pgtoken_codebooks/cl100k_base.csv`.

If auto-detection fails:
```bash
python3 setup_codebook.py \
    --csv data/cl100k_base_codebook.csv \
    --pgdata /var/lib/postgresql/18/main
```

**Building your own codebook (optional):**

The pre-built codebook was generated from WildChat (1M+ real LLM conversations)
and works well for general-purpose text. For domain-specific corpora, build your own:

```bash
# Download WildChat dataset (~5GB, one-time)
python3 scripts/download_wildchat.py

# Build codebook for tiktoken
python3 scripts/build_codebook.py \
    --source arrow \
    --data-dir ./wildchat_data \
    --tokenizer cl100k_base \
    --output data/cl100k_base_codebook.csv

# Build codebook for HuggingFace tokenizers (Qwen, Llama, etc.)
python3 scripts/build_codebook.py \
    --source arrow \
    --data-dir ./wildchat_data \
    --tokenizer-hf Qwen/Qwen2.5-1.5B-Instruct \
    --output data/qwen25_codebook.csv
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