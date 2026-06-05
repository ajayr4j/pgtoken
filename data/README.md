# data/

This directory contains pre-built pgtoken codebook CSVs.

---

## cl100k_base_codebook.csv

Pre-built frequency codebook for the `cl100k_base` tokenizer (used by GPT-4, GPT-3.5, text-embedding-3-large).

**Built from:** WildChat dataset (allenai/WildChat, 1M+ real LLM conversations)

**Format:**
```
token_id,frequency_rank
1639,0
389,1
...
```

rank 0 = most frequent token in the corpus. All 100,277 cl100k_base tokens are included. Tokens not seen in the corpus are ranked last, sorted by token_id.

**Why WildChat:**
WildChat covers over 1 million real user conversations with ChatGPT across diverse domains -- technical, creative, analytical, conversational. The token frequency distribution closely matches general LLM usage patterns, making it a strong default for most text-heavy RAG and chatbot applications.

For domain-specific deployments, build your own codebook using `scripts/build_codebook.py` with your own corpus. Medical, legal, or financial corpora will produce frequency rankings better matched to their vocabulary, yielding 5-15% better compression than this general-purpose codebook.

---

## Building your own codebook

```bash
# Download WildChat (one-time, ~5GB)
python scripts/download_wildchat.py

# Build codebook for cl100k_base
python scripts/build_codebook.py \
    --source arrow \
    --data-dir ./wildchat_data \
    --tokenizer cl100k_base \
    --output data/cl100k_base_codebook.csv

# Build codebook for Qwen2.5
python scripts/build_codebook.py \
    --source arrow \
    --data-dir ./wildchat_data \
    --tokenizer-hf Qwen/Qwen2.5-1.5B-Instruct \
    --output data/qwen25_codebook.csv

# Install to Postgres
python setup_codebook.py \
    --csv data/cl100k_base_codebook.csv \
    --name cl100k_base
```

---

## Adding more tokenizers

Each tokenizer needs its own codebook file. Name them consistently:

```
data/cl100k_base_codebook.csv    <- GPT-4, text-embedding-3-large
data/o200k_base_codebook.csv     <- GPT-4o
data/qwen25_codebook.csv         <- Qwen2.5 series
data/llama3_codebook.csv         <- Llama 3 series
```

Then reference by name in SQL:
```sql
SELECT pgtoken_encode(token_ids, 'qwen25') FROM chunks;
```