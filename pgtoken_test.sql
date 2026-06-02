-- pgtoken regression tests
-- Run: make installcheck

CREATE EXTENSION pgtoken version '1.0';

-- 1. Extension loaded
SELECT extname, extversion FROM pg_extension WHERE extname = 'pgtoken';

-- 2. Roundtrip
SELECT pgtoken_decode(
    pgtoken_encode(ARRAY[1639, 389, 257, 6290, 7234, 13], 'cl100k_base'),
    'cl100k_base'
) = ARRAY[1639, 389, 257, 6290, 7234, 13] AS roundtrip_ok;

-- 3. Count without decoding
SELECT pgtoken_count(
    pgtoken_encode(ARRAY[1639, 389, 257, 6290, 7234, 13], 'cl100k_base')
) = 6 AS count_ok;

-- 4. Empty array
SELECT pgtoken_count(
    pgtoken_encode(ARRAY[]::integer[], 'cl100k_base')
) = 0 AS empty_ok;

-- 5. Single token
SELECT pgtoken_decode(
    pgtoken_encode(ARRAY[0], 'cl100k_base'),
    'cl100k_base'
) = ARRAY[0] AS single_ok;

-- 6. List codebooks
SELECT codebook_name FROM pgtoken_codebooks;

-- 7. Table usage demo
CREATE TEMP TABLE demo (
    id        serial PRIMARY KEY,
    content   text,
    token_ids bytea
);

INSERT INTO demo (content, token_ids) VALUES (
    'My name is Ajay and the database is the source of truth.',
    pgtoken_encode(
        ARRAY[5159,836,374,362,438,279,4729,374,279,2592,315,8206,13],
        'cl100k_base'
    )
);

SELECT content, pgtoken_count(token_ids) AS tokens FROM demo;

-- 8. Context window filter
SELECT id, content, pgtoken_count(token_ids) AS tokens
FROM demo
WHERE pgtoken_count(token_ids) <= 1024;

-- pgtoken comprehensive test suite
-- Run: sudo -u postgres psql -p 5434 -f pgtoken_test.sql

\echo '=== pgtoken test suite ==='

-- ----------------------------------------------------------------
-- 1. Basic roundtrip
-- ----------------------------------------------------------------
\echo '--- 1. Basic roundtrip'

SELECT pgtoken_decode(
    pgtoken_encode(ARRAY[1639,389,257], 'cl100k_base'),
    'cl100k_base'
) = ARRAY[1639,389,257] AS roundtrip_3_tokens;

-- ----------------------------------------------------------------
-- 2. Count matches array length
-- ----------------------------------------------------------------
\echo '--- 2. Count correctness'

SELECT pgtoken_count(
    pgtoken_encode(ARRAY[1639,389,257], 'cl100k_base')
) = 3 AS count_3;

SELECT pgtoken_count(
    pgtoken_encode(ARRAY[1,2,3,4,5,6,7,8,9,10], 'cl100k_base')
) = 10 AS count_10;

-- ----------------------------------------------------------------
-- 3. Empty array
-- ----------------------------------------------------------------
\echo '--- 3. Empty array'

SELECT pgtoken_count(
    pgtoken_encode(ARRAY[]::integer[], 'cl100k_base')
) = 0 AS empty_count;

SELECT pgtoken_decode(
    pgtoken_encode(ARRAY[]::integer[], 'cl100k_base'),
    'cl100k_base'
) = ARRAY[]::integer[] AS empty_roundtrip;

-- ----------------------------------------------------------------
-- 4. Single token
-- ----------------------------------------------------------------
\echo '--- 4. Single token'

SELECT pgtoken_decode(
    pgtoken_encode(ARRAY[0], 'cl100k_base'),
    'cl100k_base'
) = ARRAY[0] AS single_token_0;

-- ----------------------------------------------------------------
-- 5. Varint boundary tokens
-- rank 0-127   = 1 byte  (most common tokens)
-- rank 128-16511 = 2 bytes
-- rank 16512+  = 3 bytes
-- We need tokens that map to ranks near those boundaries
-- ----------------------------------------------------------------
\echo '--- 5. Varint boundary coverage'

-- encode a range and verify count stays correct
SELECT pgtoken_count(
    pgtoken_encode(ARRAY[1,100,500,1000,5000,10000,50000,90000], 'cl100k_base')
) = 8 AS boundary_count;

SELECT pgtoken_decode(
    pgtoken_encode(ARRAY[1,100,500,1000,5000,10000,50000,90000], 'cl100k_base'),
    'cl100k_base'
) = ARRAY[1,100,500,1000,5000,10000,50000,90000] AS boundary_roundtrip;

-- ----------------------------------------------------------------
-- 6. Longer sequence (simulate real chunk)
-- ----------------------------------------------------------------
\echo '--- 6. Real-length chunk (50 tokens)'

WITH chunk AS (
    SELECT ARRAY(
        SELECT (random() * 90000)::int
        FROM generate_series(1,50)
    ) AS ids
)
SELECT
    pgtoken_count(pgtoken_encode(ids, 'cl100k_base')) = 50 AS length_preserved,
    pgtoken_decode(pgtoken_encode(ids, 'cl100k_base'), 'cl100k_base') = ids AS roundtrip_ok
FROM chunk;

-- ----------------------------------------------------------------
-- 7. Compression ratio check
-- token IDs as raw int4 = 4 bytes each
-- rank-varint should be smaller
-- ----------------------------------------------------------------
\echo '--- 7. Compression ratio'

WITH test AS (
    SELECT ARRAY[1639,389,257,6290,7234,13,318,257,4950,7234] AS ids
),
encoded AS (
    SELECT
        ids,
        pgtoken_encode(ids, 'cl100k_base') AS enc
    FROM test
)
SELECT
    array_length(ids, 1)                        AS token_count,
    array_length(ids, 1) * 4                    AS raw_uint32_bytes,
    length(enc)                                 AS varint_bytes,
    round(length(enc)::numeric /
          (array_length(ids, 1) * 4) * 100, 1) AS pct_of_raw,
    round(length(enc)::numeric /
          array_length(ids, 1), 2)              AS bytes_per_token
FROM encoded;

-- ----------------------------------------------------------------
-- 8. Multiple encode/decode calls (tests memory context stability)
-- This is what killed the first version — palloc in wrong context
-- ----------------------------------------------------------------
\echo '--- 8. Memory stability (10 sequential calls)'

SELECT
    i,
    pgtoken_decode(
        pgtoken_encode(ARRAY[1639, 389, i::int4], 'cl100k_base'),
        'cl100k_base'
    ) = ARRAY[1639, 389, i::int4] AS ok
FROM generate_series(1, 10) AS i;

-- ----------------------------------------------------------------
-- 9. Reload codebooks and re-query (tests reload + re-load path)
-- ----------------------------------------------------------------
\echo '--- 9. Reload codebooks'

SELECT pgtoken_reload_codebooks();

SELECT pgtoken_decode(
    pgtoken_encode(ARRAY[1639,389,257], 'cl100k_base'),
    'cl100k_base'
) = ARRAY[1639,389,257] AS works_after_reload;

-- ----------------------------------------------------------------
-- 10. Table storage pattern
-- ----------------------------------------------------------------
\echo '--- 10. Table storage and retrieval'

CREATE TEMP TABLE token_chunks (
    id          serial PRIMARY KEY,
    content     text,
    token_ids   bytea
);

INSERT INTO token_chunks (content, token_ids) VALUES
('HDFC Q3 results show 18 percent profit growth',
 pgtoken_encode(ARRAY[39308,1229,2632,1501,220,972,4,7194], 'cl100k_base')),
('RBI policy impact on lending rates',
 pgtoken_encode(ARRAY[49738,4947,5536,389,43679,7969], 'cl100k_base')),
('Nifty 50 hits all time high today',
 pgtoken_encode(ARRAY[45, 42, 19,  8, 892, 21, 93, 47], 'cl100k_base'));

-- count without decoding
SELECT id, content, pgtoken_count(token_ids) AS tokens
FROM token_chunks;

-- context window filter
SELECT id, content
FROM token_chunks
WHERE pgtoken_count(token_ids) <= 10
ORDER BY id;

-- full roundtrip from table
SELECT
    id,
    pgtoken_decode(token_ids, 'cl100k_base') AS recovered_ids
FROM token_chunks;

-- ----------------------------------------------------------------
-- 11. pgtoken_codebooks view
-- ----------------------------------------------------------------
\echo '--- 11. Codebooks view'

SELECT * FROM pgtoken_codebooks;

\echo '=== all tests done ==='