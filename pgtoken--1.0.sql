-- pgtoken--1.0.sql
-- Load via: CREATE EXTENSION pgtoken;

\echo Use "CREATE EXTENSION pgtoken" to load this file. \quit

-- Encode array of token IDs to rank-varint bytea
CREATE FUNCTION pgtoken_encode(
    token_ids   integer[],
    codebook    text DEFAULT 'cl100k_base'
)
RETURNS bytea
AS 'MODULE_PATHNAME', 'pgtoken_encode'
LANGUAGE C STRICT IMMUTABLE;

COMMENT ON FUNCTION pgtoken_encode(integer[], text) IS
'Encode token ID array to rank-varint bytea.
 Codebook CSV must be at $PGDATA/pgtoken_codebooks/<name>.csv';

-- Decode rank-varint bytea back to token ID array
CREATE FUNCTION pgtoken_decode(
    encoded     bytea,
    codebook    text DEFAULT 'cl100k_base'
)
RETURNS integer[]
AS 'MODULE_PATHNAME', 'pgtoken_decode'
LANGUAGE C STRICT IMMUTABLE;

COMMENT ON FUNCTION pgtoken_decode(bytea, text) IS
'Decode rank-varint bytea to integer array of token IDs.';

-- Count tokens without decoding (reads 4-byte header only)
CREATE FUNCTION pgtoken_count(encoded bytea)
RETURNS integer
AS 'MODULE_PATHNAME', 'pgtoken_count'
LANGUAGE C STRICT IMMUTABLE;

COMMENT ON FUNCTION pgtoken_count(bytea) IS
'Return token count from encoded payload. O(1) - reads header only.';

-- Clear in-memory codebook cache
CREATE FUNCTION pgtoken_reload_codebooks()
RETURNS void
AS 'MODULE_PATHNAME', 'pgtoken_reload_codebooks'
LANGUAGE C STRICT VOLATILE;

COMMENT ON FUNCTION pgtoken_reload_codebooks() IS
'Clear codebook cache. Reloads from CSV on next use.';

-- View: list available codebooks
CREATE VIEW pgtoken_codebooks AS
SELECT
    split_part(pg_ls_dir, '.', 1) AS codebook_name,
    pg_ls_dir                     AS filename
FROM pg_ls_dir('pgtoken_codebooks')
WHERE pg_ls_dir LIKE '%.csv';

COMMENT ON VIEW pgtoken_codebooks IS
'Lists codebook CSVs in $PGDATA/pgtoken_codebooks/';
