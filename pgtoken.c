/*
 * pgtoken.c
 * ---------
 * PostgreSQL extension for rank-varint token storage.
 *
 * Fixes in this version:
 *   1. Empty array: allow ndim=0 or n=0
 *   2. Missing codebook entry: error instead of silent corruption
 *   3. Element type check: reject int8[] with clear message
 *   4. TopMemoryContext for codebook arrays (memory stability fix)
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

PG_MODULE_MAGIC;

#define MAX_VOCAB         151936
#define MAX_CODEBOOKS     8
#define CODEBOOK_NAME_LEN 64

/* ----------------------------------------------------------------
 * Codebook
 * ---------------------------------------------------------------- */
typedef struct {
    char   name[CODEBOOK_NAME_LEN];
    int    vocab_size;
    int32 *token_to_rank;   /* index=token_id → rank,     -1=missing */
    int32 *rank_to_token;   /* index=rank     → token_id, -1=missing */
} Codebook;

static Codebook codebook_cache[MAX_CODEBOOKS];
static int      codebook_count = 0;

/* ----------------------------------------------------------------
 * Varint
 * ---------------------------------------------------------------- */
static int
varint_encode_rank(int32 rank, uint8 *out)
{
    if (rank < 128) {
        out[0] = (uint8)(rank & 0x7F);
        return 1;
    } else if (rank < 16512) {
        int32 r = rank - 128;
        out[0] = (uint8)(0x80 | ((r >> 8) & 0x3F));
        out[1] = (uint8)(r & 0xFF);
        return 2;
    } else {
        int32 r = rank - 16512;
        out[0] = (uint8)(0xC0 | ((r >> 16) & 0x3F));
        out[1] = (uint8)((r >> 8) & 0xFF);
        out[2] = (uint8)(r & 0xFF);
        return 3;
    }
}

static int
varint_decode_rank(const uint8 *in, int available, int32 *out_rank)
{
    uint8 b0;

    if (available < 1)
        ereport(ERROR,
                (errcode(ERRCODE_DATA_CORRUPTED),
                 errmsg("pgtoken: truncated varint")));

    b0 = in[0];

    if ((b0 & 0x80) == 0) {
        *out_rank = (int32)(b0 & 0x7F);
        return 1;
    } else if ((b0 & 0xC0) == 0x80) {
        if (available < 2)
            ereport(ERROR,
                    (errcode(ERRCODE_DATA_CORRUPTED),
                     errmsg("pgtoken: truncated 2-byte varint")));
        *out_rank = (int32)((b0 & 0x3F) << 8 | in[1]) + 128;
        return 2;
    } else {
        if (available < 3)
            ereport(ERROR,
                    (errcode(ERRCODE_DATA_CORRUPTED),
                     errmsg("pgtoken: truncated 3-byte varint")));
        *out_rank = (int32)((b0 & 0x3F) << 16 | (int32)in[1] << 8 | in[2]) + 16512;
        return 3;
    }
}

/* ----------------------------------------------------------------
 * Codebook loading
 * ---------------------------------------------------------------- */
static Codebook *
find_codebook(const char *name)
{
    int i;
    for (i = 0; i < codebook_count; i++)
        if (strcmp(codebook_cache[i].name, name) == 0)
            return &codebook_cache[i];
    return NULL;
}

static Codebook *
load_codebook(const char *name)
{
    Codebook *cb;
    char      path[MAXPGPATH];
    FILE     *f;
    char      line[64];
    int       token_id, rank;
    int       row_count = 0;

    cb = find_codebook(name);
    if (cb)
        return cb;

    if (codebook_count >= MAX_CODEBOOKS)
        ereport(ERROR,
                (errmsg("pgtoken: codebook cache full (max %d)", MAX_CODEBOOKS)));

    snprintf(path, sizeof(path), "%s/pgtoken_codebooks/%s.csv", DataDir, name);

    f = fopen(path, "r");
    if (!f)
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_FILE),
                 errmsg("pgtoken: cannot open codebook \"%s\"", name),
                 errhint("Expected: %s", path)));

    cb = &codebook_cache[codebook_count];
    strncpy(cb->name, name, CODEBOOK_NAME_LEN - 1);
    cb->name[CODEBOOK_NAME_LEN - 1] = '\0';

    /*
     * FIX 4: allocate in TopMemoryContext so arrays survive
     * across transactions. palloc uses per-transaction context
     * which gets freed — leaving the static struct with dangling
     * pointers and corrupting the heap on the next palloc call.
     */
    cb->token_to_rank = (int32 *) MemoryContextAlloc(TopMemoryContext,
                                      MAX_VOCAB * sizeof(int32));
    cb->rank_to_token = (int32 *) MemoryContextAlloc(TopMemoryContext,
                                      MAX_VOCAB * sizeof(int32));
    memset(cb->token_to_rank, -1, MAX_VOCAB * sizeof(int32));
    memset(cb->rank_to_token, -1, MAX_VOCAB * sizeof(int32));

    /* skip header */
    if (!fgets(line, sizeof(line), f))
        ereport(ERROR, (errmsg("pgtoken: codebook \"%s\" is empty", name)));

    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%d,%d", &rank, &token_id) == 2) {
            if (token_id < 0 || token_id >= MAX_VOCAB) continue;
            if (rank < 0 || rank >= MAX_VOCAB) continue;
            cb->token_to_rank[token_id] = rank;
            cb->rank_to_token[rank]     = token_id;
            row_count++;
        }
    }
    fclose(f);

    cb->vocab_size = row_count;
    codebook_count++;

    elog(LOG, "pgtoken: loaded \"%s\" — %d tokens", name, row_count);
    return cb;
}

/* ----------------------------------------------------------------
 * pgtoken_encode
 * ---------------------------------------------------------------- */
PG_FUNCTION_INFO_V1(pgtoken_encode);

Datum
pgtoken_encode(PG_FUNCTION_ARGS)
{
    ArrayType  *arr    = PG_GETARG_ARRAYTYPE_P(0);
    text       *cbn_t  = PG_GETARG_TEXT_PP(1);
    char       *cbname = text_to_cstring(cbn_t);
    Codebook   *cb;
    int         n;
    int32      *token_ids;
    uint8      *buf;
    int         pos = 0;
    int         i;
    bytea      *result;

    /*
     * FIX 3: check element type before touching data.
     * int8[] looks like int4[] to ARR_DATA_PTR but elements
     * are 8 bytes wide — reads garbage silently.
     */
    if (ARR_ELEMTYPE(arr) != INT4OID)
        ereport(ERROR,
                (errmsg("pgtoken_encode: token_ids must be integer[] (int4), "
                        "got OID %u — cast your array: ARRAY[...]::integer[]",
                        ARR_ELEMTYPE(arr))));

    /*
     * FIX 1: allow empty arrays.
     * ARRAY[]::integer[] has ndim=1 but n=0.
     * The cast form works fine — just handle n=0 cleanly.
     * ndim=0 should not happen with a proper cast but guard anyway.
     */
    if (ARR_NDIM(arr) == 0 || ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr)) == 0) {
        /* encode as 4-byte header with count=0, no payload */
        result = (bytea *) palloc(VARHDRSZ + 4);
        SET_VARSIZE(result, VARHDRSZ + 4);
        memset(VARDATA(result), 0, 4);
        PG_RETURN_BYTEA_P(result);
    }

    if (ARR_NDIM(arr) != 1)
        ereport(ERROR, (errmsg("pgtoken_encode: expected 1-D integer array")));
    if (ARR_HASNULL(arr))
        ereport(ERROR, (errmsg("pgtoken_encode: array must not contain NULLs")));

    cb        = load_codebook(cbname);
    n         = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
    token_ids = (int32 *) ARR_DATA_PTR(arr);
    buf       = (uint8 *) palloc(n * 3 + 4);

    /* 4-byte LE count header */
    buf[0] = (uint8)(n & 0xFF);
    buf[1] = (uint8)((n >> 8) & 0xFF);
    buf[2] = (uint8)((n >> 16) & 0xFF);
    buf[3] = (uint8)((n >> 24) & 0xFF);
    pos = 4;

    for (i = 0; i < n; i++) {
        int32 tid = token_ids[i];
        int32 rnk;

        if (tid < 0 || tid >= MAX_VOCAB)
            ereport(ERROR,
                    (errmsg("pgtoken_encode: token_id %d out of range at index %d "
                            "(valid range: 0-%d)", tid, i, MAX_VOCAB - 1)));

        rnk = cb->token_to_rank[tid];

        /*
         * FIX 2: token not in codebook → hard error, not silent corruption.
         * Previously: rnk=-1 would encode as a valid varint, decode to
         * wrong token. Now: explicit error with actionable message.
         */
        if (rnk < 0)
            ereport(ERROR,
                    (errmsg("pgtoken_encode: token_id %d is not in codebook \"%s\"",
                            tid, cbname),
                     errhint("Your codebook CSV may be missing this token. "
                             "Check: SELECT COUNT(*) FROM ... WHERE token_id = %d",
                             tid)));

        pos += varint_encode_rank(rnk, buf + pos);
    }

    result = (bytea *) palloc(VARHDRSZ + pos);
    SET_VARSIZE(result, VARHDRSZ + pos);
    memcpy(VARDATA(result), buf, pos);
    pfree(buf);

    PG_RETURN_BYTEA_P(result);
}

/* ----------------------------------------------------------------
 * pgtoken_decode
 * ---------------------------------------------------------------- */
PG_FUNCTION_INFO_V1(pgtoken_decode);

Datum
pgtoken_decode(PG_FUNCTION_ARGS)
{
    bytea       *encoded  = PG_GETARG_BYTEA_PP(0);
    text        *cbn_t    = PG_GETARG_TEXT_PP(1);
    char        *cbname   = text_to_cstring(cbn_t);
    Codebook    *cb       = load_codebook(cbname);
    const uint8 *data     = (const uint8 *) VARDATA_ANY(encoded);
    int          data_len = VARSIZE_ANY_EXHDR(encoded);
    int          pos      = 0;
    int32        n;
    Datum       *elems;
    bool        *nulls;
    ArrayType   *result;
    int          dims[1], lbs[1];
    int          i;

    if (data_len < 4)
        ereport(ERROR,
                (errmsg("pgtoken_decode: payload too short (%d bytes)", data_len)));

    n = (int32)(  (uint32)data[0]
                | (uint32)data[1] << 8
                | (uint32)data[2] << 16
                | (uint32)data[3] << 24);
    pos = 4;

    if (n < 0 || n > 1000000)
        ereport(ERROR,
                (errmsg("pgtoken_decode: implausible token count %d", n)));

    /* handle empty payload cleanly */
    if (n == 0) {
        dims[0] = 0;
        lbs[0]  = 1;
        result  = construct_md_array(NULL, NULL, 1, dims, lbs,
                                     INT4OID, sizeof(int32), true, TYPALIGN_INT);
        PG_RETURN_ARRAYTYPE_P(result);
    }

    elems = (Datum *) palloc(n * sizeof(Datum));
    nulls = (bool  *) palloc0(n * sizeof(bool));

    for (i = 0; i < n; i++) {
        int32 rnk;
        int32 tid;
        int   consumed = varint_decode_rank(data + pos, data_len - pos, &rnk);
        pos += consumed;

        if (rnk < 0 || rnk >= MAX_VOCAB)
            ereport(ERROR,
                    (errmsg("pgtoken_decode: rank %d out of range at token %d",
                            rnk, i)));

        tid = cb->rank_to_token[rnk];
        if (tid < 0)
            ereport(ERROR,
                    (errmsg("pgtoken_decode: rank %d has no token in codebook \"%s\" "
                            "at position %d", rnk, cbname, i)));

        elems[i] = Int32GetDatum(tid);
        nulls[i] = false;
    }

    dims[0] = n;
    lbs[0]  = 1;
    result  = construct_md_array(elems, nulls, 1, dims, lbs,
                                 INT4OID, sizeof(int32), true, TYPALIGN_INT);
    pfree(elems);
    pfree(nulls);

    PG_RETURN_ARRAYTYPE_P(result);
}

/* ----------------------------------------------------------------
 * pgtoken_count  (O(1) - header only)
 * ---------------------------------------------------------------- */
PG_FUNCTION_INFO_V1(pgtoken_count);

Datum
pgtoken_count(PG_FUNCTION_ARGS)
{
    bytea       *encoded  = PG_GETARG_BYTEA_PP(0);
    const uint8 *data     = (const uint8 *) VARDATA_ANY(encoded);
    int          data_len = VARSIZE_ANY_EXHDR(encoded);
    int32        n;

    if (data_len < 4)
        ereport(ERROR,
                (errmsg("pgtoken_count: payload too short (%d bytes)", data_len)));

    n = (int32)(  (uint32)data[0]
                | (uint32)data[1] << 8
                | (uint32)data[2] << 16
                | (uint32)data[3] << 24);

    PG_RETURN_INT32(n);
}

/* ----------------------------------------------------------------
 * pgtoken_reload_codebooks
 * ---------------------------------------------------------------- */
PG_FUNCTION_INFO_V1(pgtoken_reload_codebooks);

Datum
pgtoken_reload_codebooks(PG_FUNCTION_ARGS)
{
    codebook_count = 0;
    elog(NOTICE, "pgtoken: codebook cache cleared");
    PG_RETURN_VOID();
}