PGFILEDESC = "pgtoken - rank-varint token storage for LLM workloads"
EXTENSION  = pgtoken
DATA       = pgtoken--1.0.sql
MODULES    = pgtoken

PG_CONFIG  = pg_config
PGXS      := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

REGRESS    = pgtoken_test
