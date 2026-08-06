-- ============================================================================
-- Week 2, Day 5 — pgvector Operations Health Check
-- ============================================================================
--
-- Read-only. Run against a real pgvector database:
--     psql -d yourdb -f pgvector_ops_healthcheck.sql
--
-- Nine sections, ordered by how often they are the actual cause of a problem.
-- Section 2 is the one to wire into alerting: the HNSW index's OWN cache-hit
-- ratio is the leading indicator of the Day 3 RAM cliff, and the database-wide
-- cache ratio will look healthy right up until the moment it doesn't.
--
-- Adjust the schema/table filters if your vector tables live outside 'public'.
-- ============================================================================

\echo ''
\echo '=============================================================='
\echo ' 1. VECTOR COLUMNS AND THEIR DIMENSIONS'
\echo '=============================================================='
-- What vector columns exist, what type, and how wide. halfvec here means
-- someone already climbed rung 1 of the Day 3 mitigation ladder.
SELECT c.relname          AS table_name,
       a.attname          AS column_name,
       format_type(a.atttypid, a.atttypmod) AS column_type,
       c.reltuples::bigint AS approx_rows
FROM   pg_attribute a
JOIN   pg_class     c ON c.oid = a.attrelid
JOIN   pg_namespace n ON n.oid = c.relnamespace
JOIN   pg_type      t ON t.oid = a.atttypid
WHERE  t.typname IN ('vector', 'halfvec', 'sparsevec', 'bit')
  AND  a.attnum > 0
  AND  NOT a.attisdropped
  AND  c.relkind = 'r'
  AND  n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER  BY c.relname, a.attnum;


\echo ''
\echo '=============================================================='
\echo ' 2. INDEX CACHE HIT RATIO  <-- THE RAM CLIFF EARLY WARNING'
\echo '=============================================================='
-- Per-index hit ratio. For an HNSW/IVFFlat index this should sit at ~100%.
-- Sustained drift below 99% means graph traversal is going to disk, and
-- because HNSW hops are serially dependent (queue depth 1) the latency
-- penalty is 25-100x, not a gentle slope. ALERT ON THIS.
SELECT s.relname                       AS table_name,
       s.indexrelname                  AS index_name,
       pg_size_pretty(pg_relation_size(s.indexrelid)) AS index_size,
       s.idx_blks_hit                  AS blks_hit,
       s.idx_blks_read                 AS blks_read,
       CASE WHEN s.idx_blks_hit + s.idx_blks_read = 0 THEN NULL
            ELSE round(100.0 * s.idx_blks_hit
                       / (s.idx_blks_hit + s.idx_blks_read), 3)
       END                             AS hit_pct,
       CASE WHEN s.idx_blks_hit + s.idx_blks_read = 0 THEN 'unused'
            WHEN 100.0 * s.idx_blks_hit
                 / (s.idx_blks_hit + s.idx_blks_read) >= 99.9 THEN 'ok'
            WHEN 100.0 * s.idx_blks_hit
                 / (s.idx_blks_hit + s.idx_blks_read) >= 99.0 THEN 'WATCH'
            ELSE 'CLIFF RISK'
       END                             AS verdict
FROM   pg_statio_user_indexes s
JOIN   pg_class c ON c.oid = s.indexrelid
JOIN   pg_am    am ON am.oid = c.relam
WHERE  am.amname IN ('hnsw', 'ivfflat')
ORDER  BY hit_pct NULLS FIRST;


\echo ''
\echo '=============================================================='
\echo ' 3. VECTOR INDEX SIZE vs AVAILABLE MEMORY'
\echo '=============================================================='
-- Compare total ANN index size against shared_buffers. shared_buffers is not
-- the whole story (the OS page cache also holds pages) but if the index alone
-- dwarfs shared_buffers you are relying entirely on the OS cache, and any
-- memory pressure from other work will start evicting graph pages.
SELECT pg_size_pretty(SUM(pg_relation_size(s.indexrelid)))      AS ann_index_total,
       pg_size_pretty(
         (SELECT setting::bigint * 8192 FROM pg_settings
          WHERE name = 'shared_buffers'))                        AS shared_buffers,
       round(
         SUM(pg_relation_size(s.indexrelid))::numeric
         / NULLIF((SELECT setting::bigint * 8192 FROM pg_settings
                   WHERE name = 'shared_buffers'), 0), 2)        AS index_to_buffers_ratio
FROM   pg_statio_user_indexes s
JOIN   pg_class c ON c.oid = s.indexrelid
JOIN   pg_am    am ON am.oid = c.relam
WHERE  am.amname IN ('hnsw', 'ivfflat');


\echo ''
\echo '=============================================================='
\echo ' 4. ANN INDEX DEFINITIONS AND PARAMETERS'
\echo '=============================================================='
-- Confirms m / ef_construction / lists actually in effect. Catches the classic
-- "we thought we rebuilt it with m=32" discrepancy.
SELECT i.schemaname,
       i.tablename,
       i.indexname,
       am.amname AS index_type,
       i.indexdef
FROM   pg_indexes i
JOIN   pg_class c  ON c.relname = i.indexname
JOIN   pg_am    am ON am.oid = c.relam
WHERE  am.amname IN ('hnsw', 'ivfflat')
ORDER  BY i.tablename, i.indexname;


\echo ''
\echo '=============================================================='
\echo ' 5. GRAPH DECAY — DEAD TUPLES AND CHURN'
\echo '=============================================================='
-- Dead tuples on a vector table mean dead nodes still occupying the HNSW
-- graph as traversal waypoints. VACUUM reclaims heap space but does NOT
-- restore graph quality: that needs REINDEX CONCURRENTLY (with 2x index
-- disk headroom). A dead ratio climbing past ~10-15% on a vector table is
-- the signal to schedule one.
SELECT st.relname                     AS table_name,
       st.n_live_tup                  AS live_tuples,
       st.n_dead_tup                  AS dead_tuples,
       CASE WHEN st.n_live_tup + st.n_dead_tup = 0 THEN 0
            ELSE round(100.0 * st.n_dead_tup
                       / (st.n_live_tup + st.n_dead_tup), 2)
       END                            AS dead_pct,
       st.n_tup_upd                   AS updates_total,
       st.n_tup_del                   AS deletes_total,
       st.last_vacuum,
       st.last_autovacuum,
       CASE WHEN st.n_live_tup + st.n_dead_tup = 0 THEN 'empty'
            WHEN 100.0 * st.n_dead_tup
                 / (st.n_live_tup + st.n_dead_tup) > 15 THEN 'REINDEX SOON'
            WHEN 100.0 * st.n_dead_tup
                 / (st.n_live_tup + st.n_dead_tup) > 5  THEN 'watch'
            ELSE 'ok'
       END                            AS verdict
FROM   pg_stat_user_tables st
WHERE  st.relid IN (
         SELECT DISTINCT c.oid
         FROM   pg_attribute a
         JOIN   pg_class c ON c.oid = a.attrelid
         JOIN   pg_type  t ON t.oid = a.atttypid
         WHERE  t.typname IN ('vector', 'halfvec', 'sparsevec')
           AND  a.attnum > 0 AND NOT a.attisdropped)
ORDER  BY dead_pct DESC;


\echo ''
\echo '=============================================================='
\echo ' 6. IS THE ANN INDEX ACTUALLY BEING USED?'
\echo '=============================================================='
-- An unused vector index usually means queries are falling back to a seq scan
-- because of a filter, an ORDER BY that does not match the index opclass, or a
-- distance operator mismatch (<=> vs <-> vs <#>). Zero scans on a large HNSW
-- index is a silent, expensive bug.
SELECT s.relname       AS table_name,
       s.indexrelname  AS index_name,
       s.idx_scan      AS index_scans,
       s.idx_tup_read  AS tuples_read,
       pg_size_pretty(pg_relation_size(s.indexrelid)) AS index_size,
       CASE WHEN s.idx_scan = 0 THEN 'NEVER USED — investigate'
            ELSE 'in use'
       END             AS verdict
FROM   pg_stat_user_indexes s
JOIN   pg_class c  ON c.oid = s.indexrelid
JOIN   pg_am    am ON am.oid = c.relam
WHERE  am.amname IN ('hnsw', 'ivfflat')
ORDER  BY s.idx_scan;


\echo ''
\echo '=============================================================='
\echo ' 7. SETTINGS THAT MATTER FOR VECTOR WORKLOADS'
\echo '=============================================================='
-- maintenance_work_mem is the big one: too small and an HNSW build spills to
-- disk and takes roughly 5x longer. Set it per-session for builds rather than
-- globally. hnsw.ef_search / ivfflat.probes are the recall/latency dial —
-- if someone lowered these to fix a latency alert, recall dropped and nothing
-- told you (see recall_monitor.py).
SELECT name, setting, unit, source, short_desc
FROM   pg_settings
WHERE  name IN ('shared_buffers',
                'work_mem',
                'maintenance_work_mem',
                'max_parallel_maintenance_workers',
                'max_parallel_workers_per_gather',
                'effective_cache_size',
                'hnsw.ef_search',
                'hnsw.iterative_scan',
                'ivfflat.probes',
                'ivfflat.iterative_scan')
ORDER  BY name;


\echo ''
\echo '=============================================================='
\echo ' 8. INSTALLED EXTENSION VERSIONS'
\echo '=============================================================='
-- pgvector 0.7 brought halfvec/sparsevec/binary quantization; 0.8 brought
-- iterative index scans, which materially improve filtered ANN queries.
-- If default_version is ahead of installed_version, an ALTER EXTENSION
-- vector UPDATE is available (test in staging — some upgrades want a reindex).
SELECT e.extname,
       e.extversion AS installed_version,
       av.default_version,
       CASE WHEN av.default_version IS DISTINCT FROM e.extversion
            THEN 'UPGRADE AVAILABLE'
            ELSE 'current'
       END AS status
FROM   pg_extension e
LEFT   JOIN pg_available_extensions av ON av.name = e.extname
WHERE  e.extname IN ('vector', 'vectorscale', 'pg_trgm', 'pg_search', 'pg_stat_statements')
ORDER  BY e.extname;


\echo ''
\echo '=============================================================='
\echo ' 9. LONG-RUNNING QUERIES AND CONNECTION PRESSURE'
\echo '=============================================================='
-- Little'\''s Law from Day 3: concurrent = QPS x latency. If the RAM cliff is
-- reached, latency jumps ~40x and so does the connection count, which turns a
-- latency problem into pool exhaustion. This shows whether that is happening.
SELECT count(*) FILTER (WHERE state = 'active')            AS active,
       count(*) FILTER (WHERE state = 'idle')              AS idle,
       count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_txn,
       count(*)                                            AS total,
       (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') AS max_connections,
       round(100.0 * count(*)
             / (SELECT setting::int FROM pg_settings WHERE name = 'max_connections'), 1)
                                                           AS pct_of_max
FROM   pg_stat_activity
WHERE  backend_type = 'client backend';

\echo ''
\echo '--- queries running longer than 5s ---'
SELECT pid,
       now() - query_start AS duration,
       state,
       left(regexp_replace(query, '\s+', ' ', 'g'), 100) AS query_snippet
FROM   pg_stat_activity
WHERE  state = 'active'
  AND  backend_type = 'client backend'
  AND  now() - query_start > interval '5 seconds'
ORDER  BY duration DESC;

\echo ''
\echo '=============================================================='
\echo ' REMINDER: none of the above measures RECALL.'
\echo ' Every check here can pass while retrieval quality decays.'
\echo ' Run recall_monitor.py nightly against a read replica.'
\echo '=============================================================='
\echo ''
