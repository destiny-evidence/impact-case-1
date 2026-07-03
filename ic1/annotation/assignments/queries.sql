-- Scratch queries for inspecting assignment coverage.
-- The scheme id is INOUT_SCHEME_ID in ic1/core/ids.py (single source of truth in Python).
-- Run with e.g.:  psql ... -v scheme=0689d927-f78d-46aa-bbcf-190ce156f707 -f queries.sql

SELECT assignment_scope_id, count(distinct item_id)
FROM assignment
WHERE annotation_scheme_id = :'scheme'
GROUP BY assignment_scope_id;

SELECT count(distinct item_id)
FROM assignment
WHERE annotation_scheme_id = :'scheme';
