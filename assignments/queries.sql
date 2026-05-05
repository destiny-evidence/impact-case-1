SELECT assignment_scope_id, count(distinct item_id)
FROM assignment
WHERE annotation_scheme_id = '0689d927-f78d-46aa-bbcf-190ce156f707'
GROUP BY assignment_scope_id;

SELECT count(distinct item_id)
FROM assignment
WHERE annotation_scheme_id = '0689d927-f78d-46aa-bbcf-190ce156f707';
