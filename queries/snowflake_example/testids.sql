SELECT
    {{observation.split_id}},
    record_timestamp as first_split_timestamp,
    group_name,
    country,
    city,
    os,
    version,
    galaxy --default:milky way
FROM testids
WHERE 1=1
    AND experiment_name = '{{observation.db_experiment_name}}'
QUALIFY ROW_NUMBER() OVER (PARTITION BY {{observation.split_id}} ORDER BY record_timestamp) = 1
