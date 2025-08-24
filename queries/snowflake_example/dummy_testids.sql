SELECT
    {{observation.split_id}},
    record_timestamp as first_split_timestamp,
    'planning' as group_name,
    country,
    city,
    os,
    version,
    galaxy --default:milky way
FROM testids
WHERE 1=1
  AND record_timestamp BETWEEN '{{observation.exposure_start_datetime}}'
  AND '{{observation.calc_end_datetime}}'
QUALIFY ROW_NUMBER() OVER (PARTITION BY {{observation.split_id}} ORDER BY record_timestamp) = 1