WITH raw_events_cte AS (
        {% include "base/events.sql" %}
    )
    , testids_cte AS (
       {%- if purpose == 'planning' -%}
            {% include "base/dummy_testids.sql" %}
       {%- else -%}
            {% include "base/testids.sql" %}
       {%- endif -%}
    )
SELECT
    t.*,
    r.* EXCLUDE({{observation.split_id}}),
    COALESCE(
        {%- if observation.exposure_event -%}
            exposure.exposure_timestamp, {% endif -%}
        t.first_split_timestamp , null
        ) as exposure_timestamp,
    ROW_NUMBER() OVER (PARTITION BY t.{{observation.split_id}}, event_name order by event_timestamp)
        as event_row_number
FROM testids_cte as t
LEFT JOIN raw_events_cte as r
    ON t.{{observation.split_id}} = r.{{observation.split_id}}
    AND r.event_timestamp >= t.first_split_timestamp
{%- if observation.exposure_event %}
INNER JOIN (
    SELECT
        {{observation.split_id}}
        ,event_timestamp as exposure_timestamp
    FROM raw_events_cte
    WHERE event_name = '{{observation.exposure_event}}'
    QUALIFY ROW_NUMBER() OVER(PARTITION BY {{observation.split_id}} ORDER BY event_timestamp) = 1
) AS exposure
    ON t.{{observation.split_id}} = exposure.{{observation.split_id}}
    AND exposure.exposure_timestamp >= t.first_split_timestamp
{% endif -%}
