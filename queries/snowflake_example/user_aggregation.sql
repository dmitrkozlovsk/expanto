
WITH joined_events_testids AS (
    {% include "snowflake_example/events_x_testids.sql" %}
)
SELECT
   j.{{ observation.split_id }}
 , j.GROUP_NAME
--calculate metrics for users--
{% for user_formula in user_formula_list -%}
    , {{ user_formula.sql}} AS {{ user_formula.alias }}
{% endfor -%}
FROM joined_events_testids as j
--if we have segments we intersect them
{%- if observation.audience_tables %} {% for audience_table in observation.audience_tables %}
INNER JOIN {{ audience_table }} as audience_{{ loop.index }}
    ON j.{{ observation.split_id }} = audience_{{loop.index}}.{{ observation.split_id }}
        {%- endfor -%}{% endif %}
WHERE 1=1
    AND exposure_timestamp BETWEEN '{{ observation.exposure_start_datetime }}'
        AND '{{ observation.exposure_end_datetime }}'
    AND event_timestamp BETWEEN '{{ observation.calc_start_datetime }}'
        AND '{{ observation.calc_end_datetime }}'
{%- if observation.filters -%}
{%- for filter in observation.filters %}
    AND {{ filter }} {% endfor %}{% endif %}
GROUP BY
    j.{{ observation.split_id }},
    GROUP_NAME