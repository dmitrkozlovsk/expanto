WITH user_aggregation AS (
    {% include "base/user_aggregation.sql" %}
)
{% for experiment_metric in experiment_metrics_list %}
SELECT
      GROUP_NAME as group_name
     , '{{ experiment_metric.alias }}' as metric_name
     , '{{ experiment_metric.type }}' as metric_type
     , '{{ experiment_metric.display_name  }}' as metric_display_name
     , COUNT(distinct user_aggregation.{{ observation.split_id }}) as observation_cnt
     , {{ experiment_metric.sql }} as metric_value
{%- if experiment_metric.type == 'proportion' %}
     , CAST(null as float) as numerator_avg
     , CAST(null as float) as denominator_avg
     , CAST(null as float) as numerator_var
     , CAST(null as float) as denominator_var
     , CAST(null as float) as covariance
{% elif experiment_metric.type == 'avg' %}
     , CAST(null as float) as numerator_avg
     , CAST(null as float) as denominator_avg
     , VAR_SAMP({{ experiment_metric.formula.numerator.alias }}) as numerator_var
     , CAST(null as float) as denominator_var
     , CAST(null as float) as covariance
{% elif experiment_metric.type == 'ratio' %}
     , AVG({{ experiment_metric.formula.numerator.alias }}) as numerator_avg
     , AVG({{ experiment_metric.formula.denominator.alias }}) as denominator_avg
     , VAR_SAMP({{ experiment_metric.formula.numerator.alias }}) as numerator_var
     , VAR_SAMP({{ experiment_metric.formula.denominator.alias }}) as denominator_var
     , COVAR_SAMP({{ experiment_metric.formula.numerator.alias }}, {{ experiment_metric.formula.denominator.alias }}) as covariance
{% endif -%}
FROM user_aggregation
GROUP BY group_name
    {% if not loop.last %}
UNION ALL{% endif %}
{% endfor %}