from src.domain.enums import CalculationPurpose


def test_renderer(query_renderer, mock_observation):
    """Test rendering base calculation query for planning purpose"""
    query = query_renderer.render(obs=mock_observation, purpose=CalculationPurpose.PLANNING)
    # TODO: fix mock observation and fix fake template according to custom_test_ids_query
    assert isinstance(query, str)
    assert "user_id as split_id" in query
    # assert "test_experiment_db_name" not in query
    assert "planning" in query
    assert "AND (platform='web')" in query
    assert "active_users" in query


def test_renderer_with_metric_names(query_renderer, mock_observation):
    """Test rendering base calculation query with specific metric names"""
    query = query_renderer.render(
        obs=mock_observation,
        purpose=CalculationPurpose.PLANNING,
        experiment_metric_names=["click_through_rate"],
    )

    assert isinstance(query, str)
    assert "click_through_rate" in query
    for exp_alias in ["avg_session_duration", "users_who_click_product_ratio", "avg_order_value"]:
        assert exp_alias not in query
    for user_alias in ["product_purchase_cnt", "session_duration"]:
        assert user_alias not in query
