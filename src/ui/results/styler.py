"""This module provides classes for styling pandas DataFrames in Streamlit.

It includes functionality for creating Streamlit column configurations and for
applying conditional formatting to highlight significant results in a table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from streamlit.column_config import NumberColumn, TextColumn

from src.ui.results._column_configs import ST_TYPE_OF_COLUMNS_CONFIG

if TYPE_CHECKING:
    import pandas as pd  # type: ignore
    from pandas.io.formats.style import Styler  # type: ignore
    from streamlit.elements.lib.column_types import ColumnConfig


class StColumnConfig:
    """Creates st.column_config for a particular metric.

    This class provides a way to generate Streamlit column configurations
    based on a predefined set of column properties.

    Attributes:
        _config: A dictionary containing the configuration for different column types.
    """

    _config = ST_TYPE_OF_COLUMNS_CONFIG

    @classmethod
    def get(cls, column_key: str) -> ColumnConfig | None:
        """Gets the Streamlit column configuration for a given column key.

        Args:
            column_key: The key identifying the column configuration to retrieve.

        Returns:
            A Streamlit ColumnConfig object for the specified column.

        Raises:
            KeyError: If the column_key is not found in the configuration.
            ValueError: If the column type specified in the configuration is unsupported.
        """

        base = cls._config.get(column_key)
        if base is None:
            return None

        column_type = base.get("type")
        kwargs = {
            "label": base.get("label"),
            "help": base.get("help"),
            "width": base.get("width"),
            "pinned": base.get("pinned", False),
        }

        if column_type == "text":
            return TextColumn(**kwargs)

        elif column_type == "number":
            kwargs["format"] = base.get("format")
            return NumberColumn(**kwargs)

        else:
            raise ValueError(f"Unsupported column type: {column_type}")

    @classmethod
    def get_for_df(cls, df: pd.DataFrame) -> dict[str, ColumnConfig]:
        """Gets column configurations for all columns in a DataFrame.

        Args:
            df: The DataFrame for which to get column configurations.

        Returns:
            A dictionary mapping column names to their Streamlit ColumnConfig objects.
        """
        columns = df.columns.unique().tolist()
        return {col: config for col in columns if (config := cls.get(col)) is not None}


class SignificanceTableStyler:
    """Styles a DataFrame to highlight significant results.

    This class applies conditional formatting to a pandas DataFrame to make it
    easier to spot statistically significant results and metric-related columns.

    Attributes:
        p_value_threshold: The threshold below which a p-value is considered significant.
        selected_metric_groups: A list of metric groups to consider for styling.
    """

    def __init__(self, p_value_threshold: float | None):
        """Initializes the SignificanceTableStyler.

        Args:
            p_value_threshold: The significance level for highlighting rows.
            selected_metric_groups: A list of metric groups to be styled.
        """
        self.p_value_threshold = p_value_threshold or 0

    def highlight_significant_rows(self, style_df: Styler) -> Styler:
        """Highlights significant rows in a DataFrame.

        This method applies a background color to rows where the 'p_value' is below
        the specified threshold, making them stand out.

        Args:
            style_df: The Styler object to which the highlighting will be applied.

        Returns:
            The Styler object with the highlighting applied.
        """

        def style_row(row: pd.Series) -> list[str]:
            if row["p_value"] < self.p_value_threshold:
                return ["background-color: rgba(248, 232, 209, 0.6); color: #3C5A75"] * len(row)
            return [""] * len(row)

        if "p_value" in style_df.data.columns:
            return style_df.apply(style_row, axis=1)
        return style_df

    def highlight_metric_columns(self, pandas_df: pd.DataFrame | Styler) -> Styler:
        """Highlights metric columns in a DataFrame.

        This method applies a background color to columns that contain 'metric_value'
        in their name.

        Args:
            pandas_df: The DataFrame or Styler object to style.

        Returns:
            The Styler object with metric columns highlighted.
        """
        # Create a copy of the DataFrame to avoid modifying the original

        metric_columns = [col for col in pandas_df.columns if "metric_value" in col]

        # Apply styling to metric columns
        styled_df = pandas_df.style.set_properties(
            subset=metric_columns,
            **{"background-color": "#DDE4EB"},  # Secondary background for metric columns
        )
        return styled_df

    def apply_styles(self, significance_table: pd.DataFrame) -> Styler:
        """Applies all styling rules to the significance table.

        This method combines highlighting for significant rows and metric columns.

        Args:
            significance_table: The DataFrame to be styled.

        Returns:
            A Styler object with all styles applied.
        """
        styled_table = significance_table.pipe(self.highlight_metric_columns).pipe(
            self.highlight_significant_rows
        )
        return styled_table
