"""Utility functions and classes for the Expanto application."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from pydantic import BaseModel


class DatetimeUtils:
    """Utility functions for datetime operations."""

    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC datetime.

        Returns:
            Current datetime in UTC timezone.
        """
        return datetime.now(UTC)

    @staticmethod
    def utc_today() -> datetime:
        """Get today's date at midnight in UTC.

        Returns:
            Today's date at 00:00:00 in UTC timezone.
        """
        utc = UTC
        today = datetime.now(utc).date()
        return datetime.combine(today, datetime.min.time(), tzinfo=utc)


class ValidationUtils:
    """Utility functions for data validation and security checks."""

    @staticmethod
    def validate_sql_query(sql: str | None) -> None:
        """Validate SQL query to ensure it's a safe read-only query.

        This method checks that the SQL query:
        - Is a read-only operation (SELECT, WITH, VALUES)
        - Doesn't contain malicious or modifying statements
        - Supports complex constructions like JOINs, CTEs, subqueries

        Args:
            sql: The SQL string to validate.

        Raises:
            ValueError: If the SQL query is invalid or potentially unsafe.
        """
        if sql is None or not sql.strip():
            return  # Allow empty queries

        sql = sql.strip()

        # Check for multiple statements (basic protection)
        if ";" in sql[:-1]:  # Allow single trailing semicolon
            raise ValueError("Multiple SQL statements are not allowed")

        # Remove trailing semicolon if present
        if sql.endswith(";"):
            sql = sql[:-1].rstrip()

        # Remove comments to clean up the SQL
        # Multi-line comments /* ... */
        sql_clean = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
        # Single-line comments -- ...
        sql_clean = re.sub(r"--[^\n\r]*", " ", sql_clean)
        # MySQL-style comments #
        sql_clean = re.sub(r"#[^\n\r]*", " ", sql_clean)

        # Normalize whitespace
        sql_clean = re.sub(r"\s+", " ", sql_clean).strip()

        if not sql_clean:
            return  # Allow queries that are only comments

        # Check that query starts with allowed keywords
        first_word_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)", sql_clean)
        if not first_word_match:
            raise ValueError("Cannot determine SQL statement type")

        first_keyword = first_word_match.group(1).upper()
        if first_keyword not in ("SELECT", "WITH", "VALUES"):
            raise ValueError(
                f"Only SELECT, WITH, and VALUES statements are allowed. Found: {first_keyword}"
            )

        # Check for forbidden keywords that could modify data or schema
        forbidden_keywords = [
            "ALTER",
            "CREATE",
            "DROP",
            "TRUNCATE",
            "RENAME",
            "INSERT",
            "UPDATE",
            "DELETE",
            "MERGE",
            "REPLACE",
            "UPSERT",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "SAVEPOINT",
            "GRANT",
            "REVOKE",
            "COPY",
            "LOAD",
            "IMPORT",
            "EXPORT",
            "CALL",
            "EXEC",
            "EXECUTE",
            "DO",
            "VACUUM",
            "ANALYZE",
            "RESET",
            "LOCK",
            "UNLOCK",
        ]

        forbidden_pattern = re.compile(r"(?i)\b(" + "|".join(forbidden_keywords) + r")\b")
        forbidden_matches = forbidden_pattern.findall(sql_clean)
        if forbidden_matches:
            unique_matches = set(k.upper() for k in forbidden_matches)
            raise ValueError(f"Forbidden SQL keywords detected: {', '.join(unique_matches)}")

        # Prevent SELECT ... INTO (table creation)
        if re.search(r"(?i)\bSELECT\b[^;]*?\bINTO\b", sql_clean):
            raise ValueError("SELECT ... INTO statements are not allowed")

        # Prevent row locking clauses
        if re.search(r"(?i)\bFOR\s+(UPDATE|SHARE|KEY\s+SHARE|NO\s+KEY\s+UPDATE)\b", sql_clean):
            raise ValueError("Row locking clauses (FOR UPDATE/SHARE) are not allowed")

        # Prevent timing/delay functions that could be used for attacks
        if re.search(r"(?i)\b(pg_sleep|sleep|benchmark)\s*\(", sql_clean):
            raise ValueError("Timing/delay functions are not allowed")

        # Check for modifying CTEs
        if re.search(r"(?i)\bWITH\b", sql_clean):
            # Look for CTE patterns: WITH name AS ( ... )
            cte_pattern = re.compile(
                r"(?i)\bWITH\b\s+([a-zA-Z_][\w]*)\s+AS\s*\((.*?)\)(?=\s*(,|SELECT|VALUES|WITH|$))",
                flags=re.DOTALL,
            )
            for match in cte_pattern.finditer(sql_clean):
                cte_body = match.group(2).strip()
                first_cte_word = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)", cte_body)
                if first_cte_word:
                    cte_keyword = first_cte_word.group(1).upper()
                    if cte_keyword in ("INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE", "UPSERT"):
                        raise ValueError(
                            f"Modifying CTE detected: {cte_keyword} is not allowed in WITH clauses"
                        )

    @staticmethod
    def check_all_args_is_digit_or_ndarray(args: list[float | int | np.ndarray]) -> None:
        """Check if all arguments are of the same type.

        Args:
            args: List of arguments to check.

        Raises:
            TypeError: If all arguments are not of the same type.
        """
        is_digit = [isinstance(arg, float | int) for arg in args]
        is_ndarray = [isinstance(arg, np.ndarray) for arg in args]
        if not (all(is_digit) or all(is_ndarray)):
            raise TypeError(
                "All arguments must be of the same type - either all (float or int) or all (np.ndarray)"
            )

    @staticmethod
    def check_all_ndarray_has_same_shape(args: list[np.ndarray]) -> None:
        """Check if all ndarray arguments have the same shape.

        Args:
            args: List of numpy arrays to check.

        Raises:
            TypeError: If all arguments do not have the same shape.
        """
        shapes_set = set([arg.shape[0] for arg in args])
        if len(shapes_set) > 1:
            raise TypeError(f"All arguments must have the same shape, got shapes_set:{shapes_set}")

    @staticmethod
    def check_for_sql_injection(expression: str | None) -> None:
        """Check a string for common SQL injection patterns.

        Args:
            expression: The string to check.

        Raises:
            ValueError: If a potential SQL injection pattern is found.
        """
        if expression is None:
            return None

        patterns = [
            (
                r"(?i)\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE){0,1}|"
                r"INSERT( +INTO){0,1}|MERGE|SELECT|UPDATE|UNION( +ALL){0,1})\b"
            ),
            r"(\s*(--|#|\/\*))",  # Comments
            r"(?i)\b(AND|OR)\b\s+[\w\d\'\"]+\s*=\s*[\w\d\'\"]+",  # " OR 1=1
            r"(\s*;\s*)",  # Semicolon for query stacking
            r"(?i)\b(xp_cmdshell|sp_oacreate|sp_oamethod)\b",  # xp_cmdshell
            r"(?i)\b(UTL_HTTP|UTL_FILE)\b",  # Oracle specific
            r"(?i)\b(pg_sleep)\b",  # PostgreSQL specific
            r"(?i)\b(BENCHMARK|SLEEP)\b",  # MySQL specific
        ]

        for pattern in patterns:
            if re.search(pattern, expression):
                raise ValueError(
                    f"Potential SQL injection detected: Expression '{expression}', <{pattern}>"
                )


# ----------------------------- JSON UTILITIES ----------------------------- #

type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]


class JsonUtils:
    """Utility functions for JSON serialization and conversion."""

    @staticmethod
    def jsonify(obj: Any) -> Any:
        """Convert an object to a JSON-compatible representation.

        Handles various Python types including datetime, Enum, BaseModel,
        pandas NaN values, and numpy infinity values.

        Args:
            obj: The object to convert to JSON-compatible format.

        Returns:
            JSON-compatible representation of the object.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value  # or obj.name
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, dict):
            return {k: JsonUtils.jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple | set):
            return [JsonUtils.jsonify(x) for x in obj]
        if pd.isna(obj):
            return None
        if isinstance(obj, float | np.floating) and math.isinf(obj):
            return None
        return obj  # str, int, float, bool, None
