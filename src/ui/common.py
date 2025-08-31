"""Common UI utilities and components."""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, fields
from typing import Any

import streamlit as st

from src.domain.enums import PageMode
from src.ui.chat.schemas import AppContext
from src.ui.data_loaders import get_experiment_by_id, get_job_by_id, get_observation_by_id
from src.ui.state import AppContextManager


@dataclass
class URLParams:
    """URL parameters model for managing query parameters in the application.

    This class handles parsing and validation of URL query parameters used across
    different pages in the application, including observation IDs, job IDs,
    experiment IDs, and page modes.

    Attributes:
        observation_id: The ID of the observation, if any.
        job_id: The ID of the job, if any.
        experiment_id: The ID of the experiment, if any.
        mode: The current page mode, if any.
        submode: Additional submode parameter, if any.
    """

    observation_id: int | None = None
    job_id: int | None = None
    experiment_id: int | None = None
    mode: PageMode | None = None
    submode: str | None = None

    @classmethod
    def parse(cls) -> URLParams:
        """Parse URL query parameters from the current Streamlit session.

        Extracts and validates query parameters from the Streamlit query_params,
        converting string values to appropriate types (int for IDs, PageMode enum
        for mode). Invalid or missing parameters are set to None.

        Returns:
            URLParams: An instance containing the parsed URL parameters.

        Note:
            This method handles KeyError, ValueError, and TypeError exceptions
            gracefully by setting invalid parameters to None.
        """
        params: dict[str, Any] = {}
        for field in fields(cls):
            field_name = field.name
            try:
                if field_name == "mode":
                    params[field_name] = PageMode(st.query_params[field_name])
                elif field_name in ["submode"]:
                    params[field_name] = st.query_params[field_name]
                else:
                    params[field_name] = int(st.query_params[field_name])
            except (KeyError, ValueError, TypeError):
                params[field_name] = None
        return cls(**params)


@dataclass
class PageModeControl:
    """Control component for managing page mode selection.

    This class provides a segmented control interface for switching between different
    page modes in the application using Streamlit's segmented_control widget.

    Attributes:
        mode: The currently selected page mode (List, Update, or Create).
    """

    mode: PageMode | None

    @classmethod
    def render(
        cls,
        page_name: str,
        mode=None,
    ) -> PageModeControl:
        """Renders the page mode control component.

        Creates and displays a segmented control widget that allows users to switch
        between different page modes. Manages session state and URL parameters
        to maintain the selected mode across page interactions.

        Args:
            page_name: The name of the page, used for widget key generation and display.
            mode: Optional initial mode to select. If provided, it will override
                the current session state and clear the mode from query parameters.

        Returns:
            PageModeControl: Instance with the currently selected mode.

        Note:
            If the selected mode is not "Create", any "obs_to_copy" data in the
            session state will be cleared.
        """
        widget_key = f"{page_name}_page_mode"
        options = PageMode.list()
        if widget_key not in st.session_state:
            st.session_state[widget_key] = options[0]

        if mode:
            st.session_state[widget_key] = mode
            del st.query_params["mode"]

        is_blocked = st.session_state.get("block_widgets_on_rerun_number", -1) == st.session_state.get(
            "rerun_counter", 0
        )
        selected_mode = st.segmented_control(
            label="Section",
            options=options,
            format_func=lambda x: f"{page_name}: {x}",
            selection_mode="single",
            label_visibility="hidden",
            key=widget_key,
            disabled=is_blocked,
        )
        if st.session_state[widget_key] != "Create" and "obs_to_copy" in st.session_state:
            del st.session_state["obs_to_copy"]

        return cls(mode=PageMode(selected_mode) if selected_mode else None)


def put_return_in_app_ctx(render_fn):
    """Decorator that handles app context.

    Writes data in session_state["app_context"]["data"]
    if result from Page.render() is not None.

    Args:
        render_fn: The function to wrap that returns a dict or None.

    Returns:
        Wrapped function that processes return values into app context.
    """

    @functools.wraps(render_fn)
    def wrapper(*args, **kwargs):
        result = render_fn(*args, **kwargs)  # return dict or None
        if result is None:
            return wrapper

        for key, value in result.items():
            AppContextManager.add_selected(key, value)

    return wrapper


def enrich_app_ctx_experiment() -> None:
    app_ctx = AppContextManager.get_or_create_state()
    if selected_experiment_id := app_ctx.selected.get("experiment_id"):
        selected_experiment = get_experiment_by_id(selected_experiment_id)
        value: dict[str, Any] = asdict(selected_experiment) if selected_experiment else {}
        AppContextManager.add_selected("experiment", value)


def enrich_app_ctx_observation() -> None:
    app_ctx = AppContextManager.get_or_create_state()
    if selected_observation_id := app_ctx.selected.get("observation_id"):
        selected_observation = get_observation_by_id(selected_observation_id)
        value: dict[str, Any] = asdict(selected_observation) if selected_observation else {}
        if not app_ctx.selected.get("selected_experiment_id"):
            AppContextManager.add_selected("experiment_id", value.get("experiment_id"))
        AppContextManager.add_selected("observation", value)


def enrich_app_ctx_job() -> None:
    app_ctx = AppContextManager.get_or_create_state()
    if selected_job_id := app_ctx.selected.get("job_id"):
        selected_job = get_job_by_id(selected_job_id)
        value_full: dict[str, Any] = asdict(selected_job) if selected_job else {}
        value: dict[str, Any] = dict(filter(lambda kv: kv[0] not in ("query",), value_full.items()))
        if not app_ctx.selected.get("observation_id"):
            AppContextManager.add_selected("observation_id", value_full.get("observation_id"))
        AppContextManager.add_selected("job", value)


def enrich_app_ctx() -> AppContext:
    enrich_app_ctx_job()
    enrich_app_ctx_observation()
    enrich_app_ctx_experiment()
    return AppContextManager.get_or_create_state()
