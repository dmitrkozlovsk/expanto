"""Experiment page router module. It is the entry point for the experiments page.

This module handles routing between different experiment-related pages based on the URL parameters.
It supports three main modes:
    listing experiments, creating new experiments, and updating existing experiments.
"""

from src.domain.enums import PageMode
from src.ui.common import PageModeControl, URLParams
from src.ui.experiments.subpages import (
    CreateExperimentPage,
    ExperimentListPage,
    UpdateExperimentPage,
)
from src.ui.layout import AppLayout
from src.ui.state import AppContextManager

AppContextManager.set_page_name("experiments")

url_params = URLParams.parse()
with AppLayout.chat():
    pmc = PageModeControl.render(page_name="Experiments", mode=url_params.mode)
    AppContextManager.set_page_mode(pmc.mode if pmc.mode else "")
    match pmc.mode:
        case PageMode.LIST:
            ExperimentListPage.render()
        case PageMode.CREATE:
            CreateExperimentPage.render()
        case PageMode.UPDATE:
            UpdateExperimentPage.render(url_exp_id=url_params.experiment_id)
