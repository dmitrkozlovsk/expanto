"""Observations page router.

This module acts as a router for the observations page. It parses the URL
parameters to determine the page mode (LIST, CREATE, UPDATE) and then renders
the appropriate subpage.
"""

from src.domain.enums import PageMode
from src.ui.common import PageModeControl, URLParams
from src.ui.layout import AppLayout
from src.ui.observations.subpages import (
    CreateObservationPage,
    ObservationListPage,
    UpdateObservationPage,
)
from src.ui.state import AppContextManager

url_params = URLParams.parse()

AppContextManager.set_page_name("observations")

with AppLayout.chat():
    pmc = PageModeControl.render(page_name="Observations", mode=url_params.mode)
    AppContextManager.set_page_mode(pmc.mode if pmc.mode else "")
    match pmc.mode:
        case PageMode.LIST:
            ObservationListPage.render(url_params.experiment_id)
        case PageMode.CREATE:
            CreateObservationPage.render(url_params)
        case PageMode.UPDATE:
            UpdateObservationPage.render(url_params)
