from nicegui import ui
from typing import Coroutine

from src.nicegui_ui.layout.sections.right_drawer import create_right_drawer
from src.nicegui_ui.layout.sections.left_drawer import create_left_drawer
from src.nicegui_ui.layout.sections.header import create_header
from src.nicegui_ui.layout.styling import apply_colors_theme


async def create_shell():
    """Create the shell for the application."""
    
    apply_colors_theme()
    left_drawer = create_left_drawer()
    right_drawer = create_right_drawer()
    create_header(right_drawer.toggle)