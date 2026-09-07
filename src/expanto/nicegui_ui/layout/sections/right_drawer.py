from nicegui import ui
from nicegui.elements.drawer import RightDrawer

def create_right_drawer() -> RightDrawer:
    right_drawer = (
        ui.right_drawer(value=False, top_corner=True, bordered=True, fixed=False)
        .props('width=400 bordered')
        .classes('height-full')
    )
    return right_drawer
