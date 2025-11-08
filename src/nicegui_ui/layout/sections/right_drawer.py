from nicegui import ui
from nicegui.elements.drawer import RightDrawer

def create_right_drawer() -> RightDrawer:
    right_drawer = (
        ui.right_drawer(value=False, top_corner=True, bordered=True, fixed=False)
        .props('width=450 bordered')
        .classes('h-full q-pa-sm')
    )
    return right_drawer
