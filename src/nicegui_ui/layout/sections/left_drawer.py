from nicegui import ui
from nicegui.elements.drawer import LeftDrawer
from functools import partial
from dataclasses import dataclass

async def expand_left_drawer(left_drawer: LeftDrawer):
    if 'mini' in left_drawer._props:
        left_drawer.props(remove='mini')

async def collapse_left_drawer(left_drawer: LeftDrawer):
    if 'mini' not in left_drawer._props:
        left_drawer.props('mini')

@dataclass(slots=True)
class NavigationElement:
    text: str
    icon: str
    link: str


def create_nav_element(element: NavigationElement):
    with ui.item(on_click=lambda: ui.navigate.to(element.link)):
        with ui.item_section().props('avatar'):
            ui.icon(element.icon, color='black')
        with ui.item_section():
            ui.item_label(element.text)

nav_element_list = [
    NavigationElement('Home', 'sym_r_home', '/home'),
    NavigationElement('Experiments', 'sym_r_experiment', '/experiments'),
    NavigationElement('Planner', 'query_stats', '/planner'),
    NavigationElement('Data and Metrics', 'sym_r_database', '/data'),
    NavigationElement('Jobs', 'sym_r_developer_board', '/jobs'),
    NavigationElement('Documentation', 'sym_r_article', '/documents'),
    NavigationElement('Settings', 'sym_r_settings', '/settings'),
]

def create_navigation_list():
    with ui.list().classes('q-px-none'):
        for element in nav_element_list:
            create_nav_element(element)

def create_left_drawer() -> LeftDrawer:
    left_drawer = (
        ui.left_drawer(value=False, top_corner=False, bordered=True)
        .props('show-if-above mini mini-to-overlay width=200')
        .classes('q-pa-none')
    )
    left_drawer.classes.remove('nicegui-drawer')

    left_drawer.on('mouseenter', partial(expand_left_drawer, left_drawer))
    left_drawer.on('mouseleave', partial(collapse_left_drawer, left_drawer))

    with left_drawer:
        create_navigation_list()

    return left_drawer