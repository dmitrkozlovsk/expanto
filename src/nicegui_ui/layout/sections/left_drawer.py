from nicegui import ui
from nicegui.elements.drawer import LeftDrawer
from functools import partial
from dataclasses import dataclass

@dataclass(slots=True)
class NavigationElement:
    text: str
    icon: str
    link: str

navigation_element_list = [
    NavigationElement('Home', 'sym_r_home', '/home'),
    NavigationElement('Experiments', 'sym_r_experiment', '/experiments'),
    NavigationElement('Power Analysis', 'sym_r_calculate', '/planner'),
    NavigationElement('Data and Metrics', 'sym_r_database', '/data'),
    NavigationElement('Jobs', 'sym_r_developer_board', '/jobs'),
    NavigationElement('Documentation', 'sym_r_article', '/documents'),
    NavigationElement('Settings', 'sym_r_settings', '/settings'),
]

def expand_left_drawer(left_drawer: LeftDrawer):
    if 'mini' in left_drawer._props:
        left_drawer.props(remove='mini')

def collapse_left_drawer(left_drawer: LeftDrawer):
    if 'mini' not in left_drawer._props:
        left_drawer.props('mini')

def create_navigation_list(nav_element_list: list[NavigationElement]):
    with ui.list().classes('q-pa-none'):
        for element in nav_element_list:
            with ui.item(on_click=partial(ui.navigate.to, element.link)).classes('q-pa-none') as item:
                #todo: create if active -> item.props('active active-class=bg-grey-3')
                with ui.item_section().props('avatar').classes('content-center q-pr-none'):
                    ui.icon(element.icon, color='black').classes('q-pa-none')
                with ui.item_section():
                    ui.item_label(element.text)

def create_left_drawer() -> LeftDrawer:
    left_drawer = (
        ui.left_drawer(value=True, top_corner=False, bordered=True)
        .props('show-if-above mini mini-to-overlay width=200')
        .classes('q-pa-none') #shadow-[8px_0_8px_-12px_rgba(0,0,0,0.25)]
    )
    left_drawer.classes.remove('nicegui-drawer')

    left_drawer.on('mouseenter', partial(expand_left_drawer, left_drawer))
    left_drawer.on('mouseleave', partial(collapse_left_drawer, left_drawer))

    with left_drawer:
        create_navigation_list(navigation_element_list)

    return left_drawer