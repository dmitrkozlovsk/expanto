from typing import Callable
from nicegui import ui

def create_header(on_assist_btn_click: Callable[[], None]) -> None:

    header = ui.header(bordered=True).classes('bg-surface text-black q-pa-xs') #shadow-[0_8px_8px_-12px_rgba(0,0,0,0.25)]
    header.classes.remove('nicegui-header')

    with header:
        with ui.row().classes('w-full h-full items-center justify-between'):
            ui.label('EXPANTO').classes("q-pl-lg text-weight-bolder")
            with ui.element("div"):
                ui.button(icon='casino').props('flat color=ink round')\
                    .on('click', on_assist_btn_click)