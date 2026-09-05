#src/nicgeui_ui/main

from nicegui import ui
from fastapi import Request
from src.nicegui_ui.layout.shell import create_shell
from src.nicegui_ui.pages.power_analysis import render_power_analysis_page
from functools import partial

async def home_page(request: Request):
    ui.label(request.url.path)
    ui.label('Home')

async def experiments_page():
    ui.label('Experiments')

async def planner_page():
    ui.label('Planner')

async def data_page():
    ui.label('Data')

async def jobs_page():
    ui.label('Jobs')

async def documents_page():
    ui.label('Documents')

async def settings_page():
    ui.label('Settings')

async def power_analysis(request: Request):
    render_power_analysis_page(request)



@ui.page('/')
@ui.page('/{_:path}')
async def main(request: Request):  # <-- принять request # <-- передать path
    ui.sub_pages({
        '/': home_page,
        '/home': partial(home_page, request),
        '/experiments': experiments_page,
        '/planner':partial(render_power_analysis_page, request),
        '/data': data_page,
        '/jobs': jobs_page,
        '/documents': documents_page,
        '/settings': settings_page,
    }).classes('full-width', remove='')
    await create_shell(request)


ui.run()