from nicegui import ui
from src.nicegui_ui.layout.shell import create_shell
from src.nicegui_ui.pages.power_analysis import render_power_analysis_page

async def home_page():
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


@ui.page('/')
@ui.page('/{_:path}')
async def main():
    await create_shell()
    ui.sub_pages({
        '/': home_page,
        '/home': home_page,
        '/experiments': experiments_page,
        '/planner': render_power_analysis_page,
        '/data': data_page,
        '/jobs': jobs_page,
        '/documents': documents_page,
        '/settings': settings_page,
    }).classes('full-width', remove='')


ui.run()