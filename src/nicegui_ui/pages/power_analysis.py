from nicegui import ui
from dataclasses import dataclass, asdict, astuple
from functools import partial

@dataclass(slots=True, frozen=True)
class Experiment:
    id: int
    name: str

exp_list = [
    Experiment(1, 'one'),
    Experiment(2, 'two'),
]

def render_exp_selector():
    selector = (
        ui.select(
            options=[asdict(exp) for exp in exp_list],
            label='Select experiment',
            with_input=True,
            on_change=lambda e: ui.notify(e.value),  # e.value = dict {'id':..,'name':..}
        )
        .classes('full-width')
        .props('''
            dense
            clearable
            outlined
            :option-label="opt => opt.label.name"
        ''')

    )

def render_calc_scenario_selector():
    selector = (
        ui.select(
            options=['scenario1', 'scenario2'],
            label='Select scenario',
            with_input=True,
            on_change=lambda e: ui.notify(e.value),  # e.value = dict {'id':..,'name':..}
        )
        .classes('full-width')
        .props('''
            dense
            clearable
            outlined
        ''')
    )

def render_metric_selector():
    selector = (
        ui.select(
            options=['metric1', 'metric2'],
            multiple=True,
            label='Select Metrics',
            with_input=True,
            on_change=lambda e: ui.notify(e.value),  # e.value = dict {'id':..,'name':..}
        )
        .classes('full-width')
        .props('''
            dense
            clearable
            outlined
            use-chips
        ''')
    )

def render_calc_period_selector():
    input_date = ui.input('Date Range', value='2025-12-01 - 2025-12-14') \
        .classes('full-width')\
        .props('dense outlined input-class="text-xs"')\
        .style('min-width: 180px')
    input_date.on_value_change(lambda e: ui.notify(e.value))
    with input_date.add_slot('append'):
        ui.icon('date_range')
        with ui.element('q-popup-proxy').props('cover transition-show="scale" transition-hide="scale"'):
            ui.date()\
                .props('range')\
                .bind_value(
                    input_date,
                    forward=lambda x: f'{x["from"]} - {x["to"]}' if x else None,
                    backward=lambda x: {
                        'from': x.split(' - ')[0],
                        'to': x.split(' - ')[1],
                    } if ' - ' in (x or '') else None,
                )\
                .on_value_change(lambda e: ui.notify(e.value))
    ui.notify(f'value: [{input_date.value}] ')

def render_exposure_period_selector():
    input_date = ui.input('Date Range', value='2025-12-01 - 2025-12-14') \
        .classes('full-width') \
        .props('dense outlined input-class="text-xs"') \
        .style('min-width: 180px')
    input_date.on_value_change(lambda e: ui.notify(e.value))
    with input_date.add_slot('append'):
        ui.icon('date_range')
        with ui.element('q-popup-proxy').props('cover transition-show="scale" transition-hide="scale"'):
            ui.date() \
                .props('range') \
                .bind_value(
                input_date,
                forward=lambda x: f'{x["from"]} - {x["to"]}' if x else None,
                backward=lambda x: {
                    'from': x.split(' - ')[0],
                    'to': x.split(' - ')[1],
                } if ' - ' in (x or '') else None,
            ) \
                .on_value_change(lambda e: ui.notify(e.value))
    ui.notify(f'value: [{input_date.value}] ')

def render_customer_query_selector():
    selector = (
        ui.select(
            options=[asdict(exp) for exp in exp_list],
            label='Select experiment',
            with_input=True,
            on_change=lambda e: ui.notify(e.value),  # e.value = dict {'id':..,'name':..}
        )
        .classes('full-width')
        .props('''
            dense
            clearable
            outlined
            :option-label="opt => opt.label.name"
        ''')

    )

def render_exposure_event_input():
    input = (
        ui.input(
            label='Input Exposure Event',
            on_change=lambda e: ui.notify(e.value),  # e.value = dict {'id':..,'name':..}
            autocomplete=['event1', 'event2'],
        )
        .classes('full-width')
        .props('''
            dense
            clearable
            outlined
        ''')
    )

def render_split_id_input():
    input = (
        ui.input(
            label='Input Split Id',
            on_change=lambda e: ui.notify(e.value),  # e.value = dict {'id':..,'name':..}
            autocomplete=['customer_id', 'user_id', 'device_id'],
        )
        .classes('full-width')
        .props('''
            dense
            clearable
            outlined
        ''')
    )

def render_advanced_settings():

    ui.add_css("""
        .q-expansion-item__content { padding: 0 !important; }
        """)
    def open_dialog():
        with ui.dialog().props('backdrop-filter="blur(8px) brightness(40%)"') as dialog:
            card = ui.card().props('bordered').classes('w-full rounded q-pa-none', remove='nicegui-card')
            with card:
                ui.codemirror(language='SQL', theme='solarizedLight').classes('q-px-none')

        dialog.on('escape-key', lambda: ui.notify('ESC pressed'))
        dialog.open()




    button_audience_sql = (
        ui.button(text='Audience SQL', color='surface2', icon='sym_r_database')
        .props('bordered text-color=primary unelevated size=md')
        .classes('w-full q-pa-none shadow-0', remove='nicegui-button')
        .on_click(open_dialog)
    )
    button_audience_start_calc = (
        ui.button(text='fetch hist data', icon='arrow_right')
        .props('bordered rounded size=md')
        .classes('w-full q-pa-none shadow-0', remove='nicegui-button')
    )


def render_power_analysis_page():
    row = ui.row().classes(add='full-width no-wrap justify-center q-px-none q-px-md ', remove='nicegui-row')
    with row:
        left_card = (
            ui.column()
            .props('bordered')
            .classes(
            add='h-[calc(100vh-35px)] min-w-54 col-3 bg-surface q-pa-md q-gutter-y-md justify-start overflow-y-auto border-r-[1px] border-[#e8e8e8]',
            remove='nicegui-column')
        )
        with left_card:
            render_exp_selector()
            render_calc_scenario_selector()
            render_metric_selector()
            render_exposure_period_selector()
            render_calc_period_selector()
            render_exposure_event_input()
            render_split_id_input()
            render_advanced_settings()
        right_card = ui.card().classes('col-9 shadow-1 q-ma-md rounded-md', remove='nicegui-card')\
            .style('height: fit-content;')
        with right_card:
            echart = ui.echart(
            {
                'grid': {'left': 16, 'top': 16, 'right': 16, 'bottom': 16, 'containLabel': False},
                'yAxis': {'type': 'value',
                          'min': 0,
                          'max': 63,
                          'interval': 7,
                          'splitLine': { 'show': True }
                          },
                'xAxis': {'type': 'category', 'data': ['A', 'B'], 'inverse': True},
                'legend': {'textStyle': {'color': 'gray'}},
                'series': [
                    {'type': 'bar', 'name': 'Alpha', 'data': [0.1, 0.2]},
                    {'type': 'bar', 'name': 'Beta', 'data': [0.3, 0.4]},
                ],
            }
            ).classes('h-100 q-pa-sm')