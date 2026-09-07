from nicegui import ui

COLORS = {
    "primary":   '#1C1C1C',   # primary emphasis (buttons, active tabs)
    "secondary": '#ECEBE4',   # soft neutral
    "accent":    '#1C1C1C',   # keep it monochrome
        # custom named colors
    "ink":        '#1C1C1C',
    "bg":         '#FAFAFF',
    "surface":    '#FFFFFF',
    "surface2":   '#EEF0F2',
    "surfacewarm":'#ECEBE4',
    "border":     '#E7E9EC',
}

def apply_colors_theme():
    ui.colors(
        **COLORS
    )
    ui.query('body').style('background-color: var(--q-surfacewarm);')

def apply_styles_theme():
    ui.add_css('''
    :root {
        --nicegui-default-padding: 0rem;
        --nicegui-default-gap: 0rem;
        }
    ''')
    apply_colors_theme()
