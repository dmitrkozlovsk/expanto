from nicegui import ui

def apply_colors_theme():
    ui.colors(
        # brand colors (Quasar)
        primary   ='#1C1C1C',   # primary emphasis (buttons, active tabs)
        secondary ='#ECEBE4',   # soft neutral
        accent    ='#1C1C1C',   # keep it monochrome
        # custom named colors
        ink        ='#1C1C1C',
        bg         ='#FAFAFF',
        surface    ='#FFFFFF',
        surface2   ='#EEF0F2',
        surfacewarm='#ECEBE4',
        border     ='#E7E9EC',
    )