
import streamlit as st

from src.ui.state import AppContextManager
from src.ui.resources import init_resources

st.set_page_config(
    page_title="Expanto",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/dmitrkozlovsk/expanto',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "https://github.com/dmitrkozlovsk/expanto"
    })

_,_,_,_  = init_resources() # warm-up all resources

st.markdown("""
    <style>
    html, body, .stApp {
        font-size: 14px;
    }
    .block-container {
        padding-top: 3rem !important;
    }
    div.block-container[data-testid="stMainBlockContainer"] {
        padding-bottom: 1rem !important;
    }
    [data-testid="stVerticalBlock"] { gap: .7rem; }
    [data-testid="block-container"] { padding: .7rem; }
    div[data-testid="stVerticalBlock"].st-key-chat_wrap{
      height: 75dvh;                 
      flex: 0 0 75dvh !important;   
      overflow-y: auto;
      display: flex; flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("Expanto")

# chat toggle
st.sidebar.markdown("---")
chat_enabled = st.sidebar.toggle(
    "💬 Chat",
    value=st.session_state.get("chat_enabled", False),
    key="chat_enabled",
)

# app context
st.sidebar.divider()
with st.sidebar.expander('Show app context', expanded=False):
    app_ctx = AppContextManager.get_or_create_state()
    st.write(f"Page name: `{app_ctx.page_name}`")
    st.write(f"Page mode: `{app_ctx.page_mode}`")
    st.write(app_ctx.selected)

experiments_page = st.Page("src/ui/experiments/page.py", title="Experiments", url_path="experiments")
observations_page = st.Page("src/ui/observations/page.py", title="Observations", url_path="observations")
planner_page = st.Page("src/ui/planner/page.py", title="Planner", url_path="planner")
results_page = st.Page("src/ui/results/page.py", title="Results", url_path="results")

pg = st.navigation([
    experiments_page,
    observations_page,
    results_page,
    planner_page
], position='top')

pg.run()
