import streamlit as st

st.set_page_config(
    page_title="Monte Carlo Diffusion Dashboard",
    page_icon="🧪",
    layout="wide"
)

st.title("Monte Carlo Diffusion Dashboard 🧪")

st.markdown("""
Welcome to the Monte Carlo diffusion simulation dashboard!

- **Classical diffusion**: standard KMC simulation
- **Modified diffusion**: KMC simulation with custom parameters (question II)

Use the sidebar menu to navigate between pages.
""")

st.sidebar.success("Select a page on the left.")