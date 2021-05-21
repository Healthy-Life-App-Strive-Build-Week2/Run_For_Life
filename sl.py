import streamlit as st
import pandas as pd
import pickle
from script import JourneyPredict
import pages.sl_home
import pages.sl_model_info
import pages.sl_test_model
import pages.sl_kivy_app
import pages.sl_future_app




PAGES = {
    "Project Home": pages.sl_home,
    "Model Information": pages.sl_model_info,
    "Test Our Model": pages.sl_test_model,
    "Our Kivy Application": pages.sl_kivy_app,
    "Future Applications": pages.sl_future_app,
}

def main():
    """Main function of the App"""
    st.sidebar.image("img/sblogo.png")
    st.sidebar.title("What you can do with GFit")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    page.app()
    st.sidebar.title("About")
    st.sidebar.info(
    "Team GFit is comprised of aspiring AI engineers with a"
    "succesful project history. You can see our profiles;"
    "\n\n[Umut Aktaş](https://github.com/)"
    "\n\n[Daniel Biman](https://github.com/)"
    "\n\n[Mark Skinner](https://github.com/)"
    "\n\n[GFit Github](https://github.com/)")


if __name__ == "__main__":
    main()
