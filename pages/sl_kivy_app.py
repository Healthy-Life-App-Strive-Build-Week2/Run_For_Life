import streamlit as st
import pandas as pd
import pickle
from script import JourneyPredict

def app():
    st.title('CREATING A MULTIPLATFORM APPLICATION')
    st.header("We used kivy to start developing our app, here's where we are upto")

    st.subheader("PREMISE")
    st.write("""
    The purpose of our application is to group  a users activities throughout the day and
    provide them  a calorie burn information for that period.
    """)
    st.subheader("USER ENTERS DETAILS")
    st.write("""The opening screen prompts the user to enter details, they can jump past this selection but the functionality
    of the app will deteoriate as insights will not be personalised""")
    st.image("img/openscreen.png")

    st.subheader("USER SELECTS")
    st.write("""The user then selects the day they wish to get insights from and the application will present details about the day
    including some useful graphs""")
    st.image("img/ss.png")

    st.info("""
    Unfortunately final graphing functionality is still in development, we want to have it implemented prior to the application
    launch so there are no real insights to be gained from it yet, test the model on streamlit!!
    """)
