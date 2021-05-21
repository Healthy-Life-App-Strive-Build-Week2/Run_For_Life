import streamlit as st
import pandas as pd
import pickle
from script import JourneyPredict



def app():
    st.title('THE FUTURE OF OUR APP')

    st.write("""This App will calculate approximately how much calories you burn right now, based on your current activity. The classification of your activity from mobile phone sensor data is from a ML model. It gives you an overview how many calories you burned during the day. It also tells you in textform how you spent your day for example: 'You spent 2 hours in a train' 'You have been sitting for 6 hours' 'You walked today one hour'.""")
