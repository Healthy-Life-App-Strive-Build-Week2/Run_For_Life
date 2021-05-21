import streamlit as st
import pandas as pd
import pickle
from script import JourneyPredict




def app():

    st.title('GoogleFit Journey Classifier')
    st.image('img/ggif.gif')
    st.header("A product by Team GoogleFit")
    st.subheader("Using AI to make life simpler")

    og_data = pd.read_csv("dataset_halfSecondWindow.csv", index_col='id')

    st.header("THE TARGET")
    st.write(f"""
    The purpose of this project was to produce a classification model that detects a journey type from from phone sensor readings.
    The categories of journey we are trying to classify are;
    """)
    st.write("1. On a train")
    st.write("2. On the road (e.g. bus/car)")
    st.write("3. Walking")
    st.write("4. Sitting still")



    st.header("THE DATA")
    st.write(f"""
    We used a set of 229151 sensor readings taken from 100's of different journeys to train our model.
    The data set we used can be seen here.
    """)
    original_dataset = st.empty()
    original_data_view = st.button("View Dataset", key="odv")
    if original_data_view:
        original_dataset.write(og_data)
        close_original = st.button("Hide Dataset", key="co")
        if close_original:
            original_dataset= st.empty()
    st.subheader("WHAT WE DID WITH THE DATA")

    st.write(f"""
    We first ran some functions on the data to decide where a recording started and ended, this would prove vital
    later for getting consistencies over time windows as the dataset was organised in such a way that time had lost
    it's true meaning.

    In later production when we take live data from a device this step will not be necessary. We did it at this stage
    of development to replicate sequential sensor readings as the original data sometimes is not correctly ordered.
    """)

    st.subheader("HALT, DATA LEAKS!")
    st.write(f"""
    Our first models ran at 97-99% accuracy with no tuning, we were incredibly suspicous of our results. We isolated a new user
    that the model hasn't seen before we found that 'user traits' were leaking test data to our training model.

    We then took the approach to isolate individual users for testing, our predictions dropped to 45% accuracy, now we had some work to do!
    After taking sometime to analyze the feature importance and different combination of ML models we were able to improve this upto a whopping 85.14%
    over 4 classes and 97% over 3 classes.
    """)

    st.header("VIEW HOW OUR FINAL MODEL WORKS")

    view_model_workings = st.button("View the model details", key="view_model")
