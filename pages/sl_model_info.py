import streamlit as st
import pandas as pd
import pickle
from script import JourneyPredict


def app():
    st.title('HOW THE MODEL WORKS')
    st.header("A Step by step guide to the inner workings of the model")
    st.write("""in this section we look at how we were able to classify user movement type by sensor readings
    taken from an android device""")
    st.subheader("FEATURES")

    st.write("""
    We used a dataset that already had 70 features based on a selection 16 sensors. Each sensor had a
    minimum, maximum, mean & standard devation for a 0.5 second window of time.

    A combination of 9 sensors and their generated features provided the best accuracy for us, the sensors
    used were;
    """)
    st.info("""android.sensor.linear_acceleration, speed, android.sensor.gyroscope, android.sensor.gyroscope_uncalibrated,android.sensor.accelerometer,sound,
     android.sensor.game_rotation_vector, android.sensor.orientation, android.sensor.rotation_vector
        """)
    st.subheader("MODELS")

    st.write("""
    We used 16 models while developing our final classifier. Predominantly tree models worked the best with Neural Networks and multiplicative models
    performing poorly compared to the speed and accuracy of tree models.
    After selecting the three best performing models;
    """)
    st.info("XGBoost()")
    st.info("RandomForestClassifier()")
    st.info("GradientBoostingClassifier()")

    st.write("""We then tried hyperparameter tuning however this made little difference overall, a big improvement was noticed when we used a
    voting classifier to consolidate all three best performing trees into a single Voting System.""")
    st.info("VotingClassifier()")


    st.subheader("TIME VALIDATION")
    st.write("""by grouping the original data into sequential sensor readings we were able to perform an adjuster
    after our model had made it's predictions. It simply looks at the mode of predctions within a given time DataFrame
    As most predictions are from a minimal amount of sensor readings we then converted the group to the overall
    mode of the time window.""")
    st.image("img/code.png")
