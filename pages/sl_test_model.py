import streamlit as st
import pandas as pd
import pickle
from script import JourneyPredict



def app():
    st.title('TEST OUR MODEL')
    st.subheader("How to use this feature")
    st.write("""This section simulates the core of our application working, you can select
    a day of the week and analyze your journey types for that day""")
    st.write("""To start please select a day of the week, you can also view the raw test data.""")
    st.write("""Simply press the predict button and it will provide details about the journey.""")
    test_data_place = st.empty()
    day_csvs = {"Monday": "Test_users/12day1.csv",
                "Wednesday": "Test_users/10day2.csv",
                "Saturday": "Test_users/2day3.csv",
                }
    PRED_MAPPER= {2:"On The Road", 4:"Sitting Still", 1: "On The Train", 3:"Walking"}
    ESTIMATOR = JourneyPredict()
    fbs = st.selectbox("Choose a day to predict journeys", list(day_csvs.keys()))
    view_data = st.button("Raw view of data", key="vd")
    predict = st.button("Predict this data", key="pd")


    st.header("YOUR RESULTS WILL BE DISPLAYED HERE")
    if view_data:
        print(view_data)
        test_data_place.write(pd.read_csv(day_csvs[fbs]))
        close_data= st.button("Hide dataset", key="ca")
        if close_data:
            augmented_dataset = st.empty()
    if predict:
        predictions_from_csv, prediction_df, ref_score, og_score = ESTIMATOR.predict(day_csvs[fbs])
        st.write(f"Excellent we have classified this day with an astonishing {ref_score}% accuracy")
        st.write(f"Without timeslot validation we only correctly classified {og_score}%" )
        st.header("Useful stats from the classification")
        count_journey_numbers = len(prediction_df['TimeGroup'].unique())
        gb_journey = prediction_df.groupby('TimeGroup')['preds'].count()
        gb_preds = prediction_df.groupby('preds')['preds'].count()
        unique_journeys = list(prediction_df['preds'].unique())
        st.info(f"We detected {count_journey_numbers} journeys on this day")
        st.info(f"We detected they used {len(unique_journeys)} types of transport on this day;")
        for j in unique_journeys:
            st.info(f"User recorded action: {PRED_MAPPER[j]}")
