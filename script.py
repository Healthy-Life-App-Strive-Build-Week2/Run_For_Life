import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime as dt
import time
import os
from sklearn                    import metrics
from sklearn                    import svm
from sklearn.preprocessing      import StandardScaler
from sklearn                    import pipeline
from sklearn.experimental       import enable_hist_gradient_boosting # for HistGradientBoostingClassifier
from sklearn.ensemble           import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from xgboost                    import XGBClassifier
from sklearn                    import impute
from sklearn                    import compose
from statistics                 import mode



class JourneyPredict:
    def __init__(self):
        self.clf = pickle.load(open("SavedModels/2best_model.pickle",'rb'))
        self.vars = vars = [
                "android.sensor.linear_acceleration#mean",
                "android.sensor.linear_acceleration#min",
                "android.sensor.linear_acceleration#max",
                "android.sensor.linear_acceleration#std",
                "speed#min",
                "speed#mean",
                "speed#max",
                "speed#std",
                "android.sensor.gyroscope#min",
                "android.sensor.gyroscope#std",
                "android.sensor.gyroscope#max",
                "android.sensor.gyroscope#mean",
                "android.sensor.gyroscope_uncalibrated#mean",
                "android.sensor.gyroscope_uncalibrated#max",
                "android.sensor.gyroscope_uncalibrated#std",
                "android.sensor.gyroscope_uncalibrated#min",
                "android.sensor.accelerometer#std",
                "android.sensor.accelerometer#max",
                "android.sensor.accelerometer#min",
                "android.sensor.accelerometer#mean",
                "sound#mean",
                "sound#min",
                "sound#max",
                "sound#std",
                "android.sensor.game_rotation_vector#max",
                "android.sensor.game_rotation_vector#mean",
                "android.sensor.game_rotation_vector#std",
                "android.sensor.game_rotation_vector#min",
                "android.sensor.orientation#std",
                "android.sensor.orientation#mean",
                "android.sensor.orientation#max",
                "android.sensor.orientation#min",
                "android.sensor.rotation_vector#max",
                "android.sensor.rotation_vector#std",
                "android.sensor.rotation_vector#mean",
                "android.sensor.rotation_vector#min",

        ]

    def predict(self, day_csv):
        df = pd.read_csv(day_csv)
        x_test = df[self.vars]
        y_test = df['target'].map({"Bus":2, "Car":2, "Still": 4, "Train":1, "Walking": 3})
        preds = self.clf.predict(x_test)
        preds_score = metrics.accuracy_score(y_test, preds)*100
        refine_preds, df_preds = self.refine_by_group(preds, df)
        refine_score =  metrics.accuracy_score(y_test, refine_preds)*100
        print(f"ORIGINAL PREDICTION ACC: {preds_score}")
        print(f"REFINED PREDICTION ACC: {refine_score}")
        return refine_preds, df_preds, round(refine_score,2), round(preds_score,2)

    def refine_by_group(self, preds, test_df):
        df = test_df.copy()
        df['preds'] = preds
        groups = list(df['TimeGroup'].unique())
        for group in groups:
            all_group = df[df['TimeGroup'] == group]
            mode_group = mode(list(all_group['preds']))
            df.loc[(df['TimeGroup'] == group), 'preds'] = mode_group

        ref_preds = np.array(df['preds'].values)

        return ref_preds, df


if __name__ == "__main__":
    predictor = JourneyPredict()

#
# start_time = time.time()
# preds, df_preds = predictor.predict('Test_users/Day2.csv')
# print(df_preds.head())
# df_preds.sort_values('TimeGroup')
# print(df_preds.head())
#
#
# end_time = time.time() - start_time
# print(end_time)
#

























#     def __init__(self, data_csv):
#         self.ODF = pd.read_csv(data_csv)
#         self.vars =[
#                 "android.sensor.linear_acceleration#mean",
#                 "android.sensor.linear_acceleration#min",
#                 "android.sensor.linear_acceleration#max",
#                 "android.sensor.linear_acceleration#std",
#                 "speed#min",
#                 "speed#mean",
#                 "speed#max",
#                 "speed#std",
#                 "android.sensor.gyroscope#min",
#                 "android.sensor.gyroscope#std",
#                 "android.sensor.gyroscope#max",
#                 "android.sensor.gyroscope#mean",
#                 "android.sensor.gyroscope_uncalibrated#mean",
#                 "android.sensor.gyroscope_uncalibrated#max",
#                 "android.sensor.gyroscope_uncalibrated#std",
#                 "android.sensor.gyroscope_uncalibrated#min",
#                 "android.sensor.accelerometer#std",
#                 "android.sensor.accelerometer#max",
#                 "android.sensor.accelerometer#min",
#                 "android.sensor.accelerometer#mean",
#                 "sound#mean",
#                 "sound#min",
#                 "sound#max",
#                 "sound#std",
#                 "android.sensor.game_rotation_vector#max",
#                 "android.sensor.game_rotation_vector#mean",
#                 "android.sensor.game_rotation_vector#std",
#                 "android.sensor.game_rotation_vector#min",
#                 "android.sensor.orientation#std",
#                 "android.sensor.orientation#mean",
#                 "android.sensor.orientation#max",
#                 "android.sensor.orientation#min",
#                 "android.sensor.rotation_vector#max",
#                 "android.sensor.rotation_vector#std",
#                 "android.sensor.rotation_vector#mean",
#                 "android.sensor.rotation_vector#min",
#
#         ]
#         self.PPDF = None
#         self.train_users = None
#         self.x_train = None
#         self.y_train = None
#         self.clf = None
#
#         self.get_train_set()
#         self.load_model()
#
#
#
#     def create_model(self):
#         num_pipe_tree = pipeline.Pipeline(steps=[
#                     ('imputer', impute.SimpleImputer(strategy="constant", fill_value=0)),#fill_value=0
#         ])
#         tree_pipe = compose.ColumnTransformer(transformers=[
#                     ('nums', num_pipe_tree, self.vars)],
#                     remainder='drop'
#                     )
#         vc_models = [
#                 ("Random Forest",RandomForestClassifier(random_state=909)),
#                 ("Skl HistGBM",HistGradientBoostingClassifier(random_state=909)),
#                 ("XGBoost",XGBClassifier(use_label_encoder=False)),
#                 ]
#         clf = pipeline.make_pipeline(tree_pipe,VotingClassifier(estimators=vc_models, voting='hard'))
#
#         return clf
#
#     def get_train_set(self):
#         self.PPDF = self.apply_time_window_group(self.ODF)
#         train_users = df[(df['user'] != "U12")  & (df['user'] != "U9") & (df['user'] != "U2")]
#         day1 = df[(df['user'] == "U12")]
#         day2 = df[df['user'] == "U9"]
#         day3 = df[df['user'] == "U2"]
#         day1.to_csv("Test_users/Day1.csv")
#         day2.to_csv("Test_users/Day2.csv")
#         day3.to_csv("Test_users/Day3.csv")
#         return adf, train_users
#
#
#
#     def load_model(self):
#         if os.path.exists("best_model.pickle"):
#             clf = pickle.load("best_model.pickle")
#             return clf
#         else:
#             clf = create_model()
#             x_train = train_users[vars]
#             y_train = train_users['target'].map({"Bus":2, "Car":2, "Still": 4, "Train":1, "Walking": 3})
#             clf.fit(x_train, y_train)
#             new_file_name = "best_model.pickle"
#             pickle.dump(clf, open(new_file_name, 'wb'))
#             return clf
#
#
#     def get_time_window_changes(self, adf):
#         adf.sort_values('id',inplace=True)
#         time_counter = 0
#         change_index = []
#         for index, row in adf.iterrows():
#             if row['time'] > time_counter:
#                 time_counter = row['time']
#             else:
#                 time_counter = 0
#                 change_index.append(row["id"])
#         return change_index, adf
#
#     def apply_time_window_group(self):
#         windows, adf = get_time_window_changes()
#         window_names = [(i, windows[i]) for i in range(len(windows))]
#         max_id = None
#         adf['TimeGroup'] = -1
#         for win in window_names:
#              adf.loc[(adf['id'] < win[1]) & (adf['TimeGroup'] == -1), 'TimeGroup'] = win[0]
#              if max_id == None or win[1] > max_id:
#                  max_id = win[1]
#         adf.loc[(adf["id"] >= max_id), 'TimeGroup'] = len(window_names)
#         return adf
#
#
#
#     def refine_by_group(preds, test_df):
#         df = test_df.copy()
#         df['preds'] = preds
#         groups = list(df['TimeGroup'].unique())
#         for group in groups:
#             all_group = df[df['TimeGroup'] == group]
#             mode_group = mode(list(all_group['preds']))
#             df.loc[(df['TimeGroup'] == group), 'preds'] = mode_group
#
#         payload = np.array(df['preds'].values)
#
#         return payload
#
#
# def predict_day(day_csv):
#
#
#
# model = JourneyPredict("dataset_halfSecondWindow.csv")
