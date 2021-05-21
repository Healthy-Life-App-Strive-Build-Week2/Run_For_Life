import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn                    import metrics
from sklearn.linear_model       import LogisticRegression, LinearRegression, RidgeClassifier, Lasso, SGDClassifier
from sklearn.neural_network     import MLPClassifier
from sklearn                    import svm
from sklearn.preprocessing      import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.cluster            import KMeans
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.naive_bayes        import GaussianNB, MultinomialNB
from sklearn.model_selection    import cross_val_score, ShuffleSplit, GridSearchCV, train_test_split, StratifiedKFold, cross_val_predict
from sklearn                    import pipeline
from sklearn.tree               import DecisionTreeClassifier
from sklearn.experimental       import enable_hist_gradient_boosting # for HistGradientBoostingClassifier
from sklearn.ensemble           import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from xgboost                    import XGBClassifier
#from lightgbm                   import LGBMClassifier
from catboost                   import CatBoostClassifier
from sklearn                    import impute
from sklearn                    import compose
import datetime as dt
import time
import os
from sklearn.metrics import confusion_matrix
from statistics import mode
import pickle


#### PURPOSE OF MODEL: To predict whether a user is walking, sitting still or in transportation from phone sensors
df = pd.read_csv("dataset_halfSecondWindow.csv")
def consistency_adjust(preds):
    adjusted_preds = list(preds[0:2])
    for i in range(2, len(preds)-2):
        neighbors = []
        neighbors.append(preds[i-2])
        neighbors.append(preds[i-1])
        neighbors.append(preds[i+1])
        neighbors.append(preds[i+2])
        #print(f"prediction was: {preds[i]}, its neighbors are: {neighbors}}")
        neighbors_mode = mode(neighbors)
        if preds[i] == neighbors_mode:
            adjusted_preds.append(preds[i])
        else:
            neighbors_count = neighbors.count(neighbors_mode)
            if neighbors_count >= 3:
                adjusted_preds.append(neighbors_mode)
                #print(f"prediction was: {preds[i]}, its neighbors are: {neighbors}, adjusted to: {neighbors_mode}")
            else:
                adjusted_preds.append(preds[i])


    final_preds = adjusted_preds + list(preds[-2:])
    return np.array(final_preds)

def get_time_window_changes(adf):
    adf.sort_values('id',inplace=True)
    #df.rename(columns={'id': "Recording Window ID"}, inplace=True)
    time_counter = 0
    change_index = []
    for index, row in adf.iterrows():
        if row['time'] > time_counter:
            time_counter = row['time']
        else:
            time_counter = 0
            change_index.append(row["id"])
    return change_index, adf

def apply_time_window_group(adf):
    windows, adf = get_time_window_changes(adf)
    window_names = [(i, windows[i]) for i in range(len(windows))]
    max_id = None
    adf['TimeGroup'] = -1
    for win in window_names:
         adf.loc[(adf['id'] < win[1]) & (adf['TimeGroup'] == -1), 'TimeGroup'] = win[0]
         if max_id == None or win[1] > max_id:
             max_id = win[1]
    adf.loc[(adf["id"] >= max_id), 'TimeGroup'] = len(window_names)
    return adf


def refine_by_group(preds, test_df):
    df = test_df.copy()
    df['preds'] = preds
    groups = list(df['TimeGroup'].unique())
    for group in groups:
        all_group = df[df['TimeGroup'] == group]
        mode_group = mode(list(all_group['preds']))
        df.loc[(df['TimeGroup'] == group), 'preds'] = mode_group
    payload = np.array(df['preds'].values)
    return payload


df = apply_time_window_group(df)

train_users = df[(df['user'] != "U12")  & (df['user'] != "U11") & (df['user'] != "U2")]
test_users = df[(df['user'] == "U12") | (df['user'] == "11")| (df['user'] == "U2")]
test_size = round(len(test_users) / (len(train_users) + len(test_users)), 2)*100
print(f"The length of the full dataset is {len(df)}")
print(f"We split user 12, 9 & 2 from the dataset for testing")
print(f"Size of training dataframe: {len(train_users)}")
print(f"Size of test dataframe: {len(test_users)}")
print(f"The test size as a percentage is: {test_size}")

for i in range(1,13):
    day1 = df[(df['user'] == f"U{i}")]
    print(i, len(day1))


day1 = df[(df['user'] == "U12")]
day2 = df[df['user'] == "U11"]
day3 = df[df['user'] == "U2"]
day1.to_csv("Test_users/12day1.csv")
day2.to_csv("Test_users/10day2.csv")
day3.to_csv("Test_users/2day3.csv")
vc_models = [
        #("Decision Tree", DecisionTreeClassifier(random_state=909)),
        #("Extra Trees",ExtraTreesClassifier(random_state=909)),
        ("Random Forest",RandomForestClassifier(random_state=909)),
        #("AdaBoost",AdaBoostClassifier()),
        ("Skl GBM",GradientBoostingClassifier(random_state=909)),
        #("Skl HistGBM",HistGradientBoostingClassifier(random_state=909)),
        ("XGBoost",XGBClassifier(use_label_encoder=False)),
        #("CatBoost",CatBoostClassifier(verbose=0)),
        #"LightGBM":LGBMClassifier()),
        ]
tree_classifiers = {
        #"Decision Tree": DecisionTreeClassifier(random_state=909),
        # "Extra Trees":ExtraTreesClassifier(random_state=909),
        # "Random Forest":RandomForestClassifier(n_estimators=100,random_state=909),
        # "AdaBoost":AdaBoostClassifier(),
        # "Skl GBM":GradientBoostingClassifier(random_state=909),
        # "Skl HistGBM":HistGradientBoostingClassifier(random_state=909),
        # "XGBoost":XGBClassifier(use_label_encoder=True),
        # "LightGBM":LGBMClassifier(),
        #"CatBoost":CatBoostClassifier(verbose=0),
        "VotingClassifier": VotingClassifier(estimators=vc_models, voting='hard')
}
vars = [
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
x_train = train_users[vars]
x_val = test_users[vars]
y_train = train_users['target'].map({"Bus":2, "Car":2, "Still": 4, "Train":1, "Walking": 3})
y_val = test_users['target'].map({"Bus":2, "Car":2, "Still": 4, "Train":1, "Walking": 3})

##########################################################################################################################################

## Strategy for missing data - fill with 0
num_pipe_tree = pipeline.Pipeline(steps=[
            ('imputer', impute.SimpleImputer(strategy="constant", fill_value=0)),#fill_value=0
            #('Scaler',StandardScaler()),
])
tree_pipe = compose.ColumnTransformer(transformers=[
            #('cats', cat_pipe, cat_vars1),
            ('nums0', num_pipe_tree, vars)],
            #('numsM', num_pipe_nanM, num_vars_nanM)],
            remainder='drop'
            )
tree_pipes = {model_name: pipeline.make_pipeline(tree_pipe, model) for model_name, model in tree_classifiers.items()}

results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], "Adjusted Acc":[],"Adjusted Bal Acc": [],  "grouped Acc":[], "Grouped Bal Acc":[], 'Time': []})

print(x_train.isna().sum())
for model_name, model in tree_pipes.items():
    print(f"Working on: {model_name}")
    print(model)
    start_time = time.time()
    model.fit(x_train, y_train)
    if model_name == "VotingClassifier":
        pickle.dump(model,open("SavedModels/2best_model.pickle", 'wb'))
    pred = model.predict(x_val)
    end_time = time.time() - start_time
    adjusted_preds = consistency_adjust(pred)
    group_preds = refine_by_group(pred, test_users)
    results = results.append({
                          "Model":    model_name,
                          "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                          "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                          "Adjusted Acc": metrics.accuracy_score(y_val, adjusted_preds)*100,
                          "Adjusted Bal Acc": metrics.balanced_accuracy_score(y_val, adjusted_preds)*100,
                          "grouped Acc": metrics.accuracy_score(y_val, group_preds)*100,
                          "Grouped Bal Acc": metrics.balanced_accuracy_score(y_val, group_preds)*100,
                          "Time":     end_time},
                          ignore_index=True)

    pred_vs = pd.DataFrame({"Predictions": pred,
                            "Adjusted": adjusted_preds,
                            "grouped": group_preds,
                            "Actuals": y_val})
    pred_vs.to_csv("tests/"+model_name+"_4cl_vs.csv")
    print(confusion_matrix(y_val, pred))
    print(results)




def dump_timestamp_pickle(save_name, model):
    now_code = dt.datetime.now().strftime("%y%m%d%H%M")
    new_file_name = save_name + "_" + now_code + ".pickle"
    duplicate_counter = 1
    while os.path.exists(new_file_name):
        new_file_name = save_name + "_"  + now_code + f"_{str(duplicate_counter)}" + ".pickle"
        duplicate_counter += 1
    pickle.dump(model, new_file_name)
