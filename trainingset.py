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


def dump_timestamp_csv(save_name, dataframe):
    now_code = dt.datetime.now().strftime("%y%m%d%H%M")
    new_file_name = save_name + "_" + now_code + ".csv"
    duplicate_counter = 1
    while os.path.exists(new_file_name):
        new_file_name = save_name + "_"  + now_code + f"_{str(duplicate_counter)}" + ".csv"
        duplicate_counter += 1
    dataframe.to_csv(new_file_name)

tdf = pd.read_csv("training_balanced.csv")
df = pd.read_csv("dataset_halfSecondWindow.csv", index_col='id')
print(len(df))
print(len(tdf))
no_c = tdf[tdf['target']!= "Car"]

# train_users = df[(df['user'] != "U12")  & (df['user'] != "U9") & (df['user'] != "U2")]
# test_user = df[(df['user'] == "U12") | (df['user'] == "U9")| (df['user'] == "U2")]
train_users = tdf[(tdf['user'] != "U12")]# & (tdf['user'] != "U9") & (tdf['user'] != "U2")]
test_user = tdf[(tdf['user'] == "U12")]# | (tdf['user'] == "U9")| (tdf['user'] == "U2")]


#print(f"train users on tdf{len(train_users)}")
# train_users = no_c[(no_c['user'] != "U12")]# & (no_c['user'] != "U9") & (no_c['user'] != "U2")]
# test_user = no_c[(no_c['user'] == "U12")]# | (no_c['user'] == "U9")| (no_c['user'] == "U2")]

print(f"train users on tdf{len(train_users)}")
print(f"test users on no_c{len(train_users)}")
# print(train_users['target'].unique())
# print(test_user['target'].unique())
# print(f"TRAIN USERS {len(train_users)}")
# print(f"TEST USERS {len(test_user)}")
#

# na_count = train_users.isna().sum()
# na_count.to_csv("na_counts.csv")
#
# for i in range(13):
#     x = df[df['user'] == f"U{i}"]
#     print(i, len(x))
#

### make a pipeline using simple imputer
### make a pipeline using real data only
### make a pipeline trying different features
################################################################################
###
mult_classifiers = {
        #"LM Linear Regression": LinearRegression(), # not useful for classification on titanic
        "LM Logistic Regression": LogisticRegression(),
        "SGDC": SGDClassifier(),
        "LM Ridge": RidgeClassifier(),
        #"LM Lasso": Lasso(),
        "NN Multi layer Perceptron": MLPClassifier(random_state=909),
        "SVM Linear": svm.SVC(kernel='linear'),
        "SVM RBF": svm.SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(),
        "BM Guassian Naive Bayes": GaussianNB(),
}


vc_models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=909)),
        #("Extra Trees"ExtraTreesClassifier(random_state=909)),
        #("Random Forest",RandomForestClassifier(random_state=909)),
        #("AdaBoost",AdaBoostClassifier()),
        ("Skl GBM",GradientBoostingClassifier(random_state=909)),
        ("Skl HistGBM",HistGradientBoostingClassifier(random_state=909)),
        #("XGBoost",XGBClassifier(use_label_encoder=False)),
        #"LightGBM":LGBMClassifier()),
        ("CatBoost",CatBoostClassifier(verbose=0))
        #"VotingClassifier": VotingClassifier(estimators=vc_models, voting='hard')
        ]
tree_classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=909),
        "Extra Trees":ExtraTreesClassifier(random_state=909),
        "Random Forest":RandomForestClassifier(n_estimators=100,random_state=909),
        "AdaBoost":AdaBoostClassifier(),
        "Skl GBM":GradientBoostingClassifier(random_state=909),
        "Skl HistGBM":HistGradientBoostingClassifier(random_state=909),
        "XGBoost":XGBClassifier(use_label_encoder=True),
        #"LightGBM":LGBMClassifier(),
        "CatBoost":CatBoostClassifier(verbose=0),
        #"VotingClassifier": VotingClassifier(estimators=vc_models, voting='hard')
}


################################################################################
### SPLIT CAT / NUM DATA
cat_vars1 = ['user']

num_vars_nan0 = [
            "time",
            # "activityrecognition#0",
            "activityrecognition#1",
            "android.sensor.accelerometer#mean",
            "android.sensor.accelerometer#min",
            "android.sensor.accelerometer#max",
            "android.sensor.accelerometer#std",

            # "android.sensor.game_rotation_vector#mean",
            # "android.sensor.game_rotation_vector#min",
            # "android.sensor.game_rotation_vector#max",
            # "android.sensor.game_rotation_vector#std",

            # "android.sensor.gravity#mean",
            # "android.sensor.gravity#min",
            # "android.sensor.gravity#max",
            # "android.sensor.gravity#std",

            "android.sensor.gyroscope#mean",
            "android.sensor.gyroscope#min",
            "android.sensor.gyroscope#max",
            "android.sensor.gyroscope#std",

            "android.sensor.gyroscope_uncalibrated#mean",
            "android.sensor.gyroscope_uncalibrated#min",
            "android.sensor.gyroscope_uncalibrated#max",
            "android.sensor.gyroscope_uncalibrated#std",

            # "android.sensor.light#mean",
            # "android.sensor.light#min",
            # "android.sensor.light#max",
            # "android.sensor.light#std",

            "android.sensor.linear_acceleration#mean",
            "android.sensor.linear_acceleration#min",
            "android.sensor.linear_acceleration#max",
            "android.sensor.linear_acceleration#std",

            # "android.sensor.magnetic_field#mean",
            # "android.sensor.magnetic_field#min",
            # "android.sensor.magnetic_field#max",
            # "android.sensor.magnetic_field#std",

            # "android.sensor.magnetic_field_uncalibrated#mean",
            # "android.sensor.magnetic_field_uncalibrated#min",
            # "android.sensor.magnetic_field_uncalibrated#max",
            # "android.sensor.magnetic_field_uncalibrated#std",

            # "android.sensor.orientation#mean",
            # "android.sensor.orientation#min",
            # "android.sensor.orientation#max",
            # "android.sensor.orientation#std",

            "android.sensor.pressure#mean",
            "android.sensor.pressure#min",
            "android.sensor.pressure#max",
            "android.sensor.pressure#std",

            # "android.sensor.proximity#mean",
            # "android.sensor.proximity#min",
            # "android.sensor.proximity#max",
            # "android.sensor.proximity#std",

            # "android.sensor.rotation_vector#mean",
            # "android.sensor.rotation_vector#min",
            # "android.sensor.rotation_vector#max",
            # "android.sensor.rotation_vector#std",

            "android.sensor.step_counter#mean",
            "android.sensor.step_counter#min",
            "android.sensor.step_counter#max",
            "android.sensor.step_counter#std",
            # "sound#mean",
            # "sound#min",
            # "sound#max",
            # "sound#std",
            #
            "speed#mean",
            "speed#min",
            "speed#max",
            "speed#std",
            #"target",
            #"user"
            ]
num_vars_cat74 = [
            "time",
            # "activityrecognition#0",
            "activityrecognition#1",
            "android.sensor.accelerometer#mean",
            "android.sensor.accelerometer#min",
            #"android.sensor.accelerometer#max",
            "android.sensor.accelerometer#std",

            # "android.sensor.game_rotation_vector#mean",
            # "android.sensor.game_rotation_vector#min",
            # "android.sensor.game_rotation_vector#max",
            # "android.sensor.game_rotation_vector#std",

            # "android.sensor.gravity#mean",
            # "android.sensor.gravity#min",
            # "android.sensor.gravity#max",
            # "android.sensor.gravity#std",

            # "android.sensor.gyroscope#mean",
            # "android.sensor.gyroscope#min",
            # "android.sensor.gyroscope#max",
            # "android.sensor.gyroscope#std",
            #
            "android.sensor.gyroscope_uncalibrated#mean",
            "android.sensor.gyroscope_uncalibrated#min",
            "android.sensor.gyroscope_uncalibrated#max",
            "android.sensor.gyroscope_uncalibrated#std",

            # "android.sensor.light#mean",
            # "android.sensor.light#min",
            # "android.sensor.light#max",
            # "android.sensor.light#std",
            #
            "android.sensor.linear_acceleration#mean",
            "android.sensor.linear_acceleration#min",
            "android.sensor.linear_acceleration#max",
            "android.sensor.linear_acceleration#std",

            # "android.sensor.magnetic_field#mean",
            # "android.sensor.magnetic_field#min",
            # "android.sensor.magnetic_field#max",
            # "android.sensor.magnetic_field#std",

            # "android.sensor.magnetic_field_uncalibrated#mean",
            # "android.sensor.magnetic_field_uncalibrated#min",
            # "android.sensor.magnetic_field_uncalibrated#max",
            # "android.sensor.magnetic_field_uncalibrated#std",

            # "android.sensor.orientation#mean",
            # "android.sensor.orientation#min",
            # "android.sensor.orientation#max",
            # "android.sensor.orientation#std",
            #
            "android.sensor.pressure#mean",
            # "android.sensor.pressure#min",
            "android.sensor.pressure#max",
            "android.sensor.pressure#std",

            # "android.sensor.proximity#mean",
            # "android.sensor.proximity#min",
            # "android.sensor.proximity#max",
            # "android.sensor.proximity#std",

            # "android.sensor.rotation_vector#mean",
            # "android.sensor.rotation_vector#min",
            # "android.sensor.rotation_vector#max",
            # "android.sensor.rotation_vector#std",
            #
            "android.sensor.step_counter#mean",
            "android.sensor.step_counter#min",
            "android.sensor.step_counter#max",
            "android.sensor.step_counter#std",
            "sound#mean",
            #"sound#min",
            # "sound#max",
            # "sound#std",
            #
            "speed#mean",
            "speed#min",
            "speed#max",
            "speed#std",
            #"target",
            #"user"
            ]
num_vars_nanM = [
            # "time",
            #"activityrecognition#0",
            # "activityrecognition#1",
            # "android.sensor.accelerometer#mean",
            # "android.sensor.accelerometer#min",
            # "android.sensor.accelerometer#max",
            # "android.sensor.accelerometer#std",

            # "android.sensor.game_rotation_vector#mean",
            # "android.sensor.game_rotation_vector#min",
            # "android.sensor.game_rotation_vector#max",
            # "android.sensor.game_rotation_vector#std",

            # "android.sensor.gravity#mean",
            # "android.sensor.gravity#min",
            # "android.sensor.gravity#max",
            # "android.sensor.gravity#std",
            #
            # "android.sensor.gyroscope#mean",
            # "android.sensor.gyroscope#min",
            # "android.sensor.gyroscope#max",
            # "android.sensor.gyroscope#std",
            #
            # "android.sensor.gyroscope_uncalibrated#mean",
            # "android.sensor.gyroscope_uncalibrated#min",
            # "android.sensor.gyroscope_uncalibrated#max",
            # "android.sensor.gyroscope_uncalibrated#std",

            # "android.sensor.light#mean",
            # "android.sensor.light#min",
            # "android.sensor.light#max",
            # "android.sensor.light#std",

            # "android.sensor.linear_acceleration#mean",
            # "android.sensor.linear_acceleration#min",
            # "android.sensor.linear_acceleration#max",
            # "android.sensor.linear_acceleration#std",

            # "android.sensor.magnetic_field#mean",
            # "android.sensor.magnetic_field#min",
            # "android.sensor.magnetic_field#max",
            # "android.sensor.magnetic_field#std",
            #
            # "android.sensor.magnetic_field_uncalibrated#mean",
            # "android.sensor.magnetic_field_uncalibrated#min",
            # "android.sensor.magnetic_field_uncalibrated#max",
            # "android.sensor.magnetic_field_uncalibrated#std",
            #
            # "android.sensor.orientation#mean",
            # "android.sensor.orientation#min",
            # "android.sensor.orientation#max",
            # "android.sensor.orientation#std",

            # "android.sensor.pressure#mean",
            # "android.sensor.pressure#min",
            # "android.sensor.pressure#max",
            # "android.sensor.pressure#std",
            #
            # "android.sensor.proximity#mean",
            # "android.sensor.proximity#min",
            # "android.sensor.proximity#max",
            # "android.sensor.proximity#std",

            # "android.sensor.rotation_vector#mean",
            # "android.sensor.rotation_vector#min",
            # "android.sensor.rotation_vector#max",
            # "android.sensor.rotation_vector#std",
            #
            # "android.sensor.step_counter#mean",
            # "android.sensor.step_counter#min",
            # "android.sensor.step_counter#max",
            # "android.sensor.step_counter#std",
            # "sound#mean",
            # "sound#min",
            # "sound#max",
            #"sound#std",

            # "speed#mean",
            # "speed#min",
            # "speed#max",
            # "speed#std",
            #"target",
            #"user"
            ]

cat_vars2 = ['user']
num_vars_nan9 =[
    "android.sensor.rotation_vector#min",
    "android.sensor.linear_acceleration#max",
    "android.sensor.linear_acceleration#min",
    "android.sensor.game_rotation_vector#std",
    "android.sensor.linear_acceleration#std",
    "speed#std",
    "android.sensor.accelerometer#max",
    "android.sensor.gyroscope#min",
    "android.sensor.accelerometer#std",
    "android.sensor.orientation#min",
    "android.sensor.linear_acceleration#mean",
    "android.sensor.orientation#std",
    "speed#mean",
    "speed#max",
    "android.sensor.game_rotation_vector#mean",
    "android.sensor.accelerometer#mean",
    "android.sensor.gyroscope#max",
    "sound#mean",
    "android.sensor.gyroscope#std",
    "speed#min",
    "android.sensor.orientation#mean",
    "android.sensor.gyroscope#mean",
    "android.sensor.rotation_vector#mean",
    "android.sensor.rotation_vector#std",
    "android.sensor.gyroscope_uncalibrated#std",
    "android.sensor.gyroscope_uncalibrated#max",
    "android.sensor.rotation_vector#max",
    "android.sensor.gyroscope_uncalibrated#min",
    "android.sensor.orientation#max",
    "android.sensor.gyroscope_uncalibrated#mean",
    "sound#max",
    "sound#min",
    "android.sensor.game_rotation_vector#min",
    "android.sensor.accelerometer#min",
    "android.sensor.game_rotation_vector#max",
    "sound#std",
]

x_train = train_users[num_vars_nan9]
x_val = test_user[num_vars_nan9]
y_train = train_users['target'].map({"Bus":1, "Car":2, "Still": 3, "Train":4, "Walking": 5})
y_val = test_user['target'].map({"Bus":1, "Car":2, "Still": 3, "Train":4, "Walking": 5})
print("YTRAININNNGNGNEFJRIFHUH")
print(y_train.unique())
print(x_train.isna().sum())
def fill_nan_with_mean_training(training, test):
    trainingFill = training.copy()
    testFill = test.copy()
    trainingFill = trainingFill.fillna(trainingFill.mean())
    trainingFill = trainingFill.fillna(0)
    testFill = testFill.fillna(trainingFill.mean())
    testFill = testFill.fillna(0)
    return trainingFill, testFill


#x_train, x_val = fill_nan_with_mean_training(x_train, x_val)

print(x_train.isna().sum())
################################################################################
### CREATE PIPELINES

cat_pipe = pipeline.Pipeline(steps=[
            ("OneHot", OrdinalEncoder(handle_unknown='mean'))
])

num_pipe_tree = pipeline.Pipeline(steps=[
            ('imputer', impute.SimpleImputer(strategy="constant", fill_value=0)),#fill_value=0
            #('Scaler',StandardScaler()),
])
num_pipe_mult = pipeline.Pipeline(steps=[
            ('imputer', impute.SimpleImputer(strategy="constant", fill_value=0)),#fill_value=0
            ('Scaler',StandardScaler()),
])


tree_pipe = compose.ColumnTransformer(transformers=[
            #('cats', cat_pipe, cat_vars1),
            ('nums0', num_pipe_tree, num_vars_nan9)],
            #('numsM', num_pipe_nanM, num_vars_nanM)],
            remainder='drop'
            )

mult_pipe = compose.ColumnTransformer(transformers=[
            #('cats', cat_pipe, cat_vars1),
            ('nums0', num_pipe_mult, num_vars_nan9)],
            #('numsM', num_pipe_nanM, num_vars_nanM)],
            remainder='drop')
tree_pipes = {model_name: pipeline.make_pipeline(tree_pipe, model) for model_name, model in tree_classifiers.items()}
mult_pipes = {model_name: pipeline.make_pipeline(mult_pipe, model) for model_name, model in mult_classifiers.items()}

all_pipes = {**tree_pipes, **mult_pipes}
################################################################################
###


results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})
for model_name, model in all_pipes.items():
    print(f"Working on: {model_name}")
    print(model)
    start_time = time.time()
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    end_time = time.time() - start_time

    results = results.append({
                          "Model":    model_name,
                          "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                          "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                          "Time":     end_time},
                          ignore_index=True)
    print(results)
    print(confusion_matrix(y_val, pred))
dump_timestamp_csv("resultsT", results)
for a in num_vars_nan9:
    print(a)
################################################################################
### VOTING CLASSIFIER

# models = [(key, value) for key,value in tree_classifiers.items()]
# trained_models = []
# eclf = pipeline.make_pipeline(main_pipe, VotingClassifier(estimators=models, voting='hard'))
# eclf.fit(x_train, y_train)
#
# vc_pred = eclf.predict(x_val)
# vc_results = {
#             "Accuracy": metrics.accuracy_score(y_val, vc_pred)*100,
#             "Bal Acc.": metrics.balanced_accuracy_score(y_val, vc_pred)*100,}
#
# print(vc_results)
################################################################################
###
