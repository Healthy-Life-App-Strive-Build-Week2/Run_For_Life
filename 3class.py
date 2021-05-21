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



#### PURPOSE OF MODEL: To predict whether a user is walking, sitting still or in transportation from phone sensors
df = pd.read_csv("dataset_halfSecondWindow.csv", index_col='id')
train_users = df[(df['user'] != "U12")  & (df['user'] != "U9") & (df['user'] != "U2")]
test_users = df[(df['user'] == "U12") | (df['user'] == "U9")| (df['user'] == "U2")]
test_size = round(len(test_users) / (len(train_users) + len(test_users)), 2)*100
print(f"The length of the full dataset is {len(df)}")
print(f"We split user 12, 9 & 2 from the dataset for testing")
print(f"Size of training dataframe: {len(train_users)}")
print(f"Size of test dataframe: {len(test_users)}")
print(f"The test size as a percentage is: {test_size}")

# The 3 test users
day1 = df[(df['user'] == "U12")]
day2 = df[df['user'] == "U9"]
day3 = df[df['user'] == "U2"]
day1.to_csv("Day1.csv")
day2.to_csv("Day2.csv")
day3.to_csv("Day3.csv")
print(f"the size of the test users are:")
print(f"U12 = {len(day1)}, U9 = {len(day2)}, U2= {len(day3)}")


vc_models = [
         #("Decision Tree", DecisionTreeClassifier(random_state=909)),
        ("Extra Trees",ExtraTreesClassifier(random_state=909)),
        #("Random Forest",RandomForestClassifier(random_state=909)),
        ("AdaBoost",AdaBoostClassifier()),
        ("Skl GBM",GradientBoostingClassifier(random_state=909)),
        #("Skl HistGBM",HistGradientBoostingClassifier(random_state=909)),
        #("XGBoost",XGBClassifier(use_label_encoder=False)),
        #"LightGBM":LGBMClassifier()),
        #("CatBoost",CatBoostClassifierCorrected(verbose=0))
        ]
tree_classifiers = {
        #"Decision Tree": DecisionTreeClassifier(random_state=909),
        "Extra Trees":ExtraTreesClassifier(random_state=909),
        #"Random Forest":RandomForestClassifier(n_estimators=100,random_state=909),
        "AdaBoost":AdaBoostClassifier(),
        "Skl GBM":GradientBoostingClassifier(random_state=909),
        #"Skl HistGBM":HistGradientBoostingClassifier(random_state=909),
        #"XGBoost":XGBClassifier(use_label_encoder=True),
        # #"LightGBM":LGBMClassifier(),
        #"CatBoost":CatBoostClassifier(verbose=0),
        # "CatBoost":CatBoostClassifierCorrected(verbose=0),
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

#### MAP TRANSPORTATION INTO 1 CATEGORY FOR TEST & TRAIN GROUPS
x_train = train_users[vars]
x_val = test_users[vars]
y_train = train_users['target'].map({"Bus":1, "Car":1, "Still": 3, "Train":1, "Walking":2 })
y_val = test_users['target'].map({"Bus":1, "Car":1, "Still": 3, "Train":1, "Walking": 2})


##########################################################################################################################################
#### FOR FILLING WITH MEAN
def fill_nan_with_mean_training(training, test):
    trainingFill = training.copy()
    testFill = test.copy()
    trainingFill = trainingFill.fillna(trainingFill.mean())
    trainingFill = trainingFill.fillna(0)
    testFill = testFill.fillna(trainingFill.mean())
    testFill = testFill.fillna(0)
    return trainingFill, testFill


x_train, x_val = fill_nan_with_mean_training(x_train, x_val)

##########################################################################################################################################
def consistency_adjust(preds):
    adjusted_preds = list(preds[0:1])
    for i in range(1, len(preds)-1):
        matching_neighbour_quantity = []
        if preds[i] != preds[i-1]:
            matching_neighbour_quantity.append(preds[i-1])
        if preds[i] != preds[i+1]:
            matching_neighbour_quantity.append(preds[i+1])
        print(f"prediction was: {preds[i]}, different neighbors are: {matching_neighbour_quantity}")
        if len(matching_neighbour_quantity) == 2:
            possibles = set(matching_neighbour_quantity)
            if len(possibles) == 1:
                adjusted_preds.append(matching_neighbour_quantity[0])
                print(f"pred was: {preds[i]}, submitted: {matching_neighbour_quantity[0]}")
            else:
                adjusted_preds.append(preds[i])
                print(f"pred was: {preds[i]},submitted: {preds[i]}")
        else:
            adjusted_preds.append(preds[i])
            print(f"pred was: {preds[i]},submitted: {preds[i]}")

    final_preds = adjusted_preds + list(preds[-1:])
    return np.array(final_preds)
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

results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], "Adjusted Acc":[],"Adjusted Bal Acc": [],'Time': []})

print(x_train.isna().sum())
for model_name, model in tree_pipes.items():
    print(f"Working on: {model_name}")
    print(model)
    start_time = time.time()
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    end_time = time.time() - start_time
    adjusted_preds = consistency_adjust(pred)
    results = results.append({
                          "Model":    model_name,
                          "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                          "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                          "Adjusted Acc": metrics.accuracy_score(y_val, adjusted_preds)*100,
                          "Adjusted Bal Acc": metrics.balanced_accuracy_score(y_val, adjusted_preds)*100,
                          "Time":     end_time},
                          ignore_index=True)

    pred_vs = pd.DataFrame({"Predictions": pred,
                            "Actuals": y_val})
    pred_vs.to_csv(model_name+"_3cl_vs.csv")
    print(confusion_matrix(y_val, pred))
    print(results)
