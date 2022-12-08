import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
testset = pd.read_csv('./npf_test_hidden.csv')
test_X = testset.drop(['class4', 'date', 'id', 'partlybad'], axis=1)

train = pd.read_csv('./npf_train.csv')
y2 = np.where(train['class4'] == 'nonevent', 0, 1)
train['class4'] = train['class4'].replace('nonevent', 0)
train['class4'] = train['class4'].replace('Ia', 1)
train['class4'] = train['class4'].replace('Ib', 2)
train['class4'] = train['class4'].replace('II', 3)

y = train['class4']

X = train.drop(['class4', 'date', 'id', 'partlybad'], axis=1)

training_features, testing_features, training_target, testing_target = \
            train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

# Average CV score on the training set was: 0.9022774327122154
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    SelectFwe(score_func=f_classif, alpha=0.032),
    SelectPercentile(score_func=f_classif, percentile=92),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=7, min_samples_split=18, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# MULTICLASS CLASSIFIER
exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
preds = exported_pipeline.predict(test_X)
multiclass_accuracy = accuracy_score(testing_target, results)
print("MULTICLASS ACCURACY: ", multiclass_accuracy)

# BINARY CLASSIFIER
training_features, testing_features, training_target, testing_target = \
            train_test_split(X, y2, train_size=0.75, test_size=0.25, random_state=42)
exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
binary_accuracy = accuracy_score(testing_target, results)
print("BINARY ACCURACY: ", binary_accuracy)

probs = exported_pipeline.predict_proba(test_X)
classes=['nonevent', 'Ia', 'Ib', 'II']
answ = open('answer.csv', 'w')
answ.write("%s\n"%"0.879")
answ.write("%s"%"class4,p")
for p in range(len(probs)):
    answ.write("\n%s"% f"{classes[preds[p]]},{probs[p][1]}")
