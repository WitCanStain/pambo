from tpot import TPOTClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv('./npf_train.csv')
train['class4'] = train['class4'].replace('nonevent', 0)
train['class4'] = train['class4'].replace('Ia', 1)
train['class4'] = train['class4'].replace('Ib', 2)
train['class4'] = train['class4'].replace('II', 3)

y = train['class4']

X = train.drop(['class4', 'date', 'id', 'partlybad'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, cv=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline3.py')
