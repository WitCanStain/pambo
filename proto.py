import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split

npf_train = pd.read_csv('npf_train.csv')
X = npf_train.iloc[:, 3:]
# y = npf_train["class4"]
y = np.where(npf_train['class4']=='nonevent', 0, 1)
print('class4')
print(npf_train["class4"])
print('event')
print(y)
# print('X:')
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(
X, y, train_size=371, random_state=42, shuffle=True
)

clf = linear_model.LogisticRegression(penalty="l1", C=2, solver="saga", max_iter=1000000)

clf.fit(X_train, y_train)
phat = clf.predict_proba(X_train)[:,1]

# print(X_train)


print('accuracy:')
print(clf.score(X_train,y_train))

n = len(X_train)
log_phat = [np.log(x) for x in phat]
print('perplexity:')
print(np.exp(-1 * (np.sum(log_phat))/n))

