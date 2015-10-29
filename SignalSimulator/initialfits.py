import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, svm, preprocessing, cross_validation
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
SNRlim = 10

planetsall = pd.read_csv("planetssmall.csv")
pSNR = planetsall['SNR']
pAMP = planetsall['A']
signals1 = planetsall[features]
signals1['Label'] = 1
signals1[pSNR <= SNRlim] = 0

nullall = pd.read_csv("nullsignalssmall.csv")
signals2 = nullall[features]
signals2['Label'] = 0


binariesall = pd.read_csv("binariessmall.csv")
bSNR = binariesall['SNR']
bAMP = binariesall['A']
signals3 = binariesall[features]
signals3['Label'] = 2
signals3[bSNR <= SNRlim] = 0


sinuall = pd.read_csv("sinusmall.csv")
sSNR = sinuall['SNR']
sAMP = sinuall['A']
signals4 = sinuall[features]
signals4['Label'] = 3
signals4[sSNR < SNRlim] = 0


signals = pd.concat([signals1, signals2, signals3, signals4])


X = signals[features].values  # we only take the first two features.
Y = signals['Label'].values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.7, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)

svmmodel= svm.SVC(kernel="linear", C=1)#linear_model.LogisticRegression(C=1e5)

scores = cross_validation.cross_val_score(svmmodel, X_train_transformed, y_train, cv=10,  scoring='f1_weighted')
print("Accuracy: %0.4f (+/- %0.4f) for C = 1" % (scores.mean(), scores.std() * 2))

estimator = svm.SVC(kernel='linear')
Cs = np.logspace(0, 4.0, 10)
classifier = GridSearchCV(estimator=estimator, cv=10, param_grid={'C':Cs})
classifier.fit(X_train_transformed, y_train)
y_pred = classifier.predict(X_train_transformed)
print confusion_matrix(y_train, y_pred)

X_test_transformed = scaler.transform(X_test)
y_test_pred = classifier.predict(X_test_transformed)
print confusion_matrix(y_test, y_test_pred)
print classifier.score(X_test_transformed, y_test)

