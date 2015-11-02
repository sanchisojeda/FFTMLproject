import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, svm, preprocessing, cross_validation
import pandas as pd
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


X = signals[features].values 
Y = signals['Label'].values

X_fulltrain, X_test, y_fulltrain, y_test = cross_validation.train_test_split(X, Y, test_size=0.25, random_state=0)
X_train, X_CV, y_train, y_CV = cross_validation.train_test_split(X_fulltrain, y_fulltrain, test_size=0.25, random_state=0)

scalerall = preprocessing.StandardScaler().fit(X_fulltrain)
X_fulltrain_transformed = scalerall.transform(X_fulltrain)

svmmodel= svm.SVC(kernel="rbf", gamma=0, C =1)

scores = cross_validation.cross_val_score(svmmodel, X_fulltrain_transformed, y_fulltrain, cv=5,  scoring='f1_weighted')
print("Accuracy: %0.4f (+/- %0.4f) for C = 1" % (scores.mean(), scores.std() * 2))


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_CV_transformed = scaler.transform(X_CV)

gammas = np.logspace(-2, 3.0, 10)
gammas[0] = 0
allscores = np.zeros(len(gammas))
for i, gamma in enumerate(gammas):
	estimator= svm.SVC(kernel="rbf", gamma=gamma, C=1000.0)
	value = estimator.fit(X_train_transformed, y_train)
	allscores[i] = estimator.score(X_CV_transformed, y_CV)
	print gamma, allscores[i]

chosen = gammas[np.argmax(allscores)]

svmmodel= svm.SVC(kernel="rbf", gamma =chosen, C=1000.0)
value = svmmodel.fit(X_fulltrain_transformed, y_fulltrain)

y_pred = svmmodel.predict(X_fulltrain_transformed)
print confusion_matrix(y_fulltrain, y_pred)

X_test_transformed = scalerall.transform(X_test)
y_test_pred = svmmodel.predict(X_test_transformed)
print confusion_matrix(y_test, y_test_pred)
print svmmodel.score(X_test_transformed, y_test)

