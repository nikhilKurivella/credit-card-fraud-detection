from collections import Counter

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv("../creditcard.csv")
labels = data["Class"].tolist()
data = data.drop(['Class'], axis=1)
inputs = data.values.tolist()

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)


inputs = np.asarray(inputs, dtype=float)
labels = np.asarray(labels, dtype=int)

X = load("../Dataset_SMOTE/Sampled_Inputs.pck")
y = load("../Dataset_SMOTE/Sampled_Labels.pck")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=22, max_features='sqrt')
rf = clf.fit(X_train, y_train)
valid_preds = rf.predict(X_test)
precision = metrics.precision_score(y_test, valid_preds)
recall = metrics.recall_score(y_test, valid_preds)
matrix = confusion_matrix(y_test, valid_preds)
print(matrix)
print(precision)
print(recall)