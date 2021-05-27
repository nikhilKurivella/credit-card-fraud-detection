from collections import Counter

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

X = load("../Dataset_SMOTE_original_behavior/Sampled_Inputs.pck")
y = load("../Dataset_SMOTE_original_behavior/Sampled_Labels.pck")

X_test = load("../Dataset_SMOTE_original_behavior/Sampled_inputs_test.pck")
y_test = load("../Dataset_SMOTE_original_behavior/Sampled_labels_test.pck")
print(y_test.shape)
clf = RandomForestClassifier(n_estimators=50, random_state=22, max_features='sqrt')
rf = clf.fit(X, y)
valid_preds = rf.predict(X_test)
precision = metrics.precision_score(y_test, valid_preds, labels=[1])
recall = metrics.recall_score(y_test, valid_preds, labels=[1])
matrix = confusion_matrix(y_test, valid_preds)
print(matrix)
print(precision)
print(recall)