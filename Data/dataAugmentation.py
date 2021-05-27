import pickle
from collections import Counter
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE

from matplotlib import pyplot
from numpy import where
from sklearn.model_selection import train_test_split


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

X = load("../Dataset/Sampled_inputs.pck")
y = load("../Dataset/Sampled_labels.pck")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
counter = Counter(y_test)
print(counter)
oversample = SMOTE(random_state=22, sampling_strategy=0.5)
X, y = oversample.fit_resample(X_train, y_train)
counter = Counter(y)
print(counter)
print(X.shape)
print(y.shape)
save(X, "../Dataset_SMOTE_original_behavior/Sampled_inputs.pck")
save(y, "../Dataset_SMOTE_original_behavior/Sampled_labels.pck")
save(X_test, "../Dataset_SMOTE_original_behavior/Sampled_inputs_test.pck")
save(y_test, "../Dataset_SMOTE_original_behavior/Sampled_labels_test.pck")