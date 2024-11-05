# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
import sklearn
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import labels


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'data/all_labeled_data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,2], data[i,3], data[i,4]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:2],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

data = np.nan_to_num(data)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 100 Hz (sensor logger app); you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,1] - data[0,1])
sampling_rate = n_samples / time_elapsed_seconds

print("Sampling Rate: " + str(sampling_rate))

# TODO: list the class labels that you collected data for in the order of label_index (defined in labels.py)
class_names = labels.activity_labels

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []
feature_names = []
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1]
    # print("window = ")
    # print(window)
    feature_names, x = extract_features(window, sampling_rate)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)

"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
# Initialize lists to store evaluation metrics for each fold
accuracies = []
precisions = []
recalls = []

# Initialize the stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Iterate over each fold
for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    print(f"Processing Fold {fold + 1}...")
    sys.stdout.flush()

    # Split training data into train and validation sets for this fold
    X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

    # Initialize and fit the decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train_fold, y_train_fold)

    # Predict labels for validation set
    y_pred = classifier.predict(X_val_fold)

    # Calculate evaluation metrics for this fold
    accuracy = accuracy_score(y_val_fold, y_pred)
    precision = precision_score(y_val_fold, y_pred, average='weighted')
    recall = recall_score(y_val_fold, y_pred, average='weighted')

    # Append evaluation metrics to lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

    # Compute confusion matrix
    cm = confusion_matrix(y_val_fold, y_pred)
    print(f"Confusion Matrix for Fold {fold + 1}:\n{cm}")

    # Print evaluation metrics for this fold
    print(f"Accuracy for Fold {fold + 1}: {accuracy:.2f}")
    print(f"Precision for Fold {fold + 1}: {precision:.2f}")
    print(f"Recall for Fold {fold + 1}: {recall:.2f}")
    print("\n")

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
print("Average Accuracy:", avg_accuracy)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)

# TODO: train the decision tree classifier on entire dataset
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
# export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)
export_graphviz(classifier, out_file='tree.dot', feature_names=feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
# print("saving classifier model...")
# with open('classifier.pickle', 'wb') as f:
#     pickle.dump(tree, f)
print("Saving classifier model...")
with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)
