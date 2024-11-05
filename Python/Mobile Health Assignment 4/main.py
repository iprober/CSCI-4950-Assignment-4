import sys
import numpy as np
import matplotlib.pyplot as plt
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle

# Load the trained model from disk
with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

# Load the data
print("Loading data...")
sys.stdout.flush()
data_file = 'data/val/all_labeled_val_data'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# Pre-processing
print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i, 2], data[i, 3], data[i, 4]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:, 0:2], reoriented, axis=1)
data = np.append(reoriented_data_with_timestamps, data[:, -1:], axis=1)
data = np.nan_to_num(data)

# Extract Features & Labels
window_size = 20
step_size = 20

n_samples = 1000
time_elapsed_seconds = (data[n_samples, 1] - data[0, 1])
sampling_rate = n_samples / time_elapsed_seconds

print("Sampling Rate: " + str(sampling_rate))

# Extracting features
X_new = []
feature_names = []
for i, window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:, 2:-1]
    feature_names, x = extract_features(window, sampling_rate)
    X_new.append(x)

X_new = np.asarray(X_new)

# Use the trained model to make predictions on the new data
predictions = classifier.predict(X_new)

# Plot prediction
plt.figure(figsize=(10, 6))

# Start/stop index - goes up to 2000
start_index = 0
end_index = 1400

# View portion
# plt.plot(predictions[start_index:end_index], label='Predicted Values', color='blue')
plt.plot(range(start_index, end_index), predictions[start_index:end_index], label='Predicted Values', color='blue')

# View entire prediction
# plt.plot(predictions, label='Predicted Values', color='blue')

plt.xlabel('Time (seconds)')
plt.ylabel('Activity Label')
plt.title('Predicted Activity Labels (First Minute)')
plt.legend()
plt.grid(True)
plt.show()
