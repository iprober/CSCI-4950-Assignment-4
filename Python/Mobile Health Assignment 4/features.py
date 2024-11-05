# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks


def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

# TODO: define functions to compute more features


def _compute_standard_deviation(window):
    """
    Compute the standard deviation of the signal within the given window.

    Parameters:
        window (array-like): The signal window.

    Returns:
        std_dev (float): Standard deviation of the signal within the window.
    """
    std_dev = np.std(window)
    return std_dev


def _compute_dominant_frequency(window, sampling_rate):
    # Perform FFT on the windowed signal
    fft_result = np.fft.fft(window)

    # Compute the frequencies corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(window), 1 / sampling_rate)

    # Find the index of the maximum amplitude in the FFT result
    max_index = np.argmax(np.abs(fft_result))

    # Retrieve the dominant frequency from the frequencies array
    dominant_freq = frequencies[max_index]

    return abs(dominant_freq)


def _compute_entropy(window):
    """
    Compute the entropy of the signal within the given window.

    Parameters:
        window (array-like): The signal window.

    Returns:
        entropy (float): Entropy of the signal within the window.
    """
    # Compute the probability distribution of signal values
    unique_values, value_counts = np.unique(window, return_counts=True)
    probabilities = value_counts / len(window)

    # Compute entropy using the formula: H(X) = - sum(p(x) * log2(p(x)))
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def _compute_avg_peak_duration(window, sampling_rate):
    """
    Compute the average duration between peaks in the signal within the given window.

    Parameters:
        window (array-like): The signal window.
        sampling_rate (float): Sampling rate of the signal.

    Returns:
        avg_duration (float): Average duration between peaks in seconds.
    """
    # Find peaks in the signal
    peaks, _ = find_peaks(window)

    # Calculate time intervals between consecutive peaks
    time_intervals = np.diff(peaks) / sampling_rate

    # Compute the average duration between peaks
    avg_duration = np.mean(time_intervals)

    return avg_duration


def extract_features(window, sampling_rate):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    """
    Statistical
    These include the mean, variance and the rate of zero- or mean-crossings. The
    minimum and maximum may be useful, as might the median
    
    FFT features
    use rfft() to get Discrete Fourier Transform
    
    Entropy
    Integrating acceleration
    
    Peak Features:
    Sometimes the count or location of peaks or troughs in the accelerometer signal can be
    an indicator of the type of activity being performed. This is basically what you did in
    assignment A1 to detect steps. Use the peak count over each window as a feature. Or
    try something like the average duration between peaks in a window.
    """

    
    x = []
    feature_names = []
    win = np.array(window)
    x.append(_compute_mean_features(win[:,0]))
    feature_names.append("x_mean")

    x.append(_compute_mean_features(win[:,1]))
    feature_names.append("y_mean")

    x.append(_compute_mean_features(win[:,2]))
    feature_names.append("z_mean")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    # Compute standard deviation features - Statistics
    x.append(_compute_standard_deviation(win[:, 0]))
    feature_names.append("x_std_dev")

    x.append(_compute_standard_deviation(win[:, 1]))
    feature_names.append("y_std_dev")

    x.append(_compute_standard_deviation(win[:, 2]))
    feature_names.append("z_std_dev")

    # Compute dominant frequency - FFT
    x.append(_compute_dominant_frequency(win[:, 0], sampling_rate))
    feature_names.append("x_dominant_frequency")

    x.append(_compute_dominant_frequency(win[:, 1], sampling_rate))
    feature_names.append("y_dominant_frequency")

    x.append(_compute_dominant_frequency(win[:, 2], sampling_rate))
    feature_names.append("z_dominant_frequency")

    # Compute entropy feature - Other
    x.append(_compute_entropy(win[:, 0]))
    feature_names.append("x_entropy")

    x.append(_compute_entropy(win[:, 1]))
    feature_names.append("y_entropy")

    x.append(_compute_entropy(win[:, 2]))
    feature_names.append("z_entropy")

    # Compute average peak duration - Peak
    x.append(_compute_avg_peak_duration(win[:, 0], sampling_rate))
    feature_names.append("x_avg_peak_duration")

    x.append(_compute_avg_peak_duration(win[:, 1], sampling_rate))
    feature_names.append("y_avg_peak_duration")

    x.append(_compute_avg_peak_duration(win[:, 2], sampling_rate))
    feature_names.append("z_avg_peak_duration")

    feature_vector = list(x)
    return feature_names, feature_vector