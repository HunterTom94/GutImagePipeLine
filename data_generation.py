import numpy as np
import pandas as pd
from time import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sys import exit

np.set_printoptions(threshold=np.nan)


def trace_extraction(stack, labeled_neurons):
    start = time()
    # pixel_count = reduce(lambda x, y: x * y, list(raw_stack.shape[1:3]))
    neurons_count = int(np.amax(labeled_neurons))

    # ROI Index in excel starts from 1
    # neuron_raw_trace_dataframe = pd.DataFrame(0, index=np.arange(stack.shape[0]),
    #                                           columns=[str(int(x) + 1) for x in list(range(neurons_count))])
    # ROI Index in excel starts from 0
    neuron_raw_trace_dataframe = pd.DataFrame(0, index=np.arange(stack.shape[0]),
                                              columns=[str(int(x)) for x in list(range(neurons_count))])
    for x in range(neurons_count):
        mask = (labeled_neurons == x + 1).astype(int)
        masked_stack = stack * mask
        raw_trace = np.sum(masked_stack, (1, 2)) / np.sum(mask)
        neuron_raw_trace_dataframe.ix[:, x] = raw_trace
        print(str(time() - start) + '    ' + str(x+1) + '/' + str(neurons_count))

    return neuron_raw_trace_dataframe


def photo_bleach_correction(raw_stack):
    def exponential_fit(x, a, b):
        return a * np.exp(-b * x)

    def correction_formula(raw_value, b, t):
        corrected_value = raw_value / np.exp(-b * t)
        return corrected_value

    z_profile = np.mean(raw_stack, (1, 2))

    # z_profile = neuron_raw_trace_dataframe.ix[:, 0].values

    x_data = np.arange(1, raw_stack.shape[0] + 1)

    popt, _ = curve_fit(exponential_fit, x_data, z_profile, maxfev=100000)

    # corrected_z_profile = z_profile
    # for index, x in enumerate(z_profile):
    #     corrected_z_profile[index] = z_profile[index] / np.exp(-popt[1] * index)

    corrected_stack = raw_stack
    for t in x_data:
        corrected_stack[t - 1, :, :] = correction_formula(raw_stack[t - 1, :, :], popt[1], t)

    return corrected_stack


def background_subtraction(raw_stack, labeled_neurons, background_roi=[]):
    def subtract_background(raw_value, background):
        output = np.where(np.greater(raw_value, np.ones(raw_value.shape) * background), raw_value - background,
                          np.zeros(raw_value.shape))
        return output

    def divide_inverse_background(raw_stack, background):
        output = raw_stack
        background_inverse = background / np.amax(background)
        for t in range(raw_stack.shape[0]):
            output[t, :, :] = raw_stack[t, :, :] / background_inverse[t]
        return output

    if not background_roi:
        mask = (labeled_neurons == 0).astype(int)
    else:
        mask = np.zeros_like(labeled_neurons)
        mask[background_roi[0][0]:background_roi[1][0], background_roi[0][1]:background_roi[1][1]] = 1
    masked_stack = raw_stack * mask
    raw_background_trace = np.sum(masked_stack, (1, 2)) / np.sum(mask)
    background_trace_df = pd.DataFrame(raw_background_trace)
    plt.plot(raw_background_trace)
    plt.show()
    # exit()
    corrected_background_stack = raw_stack
    for t in range(raw_stack.shape[0]):
        corrected_background_stack[t, :, :] = subtract_background(raw_stack[t, :, :], raw_background_trace[t])

    # corrected_background_stack = divide_inverse_background(raw_stack, raw_background_trace)
    return corrected_background_stack, background_trace_df
