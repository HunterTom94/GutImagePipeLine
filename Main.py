from sys import exit
import cv2
import numpy as np
from skimage import io, img_as_ubyte
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import sobel
import pandas as pd
from time import time
from util import calculate_delta_f_over_f, write_dataframe_to_excel, get_roi_at_certain_time, trace_clustering, \
    plot_trace
from data_generation import trace_extraction, photo_bleach_correction, background_subtraction

from ROI_identification import find_roi, plot_neuron_label_position, find_roi_slices, erase_ROI

import matplotlib.pyplot as plt


def data_import(source_folder, video_name, sampling_rate):
    assert (sampling_rate >= 1), "Sampling Rate Lower than 1 Hz"
    stack_filepath = source_folder + video_name
    print(stack_filepath)
    video_index = video_name.split('.')[0]
    print(video_index)
    raw_stack = io.imread(stack_filepath)
    # Cut first 1 second video
    raw_stack = raw_stack[sampling_rate:, :, :]
    # Downsample by a Factor of 10
    downsampled_stack = raw_stack[::sampling_rate, :, :].astype('uint16')
    return downsampled_stack, video_index


def find_roi_execution(raw_stack, lower_threshold, upper_threshold, roi_min_pixel_size=[],specify_frame=None):
    # ROI Identification
    if specify_frame:
        image = raw_stack[specify_frame,:,:]
    else:
        image = np.max(raw_stack, axis=0)  # Max Z Projection
    start = time()

    plt.imshow(image, aspect="auto")
    plt.show()

    labeled_neurons, labeled_neurons_df = find_roi(image, lower_threshold,
                                                   upper_threshold, roi_min_pixel_size)
    print(time() - start)
    plot_neuron_label_position(labeled_neurons, labeled_neurons_df, output_folder, image, video_index)

    return labeled_neurons


def data_generation(raw_stack, labeled_neurons, output_folder, video_index, sampling_rate, background_roi=[]):
    # Set Correction Method
    correction_method = 1  # 0 for photobleaching exponential fit; 1 for background subtraction

    # Test  Extraction Before Subtraction
    # write_dataframe_to_excel(get_roi_at_certain_time(downsampled_stack,labeled_neurons,46, 64), output_folder, 'Test_Raw2' + video_index)

    if correction_method == 0:
        stack = photo_bleach_correction(raw_stack)
        correction_method_name = 'PhotoBleachingCorrected'
    elif correction_method == 1:
        stack, df = background_subtraction(raw_stack, labeled_neurons, background_roi=background_roi)
        write_dataframe_to_excel(df, output_folder, 'BG' + video_index)
        correction_method_name = 'BackGroundSubtracted'

    neuron_subtracted_trace_dataframe = trace_extraction(stack, labeled_neurons)
    write_dataframe_to_excel(neuron_subtracted_trace_dataframe, output_folder,
                             correction_method_name + 'Trace' + video_index)

    delta_f_f_dataframe = calculate_delta_f_over_f(neuron_subtracted_trace_dataframe)
    write_dataframe_to_excel(delta_f_f_dataframe, output_folder, correction_method_name + 'DFF' + video_index)


def clustering_execution(output_folder, video_index):
    cluster_input = pd.read_excel(output_folder + 'BackGroundSubtractedTrace' + str(video_index) + '.xlsx')
    trace_clustering(cluster_input, output_folder, video_index)


source_folder = 'C:\\Lab\\#Yinan\\ROI Extraction\\Test Videos\\'
output_folder = 'C:\\Lab\\#Yinan\\ROI Extraction\\Test Output\\'
# output_folder = 'C:\\Lab\\#Yinan\\ROI Extraction\\Test Output\\New folder\\'
# video_name = '496_test_els_1.tif'
video_name = '496_test.tif'
sampling_rate = 2  # Hz


# raw_stack, video_index = data_import(source_folder, video_name, sampling_rate)
#
# a, b = find_roi_slices(raw_stack, 500, 550, min_size_of_ROI=20)
#
# plot_neuron_label_position(a, b, output_folder, 1, video_index, raw_stack)
#
# exit()

















raw_stack, video_index = data_import(source_folder, video_name, sampling_rate)
#
labeled_neurons = find_roi_execution(raw_stack,400, 500,specify_frame=123)

# data_generation(raw_stack, labeled_neurons, output_folder, video_index, sampling_rate,[[8,800],[120,913]])
# data_generation(raw_stack, labeled_neurons, output_folder, video_index, sampling_rate, [])
# clustering_execution(output_folder, video_index)
# plot_trace()
