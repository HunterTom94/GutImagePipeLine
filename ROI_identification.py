from sys import exit
from skimage.filters import sobel
from skimage import morphology
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt


def neuron_label_matrix2dataframe(matrix, frame_index=None):
    records = []
    print('unfiltered by size unique_roi_num is ' + str(len(np.unique(matrix))-1))
    for unique_id in np.unique(matrix):
        if unique_id == 0:
            continue
        for x, y in zip(*np.where(matrix == unique_id)):
            records.append([unique_id, x, y])
        dataframe = pd.DataFrame(records, columns=['unique_id', 'x', 'y'])
    dataframe['pixel_size'] = ''
    if frame_index is None:
        dataframe['frame_index'] = 'max'
    else:
        dataframe['frame_index'] = frame_index
    for unique_id in np.unique(matrix):
        if unique_id == 0:
            continue
        ROI_number_of_pixels = dataframe[dataframe['unique_id'] == unique_id].count()['x']
        dataframe.loc[dataframe['unique_id'] == unique_id, 'pixel_size'] = ROI_number_of_pixels

    return dataframe


def neuron_label_dataframe2matrix(dataframe, matrix_size):
    matrix = np.zeros(matrix_size)
    for _, row in dataframe.iterrows():
        matrix[row['x'], row['y']] = row['unique_id']

    return matrix


def find_roi(image, lower_threshold, upper_threshold, min_size_of_ROI=[], frame_index=None):
    print(image.dtype)
    elevation_map = sobel(image)
    markers = np.zeros_like(image)
    markers[image < lower_threshold] = 1
    markers[image > upper_threshold] = 2

    segmentation = morphology.watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_neurons, _ = ndi.label(segmentation)

    plt.imshow(labeled_neurons, aspect="auto")
    plt.show()

    labeled_neurons_df = neuron_label_matrix2dataframe(labeled_neurons, frame_index)

    if min_size_of_ROI:
        labeled_neurons_df = labeled_neurons_df.drop(
            labeled_neurons_df[labeled_neurons_df['pixel_size'] <= min_size_of_ROI].index)

    # Reorder index after dropping
    for index, unique_id in enumerate(np.unique(labeled_neurons_df['unique_id'])):
        labeled_neurons_df['unique_id'] = labeled_neurons_df['unique_id'].replace(unique_id, index + 1)

    updated_labeled_neurons = neuron_label_dataframe2matrix(labeled_neurons_df, labeled_neurons.shape)

    return updated_labeled_neurons, labeled_neurons_df


def find_roi_slices(raw_stack, lower_threshold, upper_threshold, min_size_of_ROI=[]):
    output_df = pd.DataFrame()
    labeled_neurons_slice = np.array([])
    for frame_index in range(raw_stack.shape[0]):
        labeled_neurons, labeled_neurons_df = find_roi(raw_stack[frame_index, :, :], lower_threshold,
                                                       upper_threshold, min_size_of_ROI=min_size_of_ROI,
                                                       frame_index=frame_index)
        output_df = output_df.append(labeled_neurons_df, ignore_index=True)
        labeled_neurons_slice = np.append(labeled_neurons_slice, labeled_neurons)
        print(frame_index)

    labeled_neurons_slice = np.reshape(labeled_neurons_slice, (-1, labeled_neurons.shape[0], labeled_neurons.shape[1]))

    return labeled_neurons_slice, output_df


def plot_neuron_label_position(labeled_neurons, labeled_neurons_df, output_path, max_z_projection, filename,
                               stack=None):
    sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 0.5})
    matplotlib.rc('figure', dpi=300)  # make figures more clear
    if stack is None:
        # load data and image
        image = Image.fromarray(max_z_projection / 10)

        # get signal's median
        median_positions = labeled_neurons_df.groupby('unique_id').median()

        fig, ax = plt.subplots(figsize=(10.21, 5.51))
        # show image
        ax.imshow(image, interpolation='nearest', aspect='auto')

        # show data
        # mask some 'bad' data, in your case you would have: data == 0
        masked_data = np.ma.masked_where(labeled_neurons == 0, labeled_neurons)
        cmap = plt.cm.viridis
        cmap.set_bad(color='#00000000')
        ax.imshow(masked_data, interpolation='none', cmap=cmap)

        # add annotation
        for unique_id, row in median_positions.iterrows():
            x = int(row['x'])
            y = int(row['y'])
            ax.text(y, x, unique_id)
        plt.show()
        fig.savefig(output_path + filename + '_neuron_label.png')
    else:
        for frame_index in range(stack.shape[0]):
            image = Image.fromarray(stack[frame_index, :, :] / 10)
            median_positions = labeled_neurons_df.loc[labeled_neurons_df['frame_index'] == frame_index].groupby(
                'unique_id').median()
            fig, ax = plt.subplots(figsize=(10.21, 5.51))
            ax.imshow(image, interpolation='nearest', aspect='auto')

            masked_data = np.ma.masked_where(labeled_neurons[frame_index, :, :] == 0,
                                             labeled_neurons[frame_index, :, :])
            cmap = plt.cm.viridis
            cmap.set_bad(color='#00000000')
            ax.imshow(masked_data, interpolation='none', cmap=cmap)
            # print(median_positions)
            # for unique_id, (x, y) in median_positions.iterrows():
            #     ax.text(y, x, unique_id)
            for unique_id, row in median_positions.iterrows():
                # print(row)
                # exit()
                x = int(row['x'])
                y = int(row['y'])
                ax.text(y, x, unique_id)
            fig.savefig(output_path + filename + str(frame_index) + '_neuron_label.png')

def thresholding(image,cutoff,min):
    image[image >= cutoff] = 1
    image[image < cutoff] = 0
    return image




def erase_ROI(frame1,frame2,max_allowed_difference):
    roi_1 = find_roi(frame1, 10, 20, min_size_of_ROI=[], frame_index=None)
    plt.imshow(roi_1, aspect="auto")
    plt.show()

