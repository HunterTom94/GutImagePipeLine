import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from ROI_identification import neuron_label_matrix2dataframe
from Igor_related_util import read_igor_roi_matrix
import seaborn as sns
import matplotlib
import os
import cv2
from scipy import signal
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib as mpl
import scipy
from sys import exit

def write_dataframe_to_excel(dataframe, root_folder, filename, sheetname = 'Sheet1'):
    if filename.split('.')[-1] is not 'xlsx':
        filename = filename + '.xlsx'
    path = root_folder + filename
    exlwriter = pd.ExcelWriter(path)
    dataframe.to_excel(exlwriter, index=False, sheet_name=sheetname)
    exlwriter.save()
    exlwriter.close()


def get_roi_at_certain_time(stack, labeled_neurons, roi_index, time):
    mask = (labeled_neurons == roi_index).astype(int)
    masked_stack = stack * mask
    output_image_frame = masked_stack[time, :, :]
    # print(output_image_frame.shape)
    # exit()
    # np.reshape(output_image_frame, output_image_frame.shape()[1:3])

    return pd.DataFrame(output_image_frame)


def calculate_delta_f_over_f(neuron_trace_dataframe, f_start_frame, f_end_frame):
    f_list = []
    delta_f_f_dataframe = neuron_trace_dataframe.copy()

    for col in neuron_trace_dataframe:
        f = np.mean(neuron_trace_dataframe[col][f_start_frame:f_end_frame])
        f_list.append(f)

        delta_f_f = (neuron_trace_dataframe[col] - f) / f
        delta_f_f_dataframe[col] = delta_f_f

    return delta_f_f_dataframe


def trace_clustering(dataframe, output_path, filename):
    array = np.transpose(dataframe.values)

    clustering_method = 'KMeans'
    if clustering_method == 'AffinityPropagation':
        clustering = cluster.AffinityPropagation().fit(array)
    elif clustering_method == 'KMeans':
        clustering = cluster.KMeans(n_clusters=6).fit(array)
    labels = clustering.labels_
    print(labels)

    for cluster_group_index in np.unique(labels):

        fig = plt.figure()
        ax = fig.gca()
        trace_index_list = np.argwhere(labels == cluster_group_index)
        for count, trace_index in enumerate(trace_index_list):
            trace_length = array[trace_index, :].shape[1]
            trace = array[trace_index, :].reshape((trace_length))
            if count == 0:
                average = trace
            elif count >= 1:
                average = np.mean(np.concatenate((average, trace)).reshape(2, trace_length), axis=0)
            ax.plot(np.arange(0, len(trace), 1), trace, 'b')
        ax.plot(np.arange(0, len(average), 1), average, 'r')
        plt.show()
        fig.savefig(output_path + filename + clustering_method + '_' + str(cluster_group_index) + '_cluster.png')


def plot_trace(dataframe, output_path, filename):
    array = np.transpose(dataframe.values)
    for trace_index in range(array.shape[0]):
        fig = plt.figure()
        ax = fig.gca()
        trace = array[trace_index, :]
        stimulus = np.zeros_like(trace)
        stimulus[60:80] = 1
        stimulus = stimulus * np.amax(trace)
        ax.plot(np.arange(0, len(trace), 1), trace, 'b')
        ax.plot(np.arange(0, len(stimulus), 1), stimulus, 'r')
        plt.show()
        # fig.savefig(output_path + filename + clustering_method + '_' + str(cluster_group_index) + '_cluster.png')

def plot_cumulative(x):
    d = np.reshape(x,(1,-1))
    d = np.sort(d)
    a,b = np.histogram(d,bins=100)
    a_sum = np.sum(a)

    check = 0
    for ind,i in enumerate(a/a_sum):
        if i >0.1:
            check = 1
        if i < 0.01 and check == 1:
            break
    print(b[ind])
    return b[ind]


def plot_histogram(array,bin_numer):
    # Arg:    array -- np array
    # Plot x axis is mean of bin edges
    # array = np.ndarray(array).flatten()
    a, b = np.histogram(array, bins=bin_numer)
    # c = [np.mean(b[x],b[x+1]) for x in range(len(b)-1)]
    plt.plot(b[1:], a)
    plt.show()


def roi_plot(ROI_matrix, root_folder, label_font, targets=[], throw_ls=[]):
    #  Generate ROI graph with labelled index on the top right corner of each ROI
    #
    #  Arg:    ROI_matrix  -- 2darray, same format as output of read_igor_roi_matrix;
    #                           background as 0, roi masks with index number, roi index starts from 1;
    #                           roi matrix containing all roi information from igor output.
    #          root_folder -- string, absolute folder path where the label graph will be saved.
    #          label_font  -- int, size of font of labelled number shown in the graph, suggest range 2~20
    # ROI_matrix = np.transpose(ROI_matrix)
    # ROI_matrix = np.rot90(ROI_matrix)


    ROI_firing = np.copy(ROI_matrix)

    ROI_firing[ROI_firing != 0] = -1
    if targets:
        for target in targets:
            ROI_firing[ROI_matrix == target+1] = -5

    if len(throw_ls)>0:
        for throw in throw_ls:
            ROI_firing[ROI_matrix == throw+1] = -7

    color_min = -10
    color_max = -1

    sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 0.5})
    matplotlib.rc('figure', dpi=300)  # make figures more clear

    # get signal's median


    if os.path.exists(root_folder + "cell_position.pkl"):
        median_positions = pd.read_pickle(root_folder + 'cell_position.pkl')
    else:
        labeled_neurons_df = neuron_label_matrix2dataframe(ROI_matrix)
        median_positions = labeled_neurons_df.groupby('unique_id').median()
        median_positions.to_pickle(root_folder + 'cell_position.pkl')

    fig, ax = matplotlib.pyplot.subplots(figsize=(20, 20))

    # show data
    # mask some 'bad' data, in this case it would be: data == 0
    masked_data = np.ma.masked_where(ROI_firing == 0, ROI_firing)
    cmap = matplotlib.pyplot.cm.hot_r
    cmap.set_bad(color='w')
    ax.imshow(masked_data, interpolation='none', cmap=cmap, vmin=color_min, vmax=color_max)

    # add annotation
    for unique_id, row in median_positions.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        ax.text(y + 3, x, int(unique_id - 1), color='tab:blue', fontsize=label_font)
    matplotlib.pyplot.show()
    if targets:
        fig.savefig(root_folder + 'target_roi_neuron_label.png')
    if len(throw_ls)>0:
        fig.savefig(root_folder + 'roi_neuron_label_screened.png')
    else:
        fig.savefig(root_folder + 'roi_neuron_label.png')


def colorful_roi_map(root_folder, BGR_dict):
    ROI_matrix = read_igor_roi_matrix(root_folder + 'roi.csv')
    position_df = pd.read_pickle(root_folder + 'cell_position.pkl')

    draw_pad = np.ones((ROI_matrix.shape[0], ROI_matrix.shape[1], 3)).astype('uint8')
    draw_pad *= 255
    # zeros = cv2.normalize(np.zeros(ROI_matrix.shape).astype(int), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # draw_pad = np.stack((zeros, zeros, zeros))
    # draw_pad = np.swapaxes(draw_pad, 0, 2)

    target_dict = {}
    target_df = pd.read_csv(root_folder+'colorful_roi.txt',sep='\t')
    colors = target_df.columns
    for color in colors:
        target_dict.update({color: target_df[color].to_numpy()})

    # blue_channel = zeros.copy()
    # green_channel = zeros.copy()
    # red_channel = zeros.copy()

    # drawed = []
    for color, targets in target_dict.items():
        for target in targets:
            if not np.isnan(target):
                coordinate = [int(position_df.loc[target+1]['x']), int(position_df.loc[target+1]['y'])]
                cv2.circle(draw_pad, center=(coordinate[1], coordinate[0]), radius=3,
                           color=(BGR_dict[color][0], BGR_dict[color][1], BGR_dict[color][2]), thickness=-1,
                           lineType=cv2.LINE_AA)
                # drawed.append(target+1)


                # blue_channel[ROI_matrix == target + 1] = BGR_dict[color][0]
                # draw_pad[:, :, 0] = blue_channel
                # green_channel[ROI_matrix == target + 1] = BGR_dict[color][1]
                # draw_pad[:, :, 1] = green_channel
                # red_channel[ROI_matrix == target + 1] = BGR_dict[color][2]
                # draw_pad[:, :, 2] = red_channel

    # not_drawed = [roi for roi in position_df.index if roi not in drawed]
    # for not_drawed_target in not_drawed:
    #     blue_channel[ROI_matrix == not_drawed_target] = 100
    #     draw_pad[:, :, 0] = blue_channel
    #     green_channel[ROI_matrix == not_drawed_target] = 100
    #     draw_pad[:, :, 1] = green_channel
    #     red_channel[ROI_matrix == not_drawed_target] = 100
    #     draw_pad[:, :, 2] = red_channel
        # coordinate = [int(position_df.loc[not_drawed_target]['x']), int(position_df.loc[not_drawed_target]['y'])]
        # cv2.circle(draw_pad, center=(coordinate[1], coordinate[0]), radius=3,
        #            color=(100, 100, 100), thickness=1,
        #            lineType=cv2.LINE_AA)

    cv2.imwrite(root_folder + 'colormap_all.png', draw_pad)


def level_one_palette(name_list, order=None, palette='auto'):
    name_set = set(name_list)
    if palette == 'auto':
        if len(name_set) < 10:
            palette = 'tab10'
        elif len(name_set) < 20:
            palette = 'tab20'
        else:
            palette = 'rainbow'

    if order is None:
        order = list(sorted(name_set))
    else:
        if (set(order) != name_set) or (len(order) != len(name_set)):
            raise ValueError('Order is not equal to set(name_list).')
    n = len(order)

    colors = sns.color_palette(palette, n)
    color_palette = {}
    for name, color in zip(order, colors):
        color_palette[name] = color
    return color_palette


def _continuous_color_palette(color, n, skip_border=1):
    """
    This function concatenate the result of both sns.light_palette
    and sns.dark_palette to get a wider color range
    """
    if n == 1:
        return [color]
    if n < 1:
        raise ValueError('parameter n colors must >= 1.')

    # this is just trying to make sure len(color) == n
    light_n = (n + 2 * skip_border) // 2
    light_colors = sns.light_palette(color, n_colors=light_n)[skip_border:]
    dark_n = n + 2 * skip_border - light_n + 1
    dark_colors = sns.dark_palette(color, n_colors=dark_n, reverse=True)[1:-skip_border]
    colors = light_colors + dark_colors
    return colors


def level_two_palette(major_color, major_sub_dict,
                      major_order=None, palette='auto',
                      skip_border_color=2):
    if isinstance(major_color, list):
        major_color_dict = level_one_palette(major_color, palette=palette, order=major_order)
    else:
        major_color_dict = major_color

    sub_id_list = []
    for subs in major_sub_dict.values():
        sub_id_list += list(subs)
    if len(sub_id_list) != len(set(sub_id_list)):
        raise ValueError('Sub id in the major_dub_dict is not unique.')

    color_palette = {}
    for major, color in major_color_dict.items():
        subs = major_sub_dict[major]
        n = len(subs)
        colors = _continuous_color_palette(color, n, skip_border=skip_border_color)
        for sub, _color in zip(subs, colors):
            color_palette[sub] = _color
    return color_palette


def palplot(pal, transpose=False):
    if transpose:
        fig, ax = plt.subplots(figsize=(1, len(pal)))
    else:
        fig, ax = plt.subplots(figsize=(len(pal), 1))
    n = len(pal)
    data = np.arange(n).reshape(1, n)
    if transpose:
        data = data.T
    ax.imshow(data, interpolation="nearest", aspect="auto",
              cmap=ListedColormap(list(pal.values())))
    if not transpose:
        ax.set(xticklabels=list(pal.keys()),
               xticks=range(0, len(pal)),
               yticks=[])
        ax.xaxis.set_tick_params(labelrotation=90)
    else:
        ax.set(yticklabels=list(pal.keys()),
               yticks=range(0, len(pal)),
               xticks=[])
    return fig, ax


def plot_colorbar(cax, hue_nrom, cmap='viridis', orientation='vertical'):
    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)

    # plot color bar
    cnorm = mpl.colors.Normalize(vmin=hue_nrom[0], vmax=hue_nrom[1])
    colorbar = ColorbarBase(cax, cmap=cmap, norm=cnorm,
                            orientation=orientation)
    return colorbar


def low_pass_filter(trace, sampling_freq, cutoff_freq):
    w = cutoff_freq / (sampling_freq / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    return signal.filtfilt(b, a, trace)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h