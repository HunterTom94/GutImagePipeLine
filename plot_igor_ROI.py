import numpy as np
from ROI_identification import neuron_label_matrix2dataframe
import seaborn as sns
import matplotlib
from data_generation import trace_extraction
from util import read_tif, write_dataframe_to_excel, calculate_delta_f_over_f
import pandas as pd
from sys import exit

def roi_plot(ROI_matrix,f_dataframe):
    f = pd.read_excel('C:\\Lab\\#Yinan\\ROI Extraction\\Output\\638_Pros_25AA\\Traced638_stitched.xlsx')
    dff = calculate_delta_f_over_f(f,110,120)

    # F_value = np.loadtxt(open("C:\\Lab\\#Yinan\\ROI Extraction\\Output\\638_Pros_25AA\\f.csv", "rb"), delimiter=",", skiprows=1)
    # F_value = np.delete(F_value, 0, axis=0)

    ROI_matrix = np.rot90(ROI_matrix)
    labeled_neurons_df = neuron_label_matrix2dataframe(ROI_matrix)

    ROI_firing = np.copy(ROI_matrix)
    local_max_ls = []

    # column named after index start from 0
    for column in dff:
        local_max = np.max(dff[column][120:240])
        local_max_ls.append(local_max)
        ROI_firing[np.where(ROI_firing == column + 1)] = local_max
        if column in [255,257,281]:
            print(column)
            print(local_max)

    # for count, column in enumerate(F_value.T):
    #     local_max = np.max(column[90:150])
    #     local_max_ls.append(local_max)
    #     ROI_firing[np.where(ROI_firing == count+1)] = local_max

    color_min = np.sort(local_max_ls)[int(np.ceil(len(local_max_ls)*0.1))]
    color_max = np.sort(local_max_ls)[int(np.floor(len(local_max_ls)*0.9))]
    print(color_min)
    print(color_max)

    sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 0.5})
    matplotlib.rc('figure', dpi=300)  # make figures more clear

    # get signal's median
    median_positions = labeled_neurons_df.groupby('unique_id').median()

    fig, ax = matplotlib.pyplot.subplots(figsize=(10.21, 5.51))

    # show data
    # mask some 'bad' data, in your case you would have: data == 0
    masked_data = np.ma.masked_where(ROI_firing == 0, ROI_firing)
    cmap = matplotlib.pyplot.cm.hot_r
    cmap.set_bad(color='w')
    ax.imshow(masked_data, interpolation='none', cmap=cmap, vmin=color_min-1, vmax=color_max)

    # add annotation
    for unique_id, row in median_positions.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        ax.text(y+5, x, int(unique_id-1), color='tab:gray', fontsize = 2)
    matplotlib.pyplot.show()
    fig.savefig('C:\\Lab\\#Yinan\\ROI Extraction\\Output\\638_Pros_25AA\\roi_neuron_label.png')
    return dff

def f_gen(ROI_matrix):
    source_folder = 'C:\\Lab\\#Yinan\\ROI Extraction\\Output\\638_Pros_25AA\\'
    output_folder = 'C:\\Lab\\#Yinan\\ROI Extraction\\Output\\638_Pros_25AA\\'
    video_name = 'd638_stitched.tif'
    sampling_rate = 1  # Hz

    raw_stack, video_index = read_tif(source_folder, video_name)
    df = trace_extraction(raw_stack,ROI_matrix)
    write_dataframe_to_excel(df, output_folder,
                             'Trace' + video_index)
    return df

ROI_matrix = np.loadtxt(open("C:\\Lab\\#Yinan\\ROI Extraction\\Output\\638_Pros_25AA\\roi.csv", "rb"), delimiter=",", skiprows=1)
ROI_matrix = np.delete(ROI_matrix, 0, axis=0)
ROI_matrix = -ROI_matrix
ROI_matrix[np.where(ROI_matrix == -1)] = 0
ROI_matrix = np.transpose(ROI_matrix)
# f_df = f_gen(ROI_matrix)

dff = roi_plot(np.transpose(ROI_matrix), 1)
write_dataframe_to_excel(dff, 'C:\\Lab\\#Yinan\\ROI Extraction\\Output\\638_Pros_25AA\\',
                             'DFF' + '638')
