import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sys import exit
from matplotlib.patches import Rectangle
from util import low_pass_filter

# input_folder = 'D:\\Yinan\\LinearTroubleShoot\\osage3\\'
input_folder = 'D:\\#Yinan\\KCl_fit_test\\analysis\\'
scheme = pd.read_excel(input_folder + '\\DeliverySchemes.xlsx')

# stimuli_ls = scheme['stimulation'].to_list()
stimuli_ls = ['KCl']
frame_ls = scheme['video_end'].to_numpy() - scheme['video_start'].to_numpy() + 1
stimulus_length_ls = scheme['stimulus_end'].to_numpy() - scheme['stimulus_start'].to_numpy() + 1
time_stamp_ls = np.insert(np.cumsum(frame_ls), 0, 0)

organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')

organized = organized[~organized["ROI_index"].str.contains('_b')]

raw_organized = raw_organized[raw_organized['ROI_index'].isin(organized['ROI_index'])]
raw_organized = raw_organized.sort_values(by=['ROI_index'], ascending=True)
organized = organized.sort_values(by=['ROI_index'], ascending=True)
raw_organized.index = range(raw_organized.shape[0])
organized.index = range(organized.shape[0])
assert raw_organized.shape[0] == organized.shape[0]

organized = organized[(organized['KCl_average_response'] == 0) & (raw_organized['KCl_average_peak'] - raw_organized['KCl_f0_mean'] > 150)& (raw_organized['KCl_f0_mean'] < 25)]
raw_organized = raw_organized[(organized['KCl_average_response'] == 0) & (raw_organized['KCl_average_peak'] - raw_organized['KCl_f0_mean'] > 150)& (raw_organized['KCl_f0_mean'] < 25)]

for stimulus_ind, stimulus in enumerate(stimuli_ls):
    organized_stimulus = organized
    # organized_stimulus = organized[organized['{}_average_response'.format(stimulus)] == 0]
    # organized_stimulus = organized[organized['{}_average_peak'.format(stimulus)] > organized['{}_f0_mean'.format(stimulus)] + 3*organized['{}_f0_std'.format(stimulus)]]
    # organized_stimulus = organized_stimulus[
    #     organized_stimulus['{}_average_peak'.format(stimulus)] > organized_stimulus['{}_background_threshold'.format(stimulus)]]

    col_num = 4
    # row_num = int(np.ceil(np.min([50, organized_stimulus.shape[0]])/col_num))*2
    row_num = int(np.ceil(organized_stimulus.shape[0] / col_num)) * 2
    fig = plt.figure(figsize=(col_num*4, row_num*3), constrained_layout=True)
    grid = plt.GridSpec(row_num, col_num, wspace=0.3, hspace=0.5)

    # organized_stimulus = organized_stimulus.drop(organized_stimulus[(organized_stimulus['{}_max_first_derivative_index'.format(stimulus)] < 7) |
    #                                         ((organized_stimulus['{}_max_first_derivative_index'.format(stimulus)] > 7) &
    #                                          (organized_stimulus['{}_max_first_derivative'.format(stimulus)] <= 0.1))].index, axis=0)

    # organized_stimulus = organized_stimulus[(organized_stimulus['{}_max_first_derivative_index'.format(stimulus)] < 7) |
    #                        ((organized_stimulus['{}_max_first_derivative_index'.format(stimulus)] > 7) &
    #                         (organized_stimulus['{}_max_first_derivative'.format(stimulus)] <= 0.1))]

    # organized_stimulus = organized_stimulus[organized_stimulus['{}_max_first_derivative_index'.format(stimulus)] > 7]
    # organized_stimulus = organized_stimulus[organized_stimulus['{}_max_first_derivative'.format(stimulus)] <= 0.1]
    # organized_stimulus = organized_stimulus[organized_stimulus['{}_max_second_derivative_index'.format(stimulus)] > 7]
    # organized_stimulus = organized_stimulus[organized_stimulus['{}_max_second_derivative'.format(stimulus)] <= 0.1]
    # sampled = organized_stimulus.sample(np.min([50, organized_stimulus.shape[0]]), random_state=0)


    sampled = organized_stimulus
    trace_ind = 0
    for _, row in sampled.iterrows():
        trace = row['{}_trace'.format(stimulus)]
        trace = trace[~np.isnan(trace)]
        grid_row = int(trace_ind//4)
        grid_col = trace_ind % 4
        print(trace_ind)
        print(grid_row)
        print(grid_col)
        print()

        filtered_row = low_pass_filter(trace, 0.25, 0.03)

        std2p5 = np.round(row['{}_f0_mean'.format(stimulus)] + 2.5 * row['{}_f0_std'.format(stimulus)], 2)
        std3p5 = np.round(row['{}_f0_mean'.format(stimulus)] + 3.5 * row['{}_f0_std'.format(stimulus)], 2)
        std4p5 = np.round(row['{}_f0_mean'.format(stimulus)] + 4.5 * row['{}_f0_std'.format(stimulus)], 2)
        ax = fig.add_subplot(grid[grid_row*2, grid_col])
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
        ax.set_yticks([np.min(trace).round(1) + 0.1, 0, 2,4,6,8,10, np.max(trace).round(1) - 0.1])
        # ax.set_yticks([np.min(trace).round(1) + 0.1, 0, std2p5, std3p5, std4p5, np.max(trace).round(1) - 0.1])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim([np.nanmin(trace), np.nanmax([std4p5, np.max(trace)]) + 0.1])
        ax.set_xlim([0, len(trace)])
        ax.title.set_text(row['ROI_index'])
        sns.lineplot(x=range(len(trace)), y=trace, ax=ax)
        sns.lineplot(x=range(len(trace)), y=filtered_row, ax=ax)
        ax.axvline(x=row['{}_average_peak_index'.format(stimulus)], ymin=-1, ymax=1, color='r', ls='--')
        ax.add_patch(Rectangle((8, -5), stimulus_length_ls[stimulus_ind], 10, alpha=0.5, facecolor='orange'))

        ax.axhline(y=std2p5, xmin=0, xmax=1, color='b', ls='--', lw=0.5)
        ax.axhline(y=std3p5, xmin=0, xmax=1, color='b', ls='--', lw=0.5)
        ax.axhline(y=std4p5, xmin=0, xmax=1, color='b', ls='--', lw=0.5)
        ax.axhline(y=row['{}_average_peak'.format(stimulus)], xmin=0, xmax=1, color='g', ls='--', lw=0.5)

        ax = fig.add_subplot(grid[grid_row*2+1, grid_col])
        sns.lineplot(x=range(len(trace)), y=np.gradient(filtered_row), ax=ax)
        sns.lineplot(x=range(len(trace)), y=np.gradient(np.gradient(filtered_row)), ax=ax)

        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
        ax.set_yticks([-0.5,-0.2,0,0.2,0.5])
        # ax.set_yticks([np.min(trace).round(1) + 0.1, 0, std2p5, std3p5, std4p5, np.max(trace).round(1) - 0.1])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlim([0, len(trace)])
        ax.axhline(y=0.1, xmin=0, xmax=1, color='r', ls='--', lw=0.5)

        trace_ind += 1

    plt.tight_layout()
    plt.suptitle(stimuli_ls[stimulus_ind])
    # plt.savefig('D:\\Yinan\\Clustering\\DFF\\New folder\\Keep'+stimuli_ls[stimulus_ind]+'.pdf')
    plt.savefig(input_folder + 'trace1.pdf')
    plt.show()