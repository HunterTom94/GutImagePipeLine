import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DataOrganizer_custom import trace_for_lineplot_stimulus, trace_for_lineplot_stimulus_cell
from sys import exit
import cv2


def plot_cluster_trace_grid_new(output_folder, organized_pls_label, scheme, row_label, stimulus_ls, ylim=[-1, 7],
                                linewidth=12, color='k', height=2, width=5, svg=0, cluster_order=[], stimulus_bar_y=-1,
                                stimulus_bar_lw=20, scalebar_frame_length=10, scalebar_amplitude_length=1, min_cell_per_sample=3, plot_name = ''):
    lineplot_dict = trace_for_lineplot_stimulus(organized_pls_label, row_label, scheme, output_folder, min_cell_per_sample)
    lineplot_dict = lineplot_dict.loc[lineplot_dict['stimulus'].isin(stimulus_ls), :]

    fig = plt.figure(figsize=(width * 3, height * 3), constrained_layout=True)
    scalebar_row = 5
    scalebar_col = 5
    col_gap = 15

    frame_ls = [lineplot_dict.loc[lineplot_dict['stimulus'] == stimulus, :]['frame'].max() + 1 for stimulus in
                stimulus_ls]
    total_frame_len = np.sum(frame_ls)
    frame_ls_cum = np.insert(np.cumsum(frame_ls), 0, 0)
    grid = plt.GridSpec(len(lineplot_dict[row_label].unique()) * 10 + scalebar_row,
                        (len(stimulus_ls)) * col_gap + scalebar_col + total_frame_len)
    print([len(lineplot_dict[row_label].unique()) * 10 + scalebar_row,
                        (len(stimulus_ls)) * col_gap + scalebar_col + total_frame_len])

    if cluster_order:
        enum_obj = enumerate(cluster_order)
    else:
        enum_obj = enumerate(lineplot_dict[row_label].unique())
    for cluster_ind, key in enum_obj:
        print(key)
        plot_height = 9
        grid_row = cluster_ind * plot_height
        for stimulus_ind, stimulus in enumerate(stimulus_ls):

            trace_data = lineplot_dict.loc[lineplot_dict[row_label] == key, :].loc[
                         lineplot_dict['stimulus'] == stimulus, :]
            trace_data = trace_data[trace_data['frame']!=0]

            print('cluster index: {}'.format(cluster_ind))
            grid_col = frame_ls_cum[stimulus_ind] + col_gap * stimulus_ind
            plot_width = frame_ls[stimulus_ind]
            print('stimulus: {}'.format(stimulus))
            print([grid_row, grid_col])
            print([grid_row + plot_height, grid_col + plot_width])
            ax = fig.add_subplot(grid[grid_row:grid_row + plot_height, grid_col:grid_col + plot_width])
            sns.set_context("poster")
            sns.lineplot(x='frame', y='value', data=trace_data, c=color, ax=ax, ci=68, lw=linewidth, err_kws={'lw':0})
            ax.axhline(c='gray', lw=stimulus_bar_lw, y=stimulus_bar_y,
                       xmin=(trace_data['stimulus_start'].iloc[0] - trace_data['starting_frame'].iloc[0]) / frame_ls[
                           stimulus_ind],
                       xmax=(trace_data['stimulus_end'].iloc[0] - trace_data['starting_frame'].iloc[0]) / frame_ls[
                           stimulus_ind], solid_capstyle="butt")
            ax.set_ylim(ylim)
            ax.set_xlim([0, np.max(trace_data['frame'].to_numpy())])
            ax.axis('off')

    # Create Time Scale Bar
    ax = fig.add_subplot(grid[grid_row + plot_height:len(lineplot_dict[row_label].unique()) * 10 + scalebar_row,
                         grid_col:grid_col + plot_width])
    sns.set_context("poster")
    sns.lineplot(x='frame', y='value', data=trace_data, c=color, ax=ax, ci=68, lw=linewidth)
    ax.axhline(c='k', lw=stimulus_bar_lw, y=-100, xmin=0,xmax=scalebar_frame_length / frame_ls[stimulus_ind], solid_capstyle="butt")
    ax.set_ylim([-99, -101])
    ax.set_xlim([0, np.max(trace_data['frame'].to_numpy())])
    ax.axis('off')

    # Create Time Scale Bar
    ax = fig.add_subplot(grid[grid_row:grid_row + plot_height,
                         grid_col + plot_width:(len(stimulus_ls)) * col_gap + scalebar_col + total_frame_len])
    sns.set_context("poster")
    sns.lineplot(x='frame', y='value', data=trace_data, c=color, ax=ax, ci=68, lw=linewidth)
    ax.axvline(c='k', lw=stimulus_bar_lw, x=-100, ymin=0, ymax=scalebar_amplitude_length / np.abs(np.diff(ylim)), solid_capstyle="butt")
    ax.set_ylim(ylim)
    ax.set_xlim([-99,-101])
    ax.axis('off')
    if svg:
        plt.savefig(output_folder + 'Average_{}.svg'.format(plot_name))
    plt.savefig(output_folder + 'Average_{}.png'.format(plot_name))
    plt.show()


def plot_cluster_trace_grid_cell_average(output_folder, organized_pls_label, scheme, row_label, stimulus_ls, ylim=[-1, 7],
                                linewidth=12, color='k', height=2, width=5, svg=0, cluster_order=[], stimulus_bar_y=-1,
                                stimulus_bar_lw=20, scalebar_frame_length=10, scalebar_amplitude_length=1, min_cell_per_sample=3, plot_name = ''):
    lineplot_dict = trace_for_lineplot_stimulus_cell(organized_pls_label, row_label, scheme, output_folder, min_cell_per_sample)
    lineplot_dict = lineplot_dict.loc[lineplot_dict['stimulus'].isin(stimulus_ls), :]

    fig = plt.figure(figsize=(width * 3, height * 3), constrained_layout=True)
    scalebar_row = 5
    scalebar_col = 5
    col_gap = 15

    frame_ls = [lineplot_dict.loc[lineplot_dict['stimulus'] == stimulus, :]['frame'].max() + 1 for stimulus in
                stimulus_ls]
    total_frame_len = np.sum(frame_ls)
    frame_ls_cum = np.insert(np.cumsum(frame_ls), 0, 0)
    grid = plt.GridSpec(len(lineplot_dict[row_label].unique()) * 10 + scalebar_row,
                        (len(stimulus_ls)) * col_gap + scalebar_col + total_frame_len)
    print([len(lineplot_dict[row_label].unique()) * 10 + scalebar_row,
                        (len(stimulus_ls)) * col_gap + scalebar_col + total_frame_len])

    if cluster_order:
        enum_obj = enumerate(cluster_order)
    else:
        enum_obj = enumerate(lineplot_dict[row_label].unique())
    for cluster_ind, key in enum_obj:
        print(key)
        plot_height = 9
        grid_row = cluster_ind * plot_height
        for stimulus_ind, stimulus in enumerate(stimulus_ls):

            trace_data = lineplot_dict.loc[lineplot_dict[row_label] == key, :].loc[
                         lineplot_dict['stimulus'] == stimulus, :]
            trace_data = trace_data[trace_data['frame']!=0]

            print('cluster index: {}'.format(cluster_ind))
            grid_col = frame_ls_cum[stimulus_ind] + col_gap * stimulus_ind
            plot_width = frame_ls[stimulus_ind]
            print('stimulus: {}'.format(stimulus))
            print([grid_row, grid_col])
            print([grid_row + plot_height, grid_col + plot_width])
            ax = fig.add_subplot(grid[grid_row:grid_row + plot_height, grid_col:grid_col + plot_width])
            sns.set_context("poster")
            sns.lineplot(x='frame', y='value', data=trace_data, c=color, ax=ax, ci=68, lw=linewidth, err_kws={'lw':0})
            ax.axhline(c='gray', lw=stimulus_bar_lw, y=stimulus_bar_y,
                       xmin=(trace_data['stimulus_start'].iloc[0] - trace_data['starting_frame'].iloc[0]) / frame_ls[
                           stimulus_ind],
                       xmax=(trace_data['stimulus_end'].iloc[0] - trace_data['starting_frame'].iloc[0]) / frame_ls[
                           stimulus_ind], solid_capstyle="butt")
            ax.set_ylim(ylim)
            ax.set_xlim([0, np.max(trace_data['frame'].to_numpy())])
            ax.axis('off')

    # Create Time Scale Bar
    ax = fig.add_subplot(grid[grid_row + plot_height:len(lineplot_dict[row_label].unique()) * 10 + scalebar_row,
                         grid_col:grid_col + plot_width])
    sns.set_context("poster")
    sns.lineplot(x='frame', y='value', data=trace_data, c=color, ax=ax, ci=68, lw=linewidth)
    ax.axhline(c='k', lw=stimulus_bar_lw, y=-100, xmin=0,xmax=scalebar_frame_length / frame_ls[stimulus_ind], solid_capstyle="butt")
    ax.set_ylim([-99, -101])
    ax.set_xlim([0, np.max(trace_data['frame'].to_numpy())])
    ax.axis('off')

    # Create Time Scale Bar
    ax = fig.add_subplot(grid[grid_row:grid_row + plot_height,
                         grid_col + plot_width:(len(stimulus_ls)) * col_gap + scalebar_col + total_frame_len])
    sns.set_context("poster")
    sns.lineplot(x='frame', y='value', data=trace_data, c=color, ax=ax, ci=68, lw=linewidth)
    ax.axvline(c='k', lw=stimulus_bar_lw, x=-100, ymin=0, ymax=scalebar_amplitude_length / np.abs(np.diff(ylim)), solid_capstyle="butt")
    ax.set_ylim(ylim)
    ax.set_xlim([-99,-101])
    ax.axis('off')
    if svg:
        plt.savefig(output_folder + 'Average_{}.svg'.format(plot_name))
    plt.savefig(output_folder + 'Average_{}.png'.format(plot_name))
    plt.show()