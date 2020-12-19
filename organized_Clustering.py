import pandas as pd
import numpy as np
from time import time
import umap.umap_ as umap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import os
from util import level_one_palette, level_two_palette, plot_colorbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import math
from sys import exit

matplotlib.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 13})

def factors(value):
    factors = []
    for i in range(1, int(value ** 0.5) + 1):
        if value % i == 0:
            factors.append((i, value / i))
    return factors

def UMAP_gen(output_folder, organized, stimulus_ls, HDBSCAN_para_ls, UMAP_nearest_neighbor_ls, random_state_ls, row_num,
             col_num, average_peak=1, z_score_norm=1, svg=0, hue_norm=(-0.2, 2), AP_hue_norm=(0,100),  dot_size=5, cluster_fontsize=10,
             region_palatte_dict={}, region_type='', color_bar_tick_labels=[], AP_color_bar_tick_labels=[], hspace=3, wspace=3, dot_edgewidth=0.2,
             anyresp=1, xlim_para=[], ylim_para=[], cluster_palatte_dict={}, exclude_kcl=1, UMAP_region_as_input=0):

    def format_UMAP_plot(ax, title):
        ax.set_title(label=title, fontdict={'fontsize': 20})
        ax.legend().set_visible(False)
        sns.despine(ax=ax, offset=10)
        if ylim_para:
            ylim = ylim_para[:2]
            ax.set_yticks(np.arange(ylim_para[0], ylim_para[1] + 1, ylim_para[2]))
        else:
            ylim = [math.floor(ax.get_ylim()[0]), math.ceil(ax.get_ylim()[1])]
            ax.set_yticks(np.arange(ylim[0], ylim[1] + 1))
        ax.set_ylim(ylim)

        if xlim_para:
            xlim = xlim_para[:2]
            ax.set_xticks(np.arange(xlim_para[0], xlim_para[1] + 1, xlim_para[2]))
        else:
            xlim = [math.floor(ax.get_xlim()[0]), math.ceil(ax.get_xlim()[1])]
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 1))
        ax.set_xlim(xlim)

    assert average_peak in [0, 1]
    assert row_num * col_num >= len(stimulus_ls) + 10, 'Grids not enough for all plots'

    if average_peak == 1:
        peak_type = 'average'
    elif average_peak == 0:
        peak_type = 'individual'

    if exclude_kcl and 'KCl' in stimulus_ls:
        organized = organized[organized['KCl_{}_response'.format(peak_type)] == 1]

    clustering_data = organized[
        ['ROI_index', 'region', 'fine_region'] + ['{}_{}_peak'.format(stimulus, peak_type) for stimulus in
                                                  stimulus_ls] + [
            '{}_{}_response'.format(stimulus, peak_type)
            for stimulus in stimulus_ls]]
    if anyresp == 1:
        clustering_data = clustering_data.loc[clustering_data[
                                                  ['{}_{}_response'.format(stimulus, peak_type) for stimulus in
                                                   stimulus_ls if 'KCl' not in stimulus]].any(axis=1), :]
    clustering_data_index = clustering_data.index

    clustering_data = clustering_data.reset_index(drop=True)
    clustering_np = clustering_data[['{}_{}_peak'.format(stimulus, peak_type) for stimulus in stimulus_ls]]

    if UMAP_region_as_input:
        clustering_np['region_number'] = np.nan
        for region_ind, region in enumerate(clustering_data['region'].unique()):
            clustering_np.at[clustering_data['region'] == region, 'region_number'] = region_ind + 1

    clustering_np[['{}_{}_response'.format(stimulus, peak_type) for stimulus in stimulus_ls if 'KCl' not in stimulus]] = \
        clustering_data[
            ['{}_{}_response'.format(stimulus, peak_type) for stimulus in stimulus_ls if 'KCl' not in stimulus]]

    assert z_score_norm in [0, 1]
    if z_score_norm == 1:
        normed_data = (clustering_np - clustering_np.mean()) / clustering_np.std()
        nan_index = np.argwhere(clustering_np.std() == 0).flatten()
        clustering_np = normed_data.to_numpy()
        clustering_np[:, nan_index] = 0
    elif z_score_norm == 0:
        clustering_np = clustering_np.to_numpy()

    metric = 'euclidean'

    for hdbscan_para in HDBSCAN_para_ls:
        min_c = int(hdbscan_para[0])
        min_s = int(hdbscan_para[1])
        for n_n in UMAP_nearest_neighbor_ls:
            for random_state in random_state_ls:
                embedding = umap.UMAP(random_state=random_state, metric=metric, n_neighbors=n_n,
                                      min_dist=0).fit_transform(clustering_np)
                scatter_df = pd.DataFrame()
                scatter_df['UMAP 1'] = embedding[:, 0]
                scatter_df['UMAP 2'] = embedding[:, 1]

                fig = plt.figure(figsize=(col_num * 4, row_num * 4))
                gs = fig.add_gridspec(row_num * 3, col_num * 3, hspace=hspace, wspace=wspace)
                axes = []
                for row in range(row_num):
                    for col in range(col_num):
                        if row * col_num + col == len(stimulus_ls) or row * col_num + col == len(stimulus_ls) + 8:
                            axes.append(fig.add_subplot(gs[3 * row:3 * (row + 1), 3 * col:3 * col + 1]))
                        elif row * col_num + col == len(stimulus_ls) + 1 or row * col_num + col == len(stimulus_ls) + 9:
                            axes.append(fig.add_subplot(gs[3 * row:3 * row + 1, 3 * col:3 * (col + 1)]))
                        else:
                            axes.append(fig.add_subplot(gs[3 * row:3 * (row + 1), 3 * col:3 * (col + 1)]))

                for stimulus_index, ax in enumerate(axes):
                    if stimulus_index > len(stimulus_ls) - 1:
                        continue

                    response = clustering_data[
                        ['{}_{}_peak'.format(stimulus_ls[stimulus_index], peak_type)]].to_numpy().flatten()
                    scatter_df['response'] = response
                    sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='response',
                                    palette='jet', hue_norm=hue_norm, edgecolor='k', linewidth=dot_edgewidth)
                    format_UMAP_plot(ax, stimulus_ls[stimulus_index])

                ax = axes[len(stimulus_ls)]
                colorbar = plot_colorbar(ax, hue_nrom=hue_norm, cmap='jet', orientation='vertical')
                colorbar.set_ticks(color_bar_tick_labels)
                colorbar.set_ticklabels(color_bar_tick_labels)

                ax = axes[len(stimulus_ls) + 1]
                colorbar = plot_colorbar(ax, hue_nrom=hue_norm, cmap='jet', orientation='horizontal')
                colorbar.set_ticks(color_bar_tick_labels)
                colorbar.set_ticklabels(color_bar_tick_labels)

                ax = axes[len(stimulus_ls) + 2]
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_c, min_samples=min_s)
                clusterer.fit(embedding)
                cluster_label = clusterer.labels_ + 1
                label_num = len(np.unique(cluster_label))
                print('{} labels'.format(label_num))
                assert label_num <= len(cluster_palatte_dict), 'Number of Cluster Color Not Enough'

                scatter_df['label'] = cluster_label
                scatter_df[region_type] = organized.loc[clustering_data_index][region_type]
                sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='label',
                                palette=cluster_palatte_dict, edgecolor='k', linewidth=dot_edgewidth)
                for label in np.unique(cluster_label):
                    if label == 0:
                        plt.close()
                        continue
                    ax.text(np.mean(embedding[cluster_label == label, 0]),
                            np.mean(embedding[cluster_label == label, 1]),
                            str(label), fontsize=cluster_fontsize)
                h, l = ax.get_legend_handles_labels()
                format_UMAP_plot(ax, 'Clusters')

                ax = axes[len(stimulus_ls) + 3]
                ax.legend(h[1:], l[1:], ncol=3, fontsize=10, handletextpad=0.1, columnspacing=0.8,
                          title='Cluster Label', frameon=False, bbox_to_anchor=[-0.15, 0.5], loc='center left',
                          borderaxespad=0)
                plt.setp(ax.legend_.get_title(), fontsize=13)
                ax.axis('off')

                ax = axes[len(stimulus_ls) + 4]
                sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, facecolor='k', linewidth=0)
                format_UMAP_plot(ax, 'Raw')

                ax = axes[len(stimulus_ls) + 5]
                sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue=region_type,
                                palette=region_palatte_dict, edgecolor='k', linewidth=dot_edgewidth)

                h, l = ax.get_legend_handles_labels()
                format_UMAP_plot(ax, 'Discrete_AP')

                ax = axes[len(stimulus_ls) + 6]
                ax.legend(h[1:][::-1], l[1:][::-1], ncol=1, fontsize=10, handletextpad=0.1, columnspacing=0.8,
                          title=region_type, frameon=False, bbox_to_anchor=[-0.15, 0.5], loc='center left',
                          borderaxespad=0)

                plt.setp(ax.legend_.get_title(), fontsize=13)
                ax.axis('off')

                ax = axes[len(stimulus_ls) + 7]
                AP = organized.loc[clustering_data_index]['AP'].to_numpy()
                scatter_df['AP'] = AP/(np.max(AP)/100)
                sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='AP',
                                palette='jet', hue_norm=AP_hue_norm, edgecolor='k', linewidth=dot_edgewidth)

                format_UMAP_plot(ax, 'Continuous_AP')

                ax = axes[len(stimulus_ls)+ 8]
                colorbar = plot_colorbar(ax, hue_nrom=AP_hue_norm, cmap='jet', orientation='vertical')
                colorbar.set_ticks(AP_color_bar_tick_labels)
                colorbar.set_ticklabels(AP_color_bar_tick_labels)

                ax = axes[len(stimulus_ls) + 9]
                colorbar = plot_colorbar(ax, hue_nrom=AP_hue_norm, cmap='jet', orientation='horizontal')
                colorbar.set_ticks(AP_color_bar_tick_labels)
                colorbar.set_ticklabels(AP_color_bar_tick_labels)


                if len(stimulus_ls) + 11 < len(axes):
                    for ax_index in np.arange(len(stimulus_ls) + 10, len(axes)):
                        ax = axes[ax_index]
                        ax.axis('off')

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                if svg:
                    plt.savefig(output_folder + '{}_{}_{}_{}_discrete.svg'.format(n_n, min_c, min_s, random_state),
                                dpi=400)

                plt.savefig(output_folder + '{}_{}_{}_{}_discrete.png'.format(n_n, min_c, min_s, random_state),
                            dpi=400)
                plt.show()
            pd.concat(
                [clustering_data[['ROI_index', 'region', 'fine_region']], pd.DataFrame(embedding, columns=['x', 'y']),
                 pd.DataFrame(cluster_label, columns=['label'])],
                axis=1).to_csv(output_folder + '{}_{}_{}_{}cluster_label.txt'.format(n_n, min_c, min_s, random_state),
                               sep='\t')

def UMAP_gen_paralle(output_folder, organized, stimulus_ls, hdbscan_para, n_n, random_state, row_num=5,
             col_num=5, average_peak=1, z_score_norm=1, svg=0, hue_norm=(-0.2, 2), AP_hue_norm=(0,100),  dot_size=5, cluster_fontsize=10,
             region_palatte_dict={}, region_type='', color_bar_tick_labels=[], AP_color_bar_tick_labels=[], hspace=3, wspace=3, dot_edgewidth=0.2,
             anyresp=1, xlim_para=[], ylim_para=[], exclude_kcl=1, UMAP_region_as_input=0):
    def format_UMAP_plot(ax, title):
        ax.set_title(label=title, fontdict={'fontsize': 20})
        ax.legend().set_visible(False)
        sns.despine(ax=ax, offset=10)
        if ylim_para:
            ylim = ylim_para[:2]
            ax.set_yticks(np.arange(ylim_para[0], ylim_para[1] + 1, ylim_para[2]))
        else:
            ylim = [math.floor(ax.get_ylim()[0]), math.ceil(ax.get_ylim()[1])]
            ax.set_yticks(np.arange(ylim[0], ylim[1] + 1))
        ax.set_ylim(ylim)

        if xlim_para:
            xlim = xlim_para[:2]
            ax.set_xticks(np.arange(xlim_para[0], xlim_para[1] + 1, xlim_para[2]))
        else:
            xlim = [math.floor(ax.get_xlim()[0]), math.ceil(ax.get_xlim()[1])]
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 1))
        ax.set_xlim(xlim)

    assert average_peak in [0, 1]
    assert row_num * col_num >= len(stimulus_ls) + 10, 'Grids not enough for all plots'


    if average_peak == 1:
        peak_type = 'average'
    elif average_peak == 0:
        peak_type = 'individual'

    try:
        if exclude_kcl:
            organized = organized[organized['KCl_{}_response'.format(peak_type)] == 1]
    except KeyError:
        pass

    clustering_data = organized[
        ['ROI_index', 'region', 'fine_region'] + ['{}_{}_peak'.format(stimulus, peak_type) for stimulus in
                                                  stimulus_ls] + [
            '{}_{}_response'.format(stimulus, peak_type)
            for stimulus in stimulus_ls]]
    if anyresp == 1:
        clustering_data = clustering_data.loc[clustering_data[
                                                  ['{}_{}_response'.format(stimulus, peak_type) for stimulus in
                                                   stimulus_ls if 'KCl' not in stimulus]].any(axis=1), :]
    clustering_data_index = clustering_data.index

    clustering_data = clustering_data.reset_index(drop=True)
    clustering_np = clustering_data[['{}_{}_peak'.format(stimulus, peak_type) for stimulus in stimulus_ls]]

    if UMAP_region_as_input:
        clustering_np['region_number'] = np.nan
        for region_ind, region in enumerate(clustering_data['region'].unique()):
            clustering_np.at[clustering_data['region'] == region, 'region_number'] = region_ind + 1

    clustering_np[['{}_{}_response'.format(stimulus, peak_type) for stimulus in stimulus_ls if 'KCl' not in stimulus]] = \
        clustering_data[
            ['{}_{}_response'.format(stimulus, peak_type) for stimulus in stimulus_ls if 'KCl' not in stimulus]]

    assert z_score_norm in [0, 1]
    if z_score_norm == 1:
        normed_data = (clustering_np - clustering_np.mean()) / clustering_np.std()
        nan_index = np.argwhere(clustering_np.std() == 0).flatten()
        clustering_np = normed_data.to_numpy()
        clustering_np[:, nan_index] = 0
    elif z_score_norm == 0:
        clustering_np = clustering_np.to_numpy()

    metric = 'euclidean'

    min_c = int(hdbscan_para[0])
    min_s = int(hdbscan_para[1])

    embedding = umap.UMAP(random_state=random_state, metric=metric, n_neighbors=n_n,
                          min_dist=0).fit_transform(clustering_np)
    scatter_df = pd.DataFrame()
    scatter_df['UMAP 1'] = embedding[:, 0]
    scatter_df['UMAP 2'] = embedding[:, 1]

    fig = plt.figure(figsize=(col_num * 4, row_num * 4))
    gs = fig.add_gridspec(row_num * 3, col_num * 3, hspace=hspace, wspace=wspace)
    axes = []
    for row in range(row_num):
        for col in range(col_num):
            if row * col_num + col == len(stimulus_ls) or row * col_num + col == len(stimulus_ls) + 8:
                axes.append(fig.add_subplot(gs[3 * row:3 * (row + 1), 3 * col:3 * col + 1]))
            elif row * col_num + col == len(stimulus_ls) + 1 or row * col_num + col == len(stimulus_ls) + 9:
                axes.append(fig.add_subplot(gs[3 * row:3 * row + 1, 3 * col:3 * (col + 1)]))
            else:
                axes.append(fig.add_subplot(gs[3 * row:3 * (row + 1), 3 * col:3 * (col + 1)]))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                        [(0, 'Blue'),
                                                         (0.35, 'Cyan'),
                                                         (0.65, 'Yellow'),
                                                         (1, 'Red')], N=126)

    for stimulus_index, ax in enumerate(axes):
        if stimulus_index > len(stimulus_ls) - 1:
            continue

        response = clustering_data[
            ['{}_{}_peak'.format(stimulus_ls[stimulus_index], peak_type)]].to_numpy().flatten()
        scatter_df['response'] = response
        sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='response',
                        palette=cmap, hue_norm=hue_norm, edgecolor='k', linewidth=dot_edgewidth)
        format_UMAP_plot(ax, stimulus_ls[stimulus_index])

    ax = axes[len(stimulus_ls)]
    colorbar = plot_colorbar(ax, hue_nrom=hue_norm, cmap=cmap, orientation='vertical')
    colorbar.set_ticks(color_bar_tick_labels)
    colorbar.set_ticklabels(color_bar_tick_labels)

    ax = axes[len(stimulus_ls) + 1]
    colorbar = plot_colorbar(ax, hue_nrom=hue_norm, cmap=cmap, orientation='horizontal')
    colorbar.set_ticks(color_bar_tick_labels)
    colorbar.set_ticklabels(color_bar_tick_labels)

    ax = axes[len(stimulus_ls) + 2]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_c, min_samples=min_s)
    clusterer.fit(embedding)
    cluster_label = clusterer.labels_ + 1
    label_num = len(np.unique(cluster_label))
    if label_num == 1:
        return
    print(label_num)
    np.random.seed(random_state)
    if label_num < 20:
        order = np.unique(cluster_label)
        np.random.shuffle(order)
        palatte = level_one_palette(cluster_label, order=order)
    else:
        hue_num = 2
        major_num = label_num // hue_num + int(label_num % hue_num != 0)
        while major_num >= 20:
            hue_num += 1
            major_num = label_num // hue_num + int(label_num % hue_num != 0)
        order = np.arange(major_num)
        np.random.shuffle(order)
        level1p = level_one_palette(range(major_num), order=order)
        palatte = level_two_palette(major_color=level1p,
                                    major_sub_dict={i: [i * hue_num + j for j in range(hue_num)] for i in
                                                    range(major_num)},
                                    major_order=order, palette='auto', skip_border_color=2)

    update_flag = 0
    for dict_label, RGBA in palatte.items():
        if np.array_equal(np.array(RGBA).round(5), np.array([0.78039216, 0.78039216, 0.78039216, 1.]).round(5)):
            temp_label = dict_label
            update_flag = 1
    if update_flag:
        palatte.update({temp_label: palatte[0], 0: [0.78039216, 0.78039216, 0.78039216, 1.]})
    else:
        palatte.update({0: [0.78039216, 0.78039216, 0.78039216, 1.]})

    scatter_df['label'] = cluster_label
    scatter_df[region_type] = organized.loc[clustering_data_index][region_type]
    sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='label',
                    palette=palatte, edgecolor='k', linewidth=dot_edgewidth)
    for label in np.unique(cluster_label):
        if label == 0:
            continue
        ax.text(np.mean(embedding[cluster_label == label, 0]),
                np.mean(embedding[cluster_label == label, 1]),
                str(label), fontsize=cluster_fontsize)
    h, l = ax.get_legend_handles_labels()
    format_UMAP_plot(ax, 'Clusters')

    ax = axes[len(stimulus_ls) + 3]
    ax.legend(h[1:], l[1:], ncol=3, fontsize=10, handletextpad=0.1, columnspacing=0.8,
              title='Cluster Label', frameon=False, bbox_to_anchor=[-0.15, 0.5], loc='center left',
              borderaxespad=0)
    plt.setp(ax.legend_.get_title(), fontsize=13)
    ax.axis('off')

    ax = axes[len(stimulus_ls) + 4]
    sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, facecolor='k', linewidth=0)
    format_UMAP_plot(ax, 'Raw')

    ax = axes[len(stimulus_ls) + 5]
    sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue=region_type,
                    palette=region_palatte_dict, edgecolor='k', linewidth=dot_edgewidth)

    h, l = ax.get_legend_handles_labels()
    format_UMAP_plot(ax, 'Discrete_AP')

    ax = axes[len(stimulus_ls) + 6]
    ax.legend(h[1:][::-1], l[1:][::-1], ncol=1, fontsize=10, handletextpad=0.1, columnspacing=0.8,
              title=region_type, frameon=False, bbox_to_anchor=[-0.15, 0.5], loc='center left',
              borderaxespad=0)

    plt.setp(ax.legend_.get_title(), fontsize=13)
    ax.axis('off')

    ax = axes[len(stimulus_ls) + 7]
    AP = organized.loc[clustering_data_index]['AP'].to_numpy()
    scatter_df['AP'] = AP/(np.max(AP)/100)
    sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='AP',
                    palette='jet', hue_norm=AP_hue_norm, edgecolor='k', linewidth=dot_edgewidth)

    format_UMAP_plot(ax, 'Continuous_AP')

    ax = axes[len(stimulus_ls)+ 8]
    colorbar = plot_colorbar(ax, hue_nrom=AP_hue_norm, cmap='jet', orientation='vertical')
    colorbar.set_ticks(AP_color_bar_tick_labels)
    colorbar.set_ticklabels(AP_color_bar_tick_labels)

    ax = axes[len(stimulus_ls) + 9]
    colorbar = plot_colorbar(ax, hue_nrom=AP_hue_norm, cmap='jet', orientation='horizontal')
    colorbar.set_ticks(AP_color_bar_tick_labels)
    colorbar.set_ticklabels(AP_color_bar_tick_labels)


    if len(stimulus_ls) + 11 < len(axes):
        for ax_index in np.arange(len(stimulus_ls) + 10, len(axes)):
            ax = axes[ax_index]
            ax.axis('off')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if svg:
        plt.savefig(output_folder + '{}_{}_{}_{}_discrete.svg'.format(n_n, min_c, min_s, random_state),
                    dpi=400)

    plt.savefig(output_folder + '{}_{}_{}_{}_discrete.png'.format(n_n, min_c, min_s, random_state),
                dpi=400)
    plt.show()
    pd.concat(
        [clustering_data[['ROI_index', 'region', 'fine_region']], pd.DataFrame(embedding, columns=['x', 'y']),
         pd.DataFrame(cluster_label, columns=['label'])],
        axis=1).to_csv(output_folder + '{}_{}_{}_{}cluster_label.txt'.format(n_n, min_c, min_s, random_state),
                       sep='\t')