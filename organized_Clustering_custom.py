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
from scipy.stats import zscore, pearsonr
from DataOrganizer_custom import data_plus_label, trace_for_lineplot_stimulus
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn import decomposition
from sklearn.cluster import AgglomerativeClustering
from DataPlotter import plot_cluster_trace_grid_new, plot_cluster_trace_grid_cell_average
from scipy.cluster.hierarchy import dendrogram, ward, fcluster
from scipy.spatial.distance import pdist

matplotlib.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 13})


def factors(value):
    factors = []
    for i in range(1, int(value ** 0.5) + 1):
        if value % i == 0:
            factors.append((i, value / i))
    return factors


def UMAP_gen(output_folder, organized, stimulus_ls, HDBSCAN_para_ls, UMAP_nearest_neighbor_ls, random_state_ls, row_num,
             col_num, peak_type='average', z_score_norm=1, svg=0, hue_norm=(-0.2, 2), AP_hue_norm=(0, 100), dot_size=5,
             cluster_fontsize=10,
             region_palatte_dict={}, region_type='', color_bar_tick_labels=[], AP_color_bar_tick_labels=[], hspace=3,
             wspace=3, dot_edgewidth=0.2,
             anyresp=1, xlim_para=[], ylim_para=[], cluster_palatte_dict={}, exclude_kcl=1, UMAP_region_as_input=0,
             dynamics_dff_peak=0, dynamics_dff_peak_time=0, dynamics_dff_prime_peak=0, dynamics_dff_prime_peak_time=0):
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

    assert peak_type in ['average', 'individual', 'region', 'fine_region', 'custom']
    assert row_num * col_num >= len(stimulus_ls) + 10, 'Grids not enough for all plots'

    if exclude_kcl and 'KCl' in stimulus_ls:
        organized = organized[organized['KCl_{}_response'.format(peak_type)] == 1]

    input_ls = ['ROI_index', 'region', 'fine_region'] + ['{}_{}_peak'.format(stimulus, peak_type) for stimulus in
                                                         stimulus_ls] + ['{}_{}_response'.format(stimulus, peak_type)
                                                                         for stimulus in stimulus_ls]

    if dynamics_dff_peak:
        if peak_type != 'individual':
            input_ls += ['{}_individual_peak'.format(stimulus) for stimulus in stimulus_ls]

    if dynamics_dff_peak_time:
        input_ls += ['{}_individual_peak_index'.format(stimulus) for stimulus in stimulus_ls]

    if dynamics_dff_prime_peak:
        input_ls += ['{}_max_first_derivative'.format(stimulus) for stimulus in stimulus_ls]

    if dynamics_dff_prime_peak_time:
        input_ls += ['{}_max_first_derivative_index'.format(stimulus) for stimulus in stimulus_ls]

    clustering_data = organized[(~organized["ROI_index"].str.contains('_b'))][input_ls]
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
                scatter_df[region_type] = organized.loc[clustering_data_index][region_type].reset_index(drop=True)
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
                scatter_df['AP'] = AP / (np.max(AP) / 100)
                sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='AP',
                                palette='jet', hue_norm=AP_hue_norm, edgecolor='k', linewidth=dot_edgewidth)

                format_UMAP_plot(ax, 'Continuous_AP')

                ax = axes[len(stimulus_ls) + 8]
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
                    plt.savefig(
                        output_folder + '{}_clusters_{}_{}_{}_{}_discrete.svg'.format(label_num, n_n, min_c, min_s,
                                                                                      random_state),
                        dpi=400)

                plt.savefig(output_folder + '{}_clusters_{}_{}_{}_{}_discrete.png'.format(label_num, n_n, min_c, min_s,
                                                                                          random_state),
                            dpi=400)
                plt.show()
            pd.concat(
                [clustering_data[['ROI_index', 'region', 'fine_region']], pd.DataFrame(embedding, columns=['x', 'y']),
                 pd.DataFrame(cluster_label, columns=['label'])],
                axis=1).to_csv(
                output_folder + '{}_clusters_{}_{}_{}_{}cluster_label.txt'.format(label_num, n_n, min_c, min_s,
                                                                                  random_state),
                sep='\t')


def UMAP_gen_paralle(output_folder, organized, stimulus_ls, hdbscan_para, n_n, random_state, row_num=5,
                     col_num=5, peak_type='average', z_score_norm=1, svg=0, hue_norm=(-0.2, 2), AP_hue_norm=(0, 100),
                     dot_size=5, cluster_fontsize=10,
                     region_palatte_dict={}, region_type='', color_bar_tick_labels=[], AP_color_bar_tick_labels=[],
                     hspace=3, wspace=3, dot_edgewidth=0.2,
                     anyresp=1, xlim_para=[], ylim_para=[], exclude_kcl=1, UMAP_region_as_input=0, dynamics_dff_peak=0,
                     dynamics_dff_peak_time=0, dynamics_dff_prime_peak=0, dynamics_dff_prime_peak_time=0, num_screen=0):
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

    assert peak_type in ['average', 'individual', 'region', 'fine_region', 'custom']
    assert row_num * col_num >= len(stimulus_ls) + 10, 'Grids not enough for all plots'

    try:
        if exclude_kcl:
            organized = organized[organized['KCl_{}_response'.format(peak_type)] == 1]
    except KeyError:
        pass

    input_ls = ['ROI_index', 'region', 'fine_region'] + ['{}_{}_peak'.format(stimulus, peak_type) for stimulus in
                                                         stimulus_ls] + ['{}_{}_response'.format(stimulus, peak_type)
                                                                         for stimulus in stimulus_ls]

    if dynamics_dff_peak:
        if peak_type != 'individual':
            input_ls += ['{}_individual_peak'.format(stimulus) for stimulus in stimulus_ls]

    if dynamics_dff_peak_time:
        input_ls += ['{}_individual_peak_index'.format(stimulus) for stimulus in stimulus_ls]

    if dynamics_dff_prime_peak:
        input_ls += ['{}_max_first_derivative'.format(stimulus) for stimulus in stimulus_ls]

    if dynamics_dff_prime_peak_time:
        input_ls += ['{}_max_first_derivative_index'.format(stimulus) for stimulus in stimulus_ls]

    clustering_data = organized[(~organized["ROI_index"].str.contains('_b'))][input_ls]
    if anyresp == 1:
        clustering_data = clustering_data.loc[clustering_data[
                                                  ['{}_{}_response'.format(stimulus, peak_type) for stimulus in
                                                   stimulus_ls if 'KCl' not in stimulus]].any(axis=1), :]
    clustering_data_index = clustering_data.index

    clustering_data = clustering_data.reset_index(drop=True)
    # clustering_np = clustering_data[['{}_{}_peak'.format(stimulus, peak_type) for stimulus in stimulus_ls]]

    input_ls.remove('ROI_index')
    input_ls.remove('region')
    input_ls.remove('fine_region')

    # input_ls = []
    for stimulus in stimulus_ls:
        clustering_data['{}_{}_peak_response'.format(stimulus, peak_type)] = clustering_data[
                                                                                 ['{}_{}_peak'.format(stimulus,
                                                                                                      peak_type)]].to_numpy().flatten() * \
                                                                             clustering_data[
                                                                                 ['{}_{}_response'.format(stimulus,
                                                                                                          peak_type)]].to_numpy().flatten()
        input_ls.append('{}_{}_peak_response'.format(stimulus, peak_type))
        input_ls.remove('{}_{}_peak'.format(stimulus,peak_type))
        input_ls.remove('{}_{}_response'.format(stimulus, peak_type))

    clustering_np = clustering_data[input_ls]

    if UMAP_region_as_input:
        clustering_np['region_number'] = np.nan
        for region_ind, region in enumerate(clustering_data['region'].unique()):
            clustering_np.at[clustering_data['region'] == region, 'region_number'] = region_ind + 1

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

    def get_flattened_half_matrix(mx):
        ls = []
        for i in range(mx.shape[0]):
            for j in range(mx.shape[1]):
                if j > i:
                    ls.append(mx[i, j])
        return np.array(ls)

    raw_distance_mx = pairwise_distances(clustering_np)
    umap_distance_mx = pairwise_distances(embedding)

    pairwise_corr = pearsonr(get_flattened_half_matrix(raw_distance_mx), get_flattened_half_matrix(umap_distance_mx))[0]

    scatter_df = pd.DataFrame()
    scatter_df['UMAP 1'] = embedding[:, 0]
    scatter_df['UMAP 2'] = embedding[:, 1]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_c, min_samples=min_s)
    clusterer.fit(embedding)
    cluster_label = clusterer.labels_ + 1
    label_num = len(np.unique(cluster_label))
    outlier_perc = np.round(len(cluster_label[cluster_label==0])/len(cluster_label), 3)
    print('{} labels'.format(label_num))
    if label_num > 1:
        silhouette = silhouette_score(clustering_np, clusterer.labels_, metric='euclidean')
    else:
        silhouette = -999

    if num_screen:
        return {'pairwise_corr':pairwise_corr, 'silhouette_score':silhouette,'label_num':label_num, 'outlier_perc':outlier_perc}

    cluster_df = pd.concat(
        [clustering_data[['ROI_index', 'region', 'fine_region']], pd.DataFrame(embedding, columns=['x', 'y']),
         pd.DataFrame(cluster_label, columns=['label'])],
        axis=1)

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

        peak_with_response = clustering_data[
                                 ['{}_{}_peak'.format(stimulus_ls[stimulus_index], peak_type)]].to_numpy().flatten() * \
                             clustering_data[
                                 ['{}_{}_response'.format(stimulus_ls[stimulus_index], peak_type)]].to_numpy().flatten()
        visualize_values = peak_with_response.copy()
        visualize_values[peak_with_response != 0] = zscore(peak_with_response[peak_with_response != 0])
        visualize_values[peak_with_response == 0] = np.nan
        scatter_df['visualize_response'] = visualize_values
        response_scatter_df = scatter_df[~scatter_df.isnull().any(axis=1)]
        non_response_scatter_df = scatter_df[scatter_df.isnull().any(axis=1)]

        # sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='response',
        #                 palette=cmap, hue_norm=hue_norm, edgecolor='k', linewidth=dot_edgewidth)
        sns.scatterplot(data=response_scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='visualize_response',
                        palette=cmap, hue_norm=hue_norm, edgecolor='k', linewidth=dot_edgewidth)
        sns.scatterplot(data=non_response_scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, color='k',
                        edgecolor='k', linewidth=dot_edgewidth)
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
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=min_c, min_samples=min_s)
    # clusterer.fit(embedding)
    # cluster_label = clusterer.labels_ + 1
    # label_num = len(np.unique(cluster_label))
    # print('{} labels'.format(label_num))
    # if label_num == 1:
    #     print('Only one cluster. Skipped.')
    #     return
    #
    # cluster_df = pd.concat(
    #     [clustering_data[['ROI_index', 'region', 'fine_region']], pd.DataFrame(embedding, columns=['x', 'y']),
    #      pd.DataFrame(cluster_label, columns=['label'])],
    #     axis=1)

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
    scatter_df[region_type] = organized.loc[clustering_data_index][region_type].reset_index(drop=True)
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
    scatter_df['AP'] = AP / (np.max(AP) / 100)
    sns.scatterplot(data=scatter_df, x='UMAP 1', y='UMAP 2', ax=ax, s=dot_size, hue='AP',
                    palette='jet', hue_norm=AP_hue_norm, edgecolor='k', linewidth=dot_edgewidth)

    format_UMAP_plot(ax, 'Continuous_AP')

    ax = axes[len(stimulus_ls) + 8]
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
        plt.savefig(
            output_folder + '{}_clusters_{}_{}_{}_{}_discrete.svg'.format(label_num, n_n, min_c, min_s, random_state),
            dpi=400)

    plt.savefig(
        output_folder + '{}_clusters_{}_{}_{}_{}_discrete.png'.format(label_num, n_n, min_c, min_s, random_state),
        dpi=400)
    plt.show()
    cluster_df.to_csv(
        output_folder + '{}_clusters_{}_{}_{}_{}cluster_label.txt'.format(label_num, n_n, min_c, min_s, random_state),
        sep='\t')
    return [hdbscan_para, n_n, random_state]


def raw_pca(organized, input_folder, stimulus_ls):
    def segment_trace(trace, start, end):
        return trace.values[start:end]

    def plot_heat_dendro(all_trace_np, plot_trace, n_pc, reduced, range, ax1, ax2):
        if n_pc == -1:
            n_pc == np.min(all_trace_np.shape)
        pca = decomposition.PCA(n_components=n_pc)
        kept_pc = pca.fit_transform(all_trace_np)
        reduced_trace = pca.inverse_transform(kept_pc)

        # kept_pc_df = pd.DataFrame(kept_pc)
        # reduced_trace_df = pd.DataFrame(reduced_trace)
        # organized['kept_pc'] = kept_pc_df.apply(segment_trace, axis=1, args=(0, kept_pc.shape[1]))
        # organized['reduced_trace'] = reduced_trace_df.apply(segment_trace, axis=1, args=(0, reduced_trace.shape[1]))

        # y = pdist(kept_pc)
        y = pdist(reduced_trace, 'correlation')
        Z = ward(y)
        R = dendrogram(Z, orientation='right', color_threshold=100, distance_sort='descending', no_labels=True,
                       ax=ax2)
        dendro_label = R['leaves'][::-1]
        # labels = fcluster(Z, n_cluster, criterion='maxclust')
        if reduced:
            ax1.imshow(reduced_trace[dendro_label, :], cmap='jet', vmin=range[0], vmax=range[1], aspect='auto')
        else:
            ax1.imshow(plot_trace[dendro_label, :], cmap='jet', vmin=range[0], vmax=range[1], aspect='auto')

    def norm_combine_traces(organized, stimulus_ls):
        norm_concat_ls = []
        concat_ls = []
        organized = organized.loc[organized_copy[['{}_individual_response'.format(stimulus) for stimulus in
                                                       stimulus_ls if 'KCl' not in stimulus]].any(axis=1),
                    :].reset_index(drop=True)
        for stimulus in stimulus_ls:
            sti_np = np.concatenate(organized['{}_trace'.format(stimulus)].to_numpy()).reshape(
                organized.shape[0], -1)
            concat_ls.append(sti_np)
            norm_sti_np = sti_np / np.max(sti_np.flatten())
            norm_concat_ls.append(norm_sti_np)
        return np.concatenate(concat_ls, axis=1), np.concatenate(norm_concat_ls, axis=1)

    organized_copy = organized.copy()
    old_stimulus_ls = stimulus_ls.copy()
    stimulus_ls.append('all')
    stimulus_ls.append('AA')
    for stimulus_name in stimulus_ls:
        output_folder = input_folder+'h_c_{}\\'.format(stimulus_name)

        if stimulus_name == 'all':
            # organized = organized_copy.loc[organized_copy[['{}_individual_response'.format(stimulus) for stimulus in
            #                                      old_stimulus_ls if 'KCl' not in stimulus]].any(axis=1), :].reset_index(
            #     drop=True)
            # all_trace_np = np.concatenate(organized['all_trace'].to_numpy()).reshape(organized.shape[0], -1)
            all_trace_np, norm_all_trace_np = norm_combine_traces(organized_copy, old_stimulus_ls)
        elif stimulus_name == 'AA':
            all_trace_np, norm_all_trace_np = norm_combine_traces(organized_copy, ['BCAA','EAA', 'NEAA'])
        else:
            organized = organized_copy.loc[organized_copy['{}_individual_response'.format(stimulus_name)] == 1, :].reset_index(drop=True)
            all_trace_np = np.concatenate(organized['{}_trace'.format(stimulus_name)].to_numpy()).reshape(organized.shape[0], -1)

        pca = decomposition.PCA().fit(all_trace_np)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        pc_range = np.array(np.argwhere(np.logical_and(explained_variance>0.8, explained_variance<0.95))).flatten()
        cum_e_v = explained_variance[pc_range]
        n_pc_ls = zip(pc_range + 1, np.round(cum_e_v,2))

        for n_pc in n_pc_ls:
            e_v = n_pc[1]
            n_pc = n_pc[0]
            wspace = 0.03
            axes = []

            fig = plt.figure(figsize=(10, 6))
            gs_ls = gridspec.GridSpec(nrows=2, ncols=5, width_ratios=[4, 1, 0.5, 4, 1], wspace=wspace)
            for gs in gs_ls:
                axes.append(plt.subplot(gs))

            plt.tight_layout()

            if stimulus_name == 'all' or stimulus_name == 'AA':
                plot_heat_dendro(norm_all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, reduced=1, range=[-0.2,0.2], ax1=axes[0], ax2=axes[1])
                plot_heat_dendro(norm_all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, reduced=0, range=[-0.2,2], ax1=axes[3], ax2=axes[4])
                plot_heat_dendro(norm_all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, reduced=1, range=[-0.2, 0.8], ax1=axes[5], ax2=axes[6])
                plot_heat_dendro(norm_all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, reduced=0, range=[-0.2, 8], ax1=axes[8], ax2=axes[9])
            else:
                plot_heat_dendro(all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, reduced=1, range=[-0.2, 2],
                                 ax1=axes[0], ax2=axes[1])
                plot_heat_dendro(all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, reduced=0, range=[-0.2, 2],
                                 ax1=axes[3], ax2=axes[4])
                plot_heat_dendro(all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, reduced=1, range=[-0.2, 8],
                                 ax1=axes[5], ax2=axes[6])
                plot_heat_dendro(all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, reduced=0, range=[-0.2, 8],
                                 ax1=axes[8], ax2=axes[9])

            axes[2].axis('off')
            axes[7].axis('off')
            # plt.plot(np.cumsum(pca.explained_variance_ratio_))
            # plt.xlabel('number of components')
            # plt.ylabel('cumulative explained variance')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.savefig(output_folder+'pc_{}_explained_var_{}.png'.format(n_pc, e_v))
            plt.clf()

            # Hierarchical_Clustering_Heatmap_Dendro(all_trace_np, dendro_label)
            # Hierarchical_Clustering_Heatmap_Dendro(reduced_trace, dendro_label)

    return

def h_cluster_multi(organized_ls, input_folder, scheme, stimulus_ls, threshold,n_pc_ls, norm=0):
    def plot_heat_dendro(all_trace_np, plot_trace, genotype_index, n_pc, range, ax1, ax2):
        if n_pc == -1:
            n_pc == np.min(all_trace_np.shape)
        pca = decomposition.PCA(n_components=n_pc, random_state=0)
        kept_pc = pca.fit_transform(all_trace_np)
        reduced_trace = pca.inverse_transform(kept_pc)
        y = pdist(reduced_trace, 'correlation')
        Z = ward(y)
        R = dendrogram(Z, orientation='right', color_threshold=threshold, distance_sort='descending', no_labels=True,
                       ax=ax2)
        dendro_label = R['leaves'][::-1]
        labels = fcluster(Z, threshold, criterion='distance')
        ax1.imshow(plot_trace[dendro_label, :], cmap='jet', vmin=range[0], vmax=range[1], aspect='auto', interpolation='none')
        return labels

    def norm_combine_traces(organized, stimulus_ls, norm):
        norm_concat_ls = []
        concat_ls = []
        for stimulus in stimulus_ls:
            sti_np = np.concatenate(organized['{}_trace'.format(stimulus)].to_numpy()).reshape(
                organized.shape[0], -1)
            concat_ls.append(sti_np)
            norm_sti_np = sti_np / np.max(sti_np.flatten())
            norm_concat_ls.append(norm_sti_np)
        if norm:
            return np.concatenate(concat_ls, axis=1), np.concatenate(norm_concat_ls, axis=1)
        else:
            return np.concatenate(concat_ls, axis=1), np.concatenate(concat_ls, axis=1)

    organized = pd.concat(organized_ls,axis=0).reset_index(drop=True).copy()
    organized = organized[~organized["ROI_index"].str.contains('_b')].reset_index(drop=True).copy()
    stimulus_name = 'all'

    stimulus_ls.remove('AHL')
    # stimulus_ls.remove('BCAA')
    organized = organized.loc[organized[['{}_individual_response'.format(stimulus) for stimulus in
                                              stimulus_ls if 'KCl' not in stimulus]].any(axis=1),
                :].reset_index(drop=True)
    genotype_index = organized['genotype']
    all_trace_np, norm_all_trace_np = norm_combine_traces(organized, stimulus_ls, norm)

    pca = decomposition.PCA().fit(all_trace_np)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    # n_pc_ls = range(5,20)
    for n_pc in n_pc_ls:
        output_folder = input_folder + 'h_c_{}\\{}\\'.format(stimulus_name, n_pc)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        e_v = np.round(explained_variance[n_pc-1], 2)

        wspace = 0.03
        axes = []

        fig = plt.figure(figsize=(10, 6))
        gs_ls = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[4, 1], wspace=wspace)
        for gs in gs_ls:
            axes.append(plt.subplot(gs))

        labels = plot_heat_dendro(norm_all_trace_np, plot_trace=all_trace_np, n_pc=n_pc, range=[-0.5, 5],
                         ax1=axes[0], ax2=axes[1], genotype_index=genotype_index)


        plt.savefig(output_folder+'pc_{}_explained_var_{}.png'.format(n_pc, e_v))
        plt.savefig(output_folder + 'pc_{}_explained_var_{}.svg'.format(n_pc, e_v))
        plt.clf()
        plt.close()

        organized['label'] = labels

        organized[['sample_index', 'ROI_index', 'genotype', 'label']].to_excel(output_folder + 'ROI_names.xlsx')
        # exit()

        row_label = 'label'
        for genotype in organized['genotype'].unique():
            plot_org = organized[organized['genotype'] == genotype]
            plot_cluster_trace_grid_cell_average(output_folder, plot_org, scheme, row_label, stimulus_ls, ylim=[-1, 7],
                                        linewidth=3, color='k', height=10, width=10, svg=1, cluster_order=[], stimulus_bar_y=-1,
                                        stimulus_bar_lw=20, scalebar_frame_length=10, scalebar_amplitude_length=1,
                                        min_cell_per_sample=2, plot_name = genotype)

            # exit()

        stat_org = organized[['sample_index', 'region', 'fine_region', 'genotype', 'label']]
        for genotype in organized['genotype'].unique():
            out_df = pd.DataFrame(index=range(1, len(stat_org['label'].unique())))
            for sample_index in organized['sample_index'].unique():
                # out_df = pd.DataFrame(index=range(1, 12))
                genotype_df = stat_org[(stat_org['genotype'] == genotype) & (stat_org['sample_index'] == sample_index)]
                if genotype_df.shape[0] == 0:
                    continue

                temp_df = genotype_df[['genotype', 'label']]
                gb_out = temp_df.groupby(['label']).agg(['count'])
                # gb_out.columns = ['cluster_count']
                gb_out.columns = ['{}_cluster_count'.format(sample_index)]
                out_df = pd.concat((out_df, gb_out), axis=1)
                out_df = out_df.fillna(0)
                out_df['{}_cluster_percentage'.format(sample_index)] = np.round(
                    out_df['{}_cluster_count'.format(sample_index)].values / out_df[
                        '{}_cluster_count'.format(sample_index)].sum(), 2)

                if genotype == 'Pros':
                    for region in organized['region'].unique():
                        temp_df = genotype_df[genotype_df['region'] == region][['region', 'label']]
                        temp_gb_out = temp_df.groupby(['label']).agg(['count'])
                        temp_gb_out.columns = ['{}_{}'.format(sample_index, region)]
                        out_df = pd.concat((out_df,temp_gb_out),axis=1)
                out_df = out_df.fillna(0)
            out_df.to_csv(output_folder + '{}.csv'.format(genotype))
    return