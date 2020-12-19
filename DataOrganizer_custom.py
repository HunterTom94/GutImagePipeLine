import pandas as pd
import numpy as np
from time import time
import os
from util import low_pass_filter, mean_confidence_interval
from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt
import pickle
from sys import exit


def roiName2trace(organized, roi_name_df):
    roi_name_df.columns = ['ROI_index']
    organized = organized.loc[organized['ROI_index'].isin(roi_name_df['ROI_index']), :]
    organized_index = organized['ROI_index'].to_list()
    organized = organized.set_index('ROI_index')
    organized['ROI_index'] = organized_index
    organized = organized.reindex(roi_name_df['ROI_index'].to_list())['all_trace']
    trace_df = pd.DataFrame(np.concatenate(organized.to_numpy()).reshape(organized.shape[0], -1))
    trace_df['ROI_index'] = roi_name_df
    trace_df = trace_df.set_index('ROI_index').transpose()
    return trace_df


def DataOrganizer_updated_scheme(name, scheme, save_dir, all_data, region_data, custom_peak=None):
    std_scale = 3

    def sample_std(array):
        mean = np.mean(array)
        sq_diff = np.power(array - mean, 2)
        n_1_ave = np.sum(sq_diff) / (len(array) - 1)
        return np.sqrt(n_1_ave)

    def segment_trace(trace, start, end):
        return trace.values[start:end]

    def find_peak(trace):
        if np.isnan(trace[-1]):
            return np.nan
        max_ind = int(trace[-1])
        trace = trace[:-1]
        return np.mean(trace[max_ind - 1:max_ind + 2])

    def find_peak_index(trace, start, end):
        max_ind = np.argmax(trace[start:end]) + start
        return max_ind

    def find_f0_mean(row, start, end):
        return np.mean(row[start:end])

    def find_f0_std(row, start, end):
        return sample_std(row[start:end])

    def if_response(row):
        mean = row[0]
        std = row[1]
        peak = row[2]
        threshold = row[3]

        if peak - mean > std_scale * std and peak > threshold:
            return 1
        else:
            return 0

    def find_max_first_derivative(row, start, end):
        filtered = low_pass_filter(row, 0.25, 0.03)[start:end]
        return [np.argmax(np.gradient(filtered)) + start, np.max(np.gradient(filtered))]

    def find_max_second_derivative(row):
        filtered = low_pass_filter(row, 0.25, 0.03)
        return [np.argmax(np.gradient(np.gradient(filtered))), np.max(np.gradient(np.gradient(filtered)))]

    start_t = time()

    out_df = pd.DataFrame()

    out_df['sample_index'] = [file.split('_')[0] for file in all_data['ROI_index']]
    out_df['ROI_index'] = all_data['ROI_index']
    out_df['all_trace'] = all_data.drop(['ROI_index'], axis=1).apply(segment_trace, axis=1, args=(0, all_data.shape[1]))

    if not isinstance(region_data, int):
        region_data = region_data.iloc[:, :4]
        region_data.columns = ['ROI_index', 'AP', 'NormAP', 'fine_region']
        region_data['region'] = [str(region)[:2] for region in region_data['fine_region']]
        region_data['ROI_index'] = [str(roi_index)[1:] for roi_index in region_data['ROI_index']]
        region_data['AP'] = region_data['AP'] * 10
        out_df = pd.merge(out_df, region_data, on='ROI_index', how='left')

    for stimulus_ind in range(scheme.shape[0]):
        out_df['{}_trace'.format(scheme['stimulation'].iloc[stimulus_ind])] = all_data.drop(['ROI_index'],
                                                                                            axis=1).apply(segment_trace,
                                                                                                          axis=1, args=(
                scheme['video_start'].iloc[stimulus_ind] - 1, scheme['video_end'].iloc[stimulus_ind]))

    if custom_peak is not None:
        out_df = out_df.merge(custom_peak, how='outer', on='ROI_index')

    for stimulus_ind, stimulus in enumerate(scheme['stimulation']):
        stimulus_np = np.concatenate(out_df['{}_trace'.format(stimulus)].to_numpy()).reshape(-1,
                                                                                             scheme['video_end'].iloc[
                                                                                                 stimulus_ind] -
                                                                                             scheme['video_start'].iloc[
                                                                                                 stimulus_ind] + 1)
        individual_peak_index = np.apply_along_axis(find_peak_index, 1, stimulus_np,
                                                    scheme['peak_start'].iloc[stimulus_ind] -
                                                    scheme['video_start'].iloc[stimulus_ind],
                                                    scheme['peak_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[
                                                        stimulus_ind] + 1)
        out_df['{}_individual_peak_index'.format(stimulus)] = individual_peak_index
        individual_peak = np.apply_along_axis(find_peak, 1, np.concatenate((stimulus_np, individual_peak_index.reshape(
            out_df.shape[0],
            1)), axis=1))
        out_df['{}_individual_peak'.format(stimulus)] = individual_peak

        out_df['{}_region_peak_index'.format(stimulus)] = np.nan
        for region in out_df['region'].unique():
            sample_mean = np.zeros((len(out_df['sample_index'].unique()), stimulus_np.shape[1]))
            for sample_ind, sample in enumerate(out_df['sample_index'].unique()):
                sample_index = np.array(
                    out_df[(out_df['sample_index'] == sample) & (~out_df["ROI_index"].str.contains('_b')) & (
                                out_df['region'] == region)].index)
                sample_mean[sample_ind, :] = np.mean(stimulus_np[sample_index, :], axis=0)

            average_peak_index = np.apply_along_axis(find_peak_index, 0, np.mean(sample_mean, axis=0),
                                                     scheme['peak_start'].iloc[stimulus_ind] -
                                                     scheme['video_start'].iloc[stimulus_ind],
                                                     scheme['peak_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[
                                                         stimulus_ind] + 1)
            out_df.at[np.array(
                out_df[(~out_df["ROI_index"].str.contains('_b')) & (
                            out_df['region'] == region)].index), '{}_region_peak_index'.format(
                stimulus)] = average_peak_index

        region_peak_index = out_df['{}_region_peak_index'.format(stimulus)].to_numpy()
        region_peak = np.apply_along_axis(find_peak, 1, np.concatenate(
            (stimulus_np, region_peak_index.reshape(out_df.shape[0], 1)), axis=1))
        out_df['{}_region_peak'.format(stimulus)] = region_peak

        out_df['{}_fine_region_peak_index'.format(stimulus)] = np.nan
        for region in out_df['fine_region'].unique():
            sample_mean = np.zeros((len(out_df['sample_index'].unique()), stimulus_np.shape[1]))
            for sample_ind, sample in enumerate(out_df['sample_index'].unique()):
                sample_index = np.array(
                    out_df[(out_df['sample_index'] == sample) & (~out_df["ROI_index"].str.contains('_b')) & (
                            out_df['fine_region'] == region)].index)
                sample_mean[sample_ind, :] = np.mean(stimulus_np[sample_index, :], axis=0)

            average_peak_index = np.apply_along_axis(find_peak_index, 0, np.mean(sample_mean, axis=0),
                                                     scheme['peak_start'].iloc[stimulus_ind] -
                                                     scheme['video_start'].iloc[stimulus_ind],
                                                     scheme['peak_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[
                                                         stimulus_ind] + 1)
            out_df.at[np.array(
                out_df[(~out_df["ROI_index"].str.contains('_b')) & (
                        out_df['fine_region'] == region)].index), '{}_fine_region_peak_index'.format(
                stimulus)] = average_peak_index

        fine_region_peak_index = out_df['{}_fine_region_peak_index'.format(stimulus)].to_numpy()
        fine_region_peak = np.apply_along_axis(find_peak, 1, np.concatenate(
            (stimulus_np, fine_region_peak_index.reshape(out_df.shape[0], 1)), axis=1))
        out_df['{}_fine_region_peak'.format(stimulus)] = fine_region_peak

        sample_mean = np.zeros((len(out_df['sample_index'].unique()), stimulus_np.shape[1]))
        for sample_ind, sample in enumerate(out_df['sample_index'].unique()):
            sample_index = np.array(
                out_df[(out_df['sample_index'] == sample) & (~out_df["ROI_index"].str.contains('_b'))].index)
            sample_mean[sample_ind, :] = np.mean(stimulus_np[sample_index, :], axis=0)

        average_peak_index = np.apply_along_axis(find_peak_index, 0, np.mean(sample_mean, axis=0),
                                                 scheme['peak_start'].iloc[stimulus_ind] -
                                                 scheme['video_start'].iloc[stimulus_ind],
                                                 scheme['peak_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[
                                                     stimulus_ind] + 1)
        out_df['{}_average_peak_index'.format(stimulus)] = average_peak_index
        average_peak = np.apply_along_axis(find_peak, 1, np.concatenate(
            (stimulus_np, np.array([average_peak_index] * out_df.shape[0]).reshape(out_df.shape[0], 1)), axis=1))
        out_df['{}_average_peak'.format(stimulus)] = average_peak

        temp_m = np.mean(out_df[out_df["ROI_index"].str.contains('_b')]['{}_individual_peak'.format(stimulus)])
        temp_std = np.std(out_df[out_df["ROI_index"].str.contains('_b')]['{}_individual_peak'.format(stimulus)])
        out_df['{}_background_threshold'.format(stimulus)] = temp_m + temp_std * 3

        f0_mean = np.apply_along_axis(find_f0_mean, 1, stimulus_np, scheme['F0_start'].iloc[stimulus_ind] -
                                      scheme['video_start'].iloc[stimulus_ind],
                                      scheme['F0_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[
                                          stimulus_ind] + 1)
        out_df['{}_f0_mean'.format(stimulus)] = f0_mean
        f0_std = np.apply_along_axis(find_f0_std, 1, stimulus_np, scheme['F0_start'].iloc[stimulus_ind] -
                                     scheme['video_start'].iloc[stimulus_ind],
                                     scheme['F0_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[
                                         stimulus_ind] + 1)
        out_df['{}_f0_std'.format(stimulus)] = f0_std

        max_first_derivative_result = np.apply_along_axis(find_max_first_derivative, 1, stimulus_np, scheme['peak_start'].iloc[stimulus_ind] - scheme['video_start'].iloc[stimulus_ind],
                             scheme['peak_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[stimulus_ind] + 1)
        out_df['{}_max_first_derivative_index'.format(stimulus)] = max_first_derivative_result[:, 0]
        out_df['{}_max_first_derivative'.format(stimulus)] = max_first_derivative_result[:, 1]
        max_second_derivative_result = np.apply_along_axis(find_max_second_derivative, 1, stimulus_np)
        out_df['{}_max_second_derivative_index'.format(stimulus)] = max_second_derivative_result[:, 0]
        out_df['{}_max_second_derivative'.format(stimulus)] = max_second_derivative_result[:, 1]

        out_df['{}_individual_response'.format(stimulus)] = np.apply_along_axis(if_response, 1, np.stack(
            (f0_mean, f0_std, individual_peak, out_df['{}_background_threshold'.format(stimulus)]), axis=1))
        out_df['{}_average_response'.format(stimulus)] = np.apply_along_axis(if_response, 1, np.stack((f0_mean, f0_std,
                                                                                                       average_peak,
                                                                                                       out_df[
                                                                                                           '{}_background_threshold'.format(
                                                                                                               stimulus)]),
                                                                                                      axis=1))
        if custom_peak is not None:
            custom_peak = out_df['{}_custom_peak'.format(stimulus)].to_numpy()
            out_df['{}_custom_response'.format(stimulus)] = np.apply_along_axis(if_response, 1, np.stack(
                (f0_mean, f0_std, custom_peak, out_df['{}_background_threshold'.format(stimulus)]), axis=1))
        out_df['{}_region_response'.format(stimulus)] = np.apply_along_axis(if_response, 1, np.stack(
            (f0_mean, f0_std, region_peak, out_df['{}_background_threshold'.format(stimulus)]), axis=1))
        out_df['{}_fine_region_response'.format(stimulus)] = np.apply_along_axis(if_response, 1, np.stack(
            (f0_mean, f0_std, fine_region_peak, out_df['{}_background_threshold'.format(stimulus)]), axis=1))

    original_index = out_df.columns
    temp_index = []
    stimulus_suffix = []
    for index in original_index:
        if any(stimulus in index for stimulus in scheme['stimulation']):
            stimulus_suffix.append('_'.join(index.split('_')[1:]))
        else:
            temp_index.append(index)
    stimulus_suffix = np.unique(np.array(stimulus_suffix)).tolist()
    for suffix in stimulus_suffix:
        for stimulus in scheme['stimulation']:
            temp_index.append('{}_{}'.format(stimulus, suffix))
    target_order = temp_index
    out_df = out_df[target_order]

    out_df.to_csv(save_dir + name + '_organized_data.csv')
    out_df.to_pickle(save_dir + name + '_organized_data.pkl')
    print('{} ROIs'.format(out_df[~out_df["ROI_index"].str.contains('_b')].shape[0]))
    print('Time elapsed: {} seconds'.format(np.round(time() - start_t, 1)))
    print()
    return out_df


def trace_for_lineplot(organized_with_label, groupbyLabel):
    lineplot_df = pd.DataFrame(columns=['frame', 'value', 'sample_index'])
    for groupby in np.sort(organized_with_label[groupbyLabel].unique()):
        groupby_df = organized_with_label.loc[organized_with_label[groupbyLabel] == groupby, :]
        for sample in groupby_df['sample_index'].unique():
            sample_df = groupby_df.loc[groupby_df['sample_index'] == sample, :]
            trace_np = np.concatenate(sample_df['all_trace'].to_numpy()).reshape(sample_df.shape[0], -1)
            ave_trace = pd.DataFrame(np.mean(trace_np, axis=0), columns=['value'])
            ave_trace['frame'] = ave_trace.index
            ave_trace['sample_index'] = sample
            ave_trace[groupbyLabel] = groupby
            lineplot_df = lineplot_df.append(ave_trace, ignore_index=True)
    return lineplot_df


def trace_for_lineplot_stimulus(organized_with_label, groupbyLabel, scheme, output_folder, min_cell_per_sample=3):
    assert isinstance(min_cell_per_sample, int) and min_cell_per_sample > 0
    # If Scheme File is in Old Format
    if 'Frame/video' in scheme.columns:
        stimulus_ls = scheme.iloc[:, 1]
        frame_ls = scheme.iloc[:, 2].to_numpy()

        lineplot_df = pd.DataFrame(columns=['frame', 'value', 'sample_index'])
        discard_matrix = np.zeros((len(organized_with_label[groupbyLabel].unique()), len(stimulus_ls)))
        for groupby_ind, groupby in enumerate(np.sort(organized_with_label[groupbyLabel].unique())):
            groupby_df = organized_with_label.loc[organized_with_label[groupbyLabel] == groupby, :]
            for sample in organized_with_label['sample_index'].unique():
                sample_df = groupby_df.loc[groupby_df['sample_index'] == sample, :]
                for ind, stimulus in enumerate(stimulus_ls):
                    if sample_df.shape[0] >= 1:
                        trace_np = np.concatenate(sample_df['{}_trace'.format(stimulus)].to_numpy()).reshape(
                            sample_df.shape[0], -1)
                    else:
                        discard_matrix[groupby_ind, ind] += 1
                        trace_np = np.zeros((1, frame_ls[ind]))

                    ave_trace = pd.DataFrame(np.mean(trace_np, axis=0), columns=['value'])
                    ave_trace['frame'] = ave_trace.index
                    ave_trace['sample_index'] = sample
                    ave_trace['stimulus'] = stimulus
                    ave_trace['stimulus_start'] = scheme.iloc[ind, 9]
                    ave_trace['stimulus_end'] = scheme.iloc[ind, 10]
                    ave_trace['starting_frame'] = np.insert(scheme.iloc[:, 3].to_numpy(), 0, 0)[ind]
                    ave_trace[groupbyLabel] = groupby
                    lineplot_df = lineplot_df.append(ave_trace, ignore_index=True)
        pd.DataFrame(discard_matrix.astype(int)).to_csv(output_folder + "discard_mx.csv")
        return lineplot_df
    # If Scheme File is in New Format
    else:
        # Initialize Variables
        stimulus_ls = scheme['stimulation']
        frame_ls = scheme['video_end'].to_numpy() - scheme['video_start'].to_numpy() + 1
        lineplot_df = pd.DataFrame(columns=['frame', 'value', 'sample_index'])
        discard_matrix = np.zeros((len(organized_with_label[groupbyLabel].unique()), len(stimulus_ls)))

        # Loop through each groupby labels
        for groupby_ind, groupby in enumerate(np.sort(organized_with_label[groupbyLabel].unique())):
            groupby_df = organized_with_label.loc[organized_with_label[groupbyLabel] == groupby, :]
            # Loop through each sample
            for sample in organized_with_label['sample_index'].unique():
                sample_df = groupby_df.loc[groupby_df['sample_index'] == sample, :]
                # Loop through each stimulus
                for ind, stimulus in enumerate(stimulus_ls):
                    # If sample has one or more cells belonging to the group labels, average their trace
                    if sample_df.shape[0] >= min_cell_per_sample:
                        trace_np = np.concatenate(sample_df['{}_trace'.format(stimulus)].to_numpy()).reshape(
                            sample_df.shape[0], -1)
                    # Otherwise trace is a zero flat line
                    else:
                        discard_matrix[groupby_ind, ind] += 1
                        trace_np = np.zeros((1, frame_ls[ind]))

                    ave_trace = pd.DataFrame(np.nanmean(trace_np, axis=0), columns=['value'])
                    ave_trace['frame'] = ave_trace.index
                    ave_trace['sample_index'] = sample
                    ave_trace['stimulus'] = stimulus
                    ave_trace['stimulus_start'] = scheme['stimulus_start'].iloc[ind]
                    ave_trace['stimulus_end'] = scheme['stimulus_end'].iloc[ind]
                    ave_trace['starting_frame'] = scheme['video_start'].iloc[ind] - 1
                    ave_trace[groupbyLabel] = groupby
                    lineplot_df = lineplot_df.append(ave_trace, ignore_index=True)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        pd.DataFrame(discard_matrix.astype(int)).to_csv(output_folder + "discard_mx.csv")
        return lineplot_df

def trace_for_lineplot_stimulus_cell(organized_with_label, groupbyLabel, scheme, output_folder, min_cell_per_sample, min_sample_per_groupby=3):
    assert isinstance(min_cell_per_sample, int) and min_cell_per_sample > 0
    # Initialize Variables
    stimulus_ls = scheme['stimulation']
    frame_ls = scheme['video_end'].to_numpy() - scheme['video_start'].to_numpy() + 1
    lineplot_df = pd.DataFrame(columns=['frame', 'value', 'sample_index'])

    # Loop through each groupby labels
    for groupby_ind, groupby in enumerate(np.sort(organized_with_label[groupbyLabel].unique())):
        groupby_df = organized_with_label.loc[organized_with_label[groupbyLabel] == groupby, :]
        # Loop through each sample
        if len(groupby_df['sample_index'].unique()) < min_sample_per_groupby:
            for ind, stimulus in enumerate(stimulus_ls):
                trace_np = np.zeros((1, frame_ls[ind]))
                ave_trace = pd.DataFrame(trace_np[0, :], columns=['value'])
                ave_trace['frame'] = ave_trace.index
                ave_trace['sample_index'] = 'not enough sample'
                ave_trace['stimulus'] = stimulus
                ave_trace['stimulus_start'] = scheme['stimulus_start'].iloc[ind]
                ave_trace['stimulus_end'] = scheme['stimulus_end'].iloc[ind]
                ave_trace['starting_frame'] = scheme['video_start'].iloc[ind] - 1
                ave_trace[groupbyLabel] = groupby
                lineplot_df = lineplot_df.append(ave_trace, ignore_index=True)
        else:
            valid_sample_count = 0
            temp_lineplot_df = pd.DataFrame(columns=['frame', 'value', 'sample_index'])
            for sample in groupby_df['sample_index'].unique():
                sample_df = groupby_df.loc[groupby_df['sample_index'] == sample, :]
                if sample_df.shape[0] < min_cell_per_sample:
                    continue
                # Loop through each stimulus
                for ind, stimulus in enumerate(stimulus_ls):
                    # If sample has one or more cells belonging to the group labels, average their trace
                    trace_np = np.concatenate(sample_df['{}_trace'.format(stimulus)].to_numpy()).reshape(
                        sample_df.shape[0], -1)
                    for cell_index in range(trace_np.shape[0]):
                        ave_trace = pd.DataFrame(trace_np[cell_index, :], columns=['value'])
                        ave_trace['frame'] = ave_trace.index
                        ave_trace['sample_index'] = sample
                        ave_trace['stimulus'] = stimulus
                        ave_trace['stimulus_start'] = scheme['stimulus_start'].iloc[ind]
                        ave_trace['stimulus_end'] = scheme['stimulus_end'].iloc[ind]
                        ave_trace['starting_frame'] = scheme['video_start'].iloc[ind]-1
                        ave_trace[groupbyLabel] = groupby
                        temp_lineplot_df = temp_lineplot_df.append(ave_trace, ignore_index=True)
                valid_sample_count += 1
            if valid_sample_count >= min_sample_per_groupby:
                lineplot_df = lineplot_df.append(temp_lineplot_df, ignore_index=True)
            else:
                for ind, stimulus in enumerate(stimulus_ls):
                    trace_np = np.zeros((1, frame_ls[ind]))
                    ave_trace = pd.DataFrame(trace_np[0, :], columns=['value'])
                    ave_trace['frame'] = ave_trace.index
                    ave_trace['sample_index'] = 'not enough sample'
                    ave_trace['stimulus'] = stimulus
                    ave_trace['stimulus_start'] = scheme['stimulus_start'].iloc[ind]
                    ave_trace['stimulus_end'] = scheme['stimulus_end'].iloc[ind]
                    ave_trace['starting_frame'] = scheme['video_start'].iloc[ind] - 1
                    ave_trace[groupbyLabel] = groupby
                    lineplot_df = lineplot_df.append(ave_trace, ignore_index=True)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return lineplot_df


def data_plus_label(organized, label):
    return pd.merge(organized, label, on='ROI_index')


def parameters_mean_sem(organized, save_dir, roi_names_file, scheme, cluster=0):
    mean_data = organized.groupby('sample_index').mean().transpose()
    response_columns = [column for column in organized.columns if 'response' in column]
    sum_data = organized.groupby('sample_index').sum()[response_columns]
    sum_data.columns = [column + '_sum' for column in sum_data.columns]
    data = mean_data.append(sum_data.transpose())
    summary_data = data.copy()
    summary_data['average'] = data.copy().mean(numeric_only=True, axis=1)
    summary_data['sem'] = data.copy().sem(numeric_only=True, axis=1)

    original_index = list(summary_data.index)
    temp_index = []
    stimulus_suffix = []
    for index in original_index:
        if any(stimulus in index for stimulus in scheme['stimulation']):
            stimulus_suffix.append('_'.join(index.split('_')[1:]))
        else:
            temp_index.append(index)
    stimulus_suffix = np.unique(np.array(stimulus_suffix)).tolist()
    for suffix in stimulus_suffix:
        for stimulus in scheme['stimulation']:
            temp_index.append('{}_{}'.format(stimulus, suffix))
    target_order = temp_index

    summary_data = summary_data.reindex(target_order)
    if not cluster:
        summary_data.to_csv(save_dir + roi_names_file.split('.')[0] + '_para_mean_sem.txt', sep='\t')
    return summary_data


def parameters_mean_sem_cluster(organized, save_dir, roi_names_file, cluster_label_file, scheme):
    results_dict = {}
    cluster_label_df = pd.read_csv(save_dir + 'SelectedCluster\\' + cluster_label_file, sep='\t')
    organized = pd.merge(organized, cluster_label_df[['ROI_index', 'label']], on='ROI_index')
    for cluster in np.unique(organized['label'].astype(int)):
        organized_cluster = organized[organized['label'] == cluster]
        results_dict.update(
            {cluster: parameters_mean_sem(organized_cluster, save_dir, roi_names_file, cluster=1, scheme=scheme)})

    with pd.ExcelWriter(save_dir + roi_names_file.split('.')[0] + '_cluster_para_mean_sem.xlsx') as writer:
        for cluster in np.unique(organized['label'].astype(int)):
            results_dict[cluster].iloc[:, :-2].to_excel(writer, sheet_name=str(cluster))

    ave_sem_df = pd.DataFrame()
    for cluster in np.unique(organized['label'].astype(int)):
        temp_ave_sem = results_dict[cluster].iloc[:, -2:]
        temp_ave_sem.columns = ['{}_ave'.format(cluster), '{}_sem'.format(cluster)]
        ave_sem_df['{}_ave'.format(cluster)] = temp_ave_sem['{}_ave'.format(cluster)]
        ave_sem_df['{}_sem'.format(cluster)] = temp_ave_sem['{}_sem'.format(cluster)]

    original_index = ave_sem_df.columns
    temp_index = []
    stimulus_suffix = []
    for index in original_index:
        if any(stimulus in index for stimulus in scheme['stimulation']):
            stimulus_suffix.append('_'.join(index.split('_')[1:]))
        else:
            temp_index.append(index)
    stimulus_suffix = np.unique(np.array(stimulus_suffix)).tolist()
    for suffix in stimulus_suffix:
        for stimulus in scheme['stimulation']:
            temp_index.append('{}_{}'.format(stimulus, suffix))
    target_order = temp_index
    ave_sem_df = ave_sem_df[target_order]

    ave_sem_df.to_csv(save_dir + roi_names_file.split('.')[0] + '_cluster_para_mean_sem.csv')


def organized_gen(sample_ls, scheme, input_folder, raw=0, svm_model_name='', filter_ls=[], custom_peak=''):
    def load_background(sample_index):
        def f0_calculation(row, f0_start, f0_end):
            f0 = np.mean(row[f0_start:f0_end])
            f0_array = (row - f0) / f0
            return f0_array

        def segment_trace(trace, start, end):
            return trace.values[start:end]

        temp_background_df = pd.read_csv(
            'D:\\Gut Imaging\\Videos\\background_files\\background_{}.txt'.format(sample_index), sep='\t',
            header=None)
        temp_background_np = np.empty((temp_background_df.shape[0], 0))
        for stimulus_ind in range(scheme.shape[0]):
            stimulus_np = np.concatenate(temp_background_df.apply(segment_trace, axis=1, args=(
                scheme['video_start'].iloc[stimulus_ind] - 1,
                scheme['video_end'].iloc[stimulus_ind])).to_numpy()).reshape(
                -1, scheme['video_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[stimulus_ind] + 1)
            f0_np = np.apply_along_axis(f0_calculation, 1, stimulus_np,
                                        scheme['F0_start'].iloc[stimulus_ind] - scheme['video_start'].iloc[
                                            stimulus_ind],
                                        scheme['F0_end'].iloc[stimulus_ind] - scheme['video_start'].iloc[
                                            stimulus_ind] + 1)
            temp_background_np = np.append(temp_background_np, f0_np, axis=1)
        temp_background_df = pd.DataFrame(temp_background_np)
        temp_background_index = ['{}_b{}'.format(sample_index, bg_ind) for bg_ind in
                                 range(temp_background_df.shape[0])]
        temp_background_df.index = temp_background_index
        temp_background_df.columns = np.array([int(column) for column in temp_background_df.columns]) + 1
        temp_background_df['ROI_index'] = temp_background_index

        return temp_background_df

    def region_data_gen(dff_ls):
        def AP_norm(cum_AP, boundary):
            if boundaries[-1] < np.max(cum_AP):
                boundaries[-1] = np.max(cum_AP)
            boundary = [0] + boundary
            out = np.empty((0,))
            for b_ind in range(len(boundary)-1):
                cum = cum_AP[(cum_AP > boundary[b_ind]) & (cum_AP <= boundary[b_ind+1])]
                out = np.append(out, 100*b_ind + (cum-boundary[b_ind]) * (100/(boundary[b_ind+1] - boundary[b_ind])))
            return np.round(out,2)

        region_data = pd.DataFrame(columns=['ROI_ID', 'cum_AP', 'NormAP', 'Region'])
        for dff in dff_ls:
            sample_ind = dff.split('_')[1]
            temp_seg_df = pd.read_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\manual_segment\\{}_manual_segment.csv'.format(sample_ind))[['ROI_ID', 'cum_AP', 'region']].sort_values('cum_AP').reset_index(drop=True)
            boundaries = (pd.read_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\region_boundary\\{}_region_boundary.csv'.format(sample_ind), index_col=0).values.flatten()*100).tolist()
            temp_region_data = pd.DataFrame()
            temp_region_data['ROI_ID'] = ['X{}_{}'.format(sample_ind, roi) for roi in temp_seg_df['ROI_ID']]
            temp_region_data['cum_AP'] = temp_seg_df['cum_AP'].values * 100
            temp_region_data['NormAP'] = AP_norm(temp_region_data['cum_AP'].values, boundaries)
            temp_region_data['Region'] = temp_seg_df['region']
            region_data = region_data.append(temp_region_data, ignore_index=True)
        region_data = region_data.sort_values(['cum_AP']).reset_index(drop=True)[['ROI_ID', 'cum_AP', 'NormAP', 'Region']]
        return region_data

    raw_ls = []
    dff_ls = []

    raw_directory_str = 'D:\\Gut Imaging\\Videos\\CommonFiles\\f_file\\'
    raw_directory = os.fsencode(raw_directory_str)

    for file in os.listdir(raw_directory):
        filename = os.fsdecode(file)
        if os.path.isdir(raw_directory_str + filename):
            continue
        elif filename.startswith("f_") and any(sample in filename for sample in sample_ls):
            raw_ls.append(filename)

    directory_str = 'D:\\Gut Imaging\\Videos\\CommonFiles\\dff_file\\'
    directory = os.fsencode(directory_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if os.path.isdir(input_folder + filename):
            continue
        elif filename.endswith("Ave.txt") and any(sample in filename for sample in sample_ls):
            dff_ls.append(filename)

    region_data = region_data_gen(dff_ls)
    # region_data
    # exit()
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     if 'Peak' in filename and not filename.startswith('~'):
    #         region_data = pd.read_csv(input_folder + filename, delimiter='\t')

    if not raw:
        file_ls = dff_ls
    else:
        file_ls = raw_ls

    all_data = pd.DataFrame()

    for file in file_ls:
        if not raw:
            temp_df = pd.read_csv(directory_str + file, delimiter='\t').iloc[:, 2:]
            temp_columns = temp_df.columns
            temp_df = temp_df.transpose()
            temp_df['ROI_index'] = temp_columns
            temp_df = temp_df.iloc[:, 1:]
            sample_index = file.split('_')[1]
        else:
            temp_df = pd.read_csv(raw_directory_str + file, delimiter='\t').iloc[:, 1:]
            sample_index = file.split('.')[0].split('_')[1]
            temp_columns = [sample_index + '_' + roi_name for roi_name in temp_df.columns]
            temp_df.columns = temp_columns
            temp_df = temp_df.transpose()
            temp_df.columns = [int(column) + 1 for column in temp_df.columns]
            temp_df['ROI_index'] = temp_columns

        all_data = all_data.append(temp_df, ignore_index=True)

    if svm_model_name != '':
        raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')
        sti_ls = [sti for sti in scheme['stimulation'] if
                  'KCl' not in sti and 'AHL' not in sti and not np.any(sti in filter_ls)]
        sti_df = pd.DataFrame()
        for sti in sti_ls:
            for sample in raw_organized['sample_index'].unique():
                raw_copy = raw_organized.copy()
                raw_copy = raw_copy[raw_copy['sample_index'] == sample]

                temp_sti_df = pd.DataFrame()
                temp_sti_df['ROI_index'] = raw_copy['ROI_index']
                temp_sti_df['basal'] = raw_copy['{}_f0_mean'.format(sti)]
                temp_sti_df['std_basal'] = raw_copy['{}_f0_std'.format(sti)]
                temp_sti_df['cv'] = temp_sti_df['std_basal'] / temp_sti_df['basal']
                temp_sti_df['average_basal'] = temp_sti_df['basal'].mean()
                temp_sti_df['median_basal'] = temp_sti_df['basal'].median()
                temp_sti_df['std_basal'] = temp_sti_df['basal'].std()
                temp_sti_df['n_mean_basal'] = temp_sti_df['basal'] / temp_sti_df['basal'].mean()
                temp_sti_df['n_median_basal'] = temp_sti_df['basal'] / temp_sti_df['basal'].median()
                temp_sti_df['std_n_mean_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].mean()) / temp_sti_df[
                    'basal'].std()
                temp_sti_df['std_n_median_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].median()) / \
                                                    temp_sti_df['basal'].std()

                sti_df = sti_df.append(temp_sti_df, ignore_index=True)

        X = np.hstack((sti_df['basal'].to_numpy().reshape((-1, 1)), sti_df['std_basal'].to_numpy().reshape((-1, 1)),
                       sti_df['cv'].to_numpy().reshape((-1, 1)), sti_df['average_basal'].to_numpy().reshape((-1, 1)),
                       sti_df['median_basal'].to_numpy().reshape((-1, 1)),
                       sti_df['std_basal'].to_numpy().reshape((-1, 1)),
                       sti_df['n_mean_basal'].to_numpy().reshape((-1, 1)),
                       sti_df['n_median_basal'].to_numpy().reshape((-1, 1)),
                       sti_df['std_n_mean_basal'].to_numpy().reshape((-1, 1)),
                       sti_df['std_n_median_basal'].to_numpy().reshape((-1, 1))))

        dict_name = 'D:\\Gut Imaging\\Videos\\CommonFiles\\svm_models.pkl'
        assert os.path.exists(dict_name)

        with open(dict_name, 'rb') as handle:
            model_dict = pickle.load(handle)

        coef = model_dict['{}_coef'.format(svm_model_name)].reshape((-1,1))
        intercept = model_dict['{}_intercept'.format(svm_model_name)]

        y_pred = np.array(np.matmul(X, coef) + intercept > 0).astype(int).flatten()

        df = pd.DataFrame(sti_df['ROI_index'])
        df['pred'] = y_pred
        throw_ls = df[df['pred'] == 0]['ROI_index'].unique()
        keep_ls = df[~df['ROI_index'].isin(throw_ls)]['ROI_index'].unique()
        if not raw:
            throw_df = all_data[all_data['ROI_index'].isin(throw_ls)]['ROI_index'].reset_index(drop=True)
            temp_raw_org = raw_organized.copy()[['sample_index', 'ROI_index', 'region', 'fine_region']]
            throw_df = temp_raw_org[temp_raw_org['ROI_index'].isin(throw_df)]
            if not os.path.exists(input_folder + 'region_count\\'):
                os.makedirs(input_folder + 'region_count\\')
            for sample in throw_df['sample_index'].unique():
                temp_sample_df = throw_df.copy()
                temp_sample_df = temp_sample_df[temp_sample_df['sample_index'] == sample]
                throw_df_count = temp_sample_df.groupby('fine_region').count().iloc[:,0]
                throw_df_count.name = 'SVM_throw'
                count_df = pd.read_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\region_count\\{}_region_count.csv'.format(sample), index_col=0)
                count_df = pd.concat((count_df, throw_df_count),axis=1)
                count_df = count_df.fillna(0)
                count_df['SVM_left'] = count_df['kept'] - count_df['SVM_throw']
                count_df.to_csv(input_folder + 'region_count\\{}_region_count.csv'.format(sample))

        all_data = all_data[~all_data['ROI_index'].isin(throw_ls)].reset_index(drop=True)

        if not os.path.exists(input_folder + 'svm_filtered\\'):
            os.makedirs(input_folder + 'svm_filtered\\')
        keep_df = pd.DataFrame(keep_ls, columns=['ROI_index'])
        keep_df.to_pickle(input_folder + 'svm_filtered\\{}_keep.pkl'.format(svm_model_name))

        for file in dff_ls:
            temp_df = pd.read_csv(directory_str + file, delimiter='\t', index_col=0)
            temp_df = temp_df[['Ave'] + [column for column in temp_df.columns if column in keep_ls]]
            temp_df['Ave'] = temp_df[[column for column in temp_df.columns if column in keep_ls]].mean(axis=1)
            temp_df.at['AP', 'Ave'] = 0
            temp_df.to_csv(input_folder + 'svm_filtered\\' + file.split('.')[0] + '_svm_filtered.txt', sep='\t')

        for file in raw_ls:
            temp_df = pd.read_csv(raw_directory_str + file, delimiter='\t', index_col=0)
            sample_index = file.split('.')[0].split('_')[1]
            temp_df.columns = ['{}_{}'.format(sample_index, roi_index) for roi_index in temp_df.columns]
            temp_df = temp_df[[column for column in temp_df.columns if column in keep_ls]]
            temp_df.columns = [column.split('_')[1] for column in temp_df.columns]
            temp_df.to_csv(input_folder + 'svm_filtered\\' + file.split('.')[0] + '_svm_filtered.txt', sep='\t')

    for file in file_ls:
        if not raw:
            sample_index = file.split('_')[1]
        else:
            sample_index = file.split('.')[0].split('_')[1]

        temp_background_df = load_background(sample_index)
        all_data = all_data.append(temp_background_df, ignore_index=True)

    organized_name = input_folder.split('\\')[-2]
    if raw:
        organized_name = 'Raw_' + organized_name

    if len(custom_peak) != 0:
        custom_peak = pd.read_csv(input_folder + '\\{}'.format(custom_peak), sep='\t')
    else:
        custom_peak = None

    stimuli_ls = scheme.iloc[:, 1].tolist()
    assert len(stimuli_ls) == len(np.unique(stimuli_ls)), "Repetition of Stimuli name in Scheme file"
    organized = DataOrganizer_updated_scheme(name=organized_name, scheme=scheme, save_dir=input_folder,
                                             all_data=all_data, region_data=region_data, custom_peak=custom_peak)
    if svm_model_name != '':
        print('{} % kept'.format(np.round(organized[~organized["ROI_index"].str.contains('_b')].shape[0] /
                                          raw_organized[~raw_organized["ROI_index"].str.contains('_b')].shape[0] * 100,
                                          1)))

    return organized

def organized2excel(organized, input_folder, stimulus_ls, all_trace=0):
    if all_trace:
        organized = organized.loc[organized[['{}_individual_response'.format(stimulus) for stimulus in
                                                   stimulus_ls if 'KCl' not in stimulus]].any(axis=1), :].reset_index(drop=True)
        trace_np = np.concatenate(organized['all_trace'].to_numpy()).reshape(organized.shape[0], -1)
        trace_np_df = pd.DataFrame(trace_np)
        trace_np_df.insert(loc=0, column='ROI_index', value=organized['ROI_index'])
        trace_np_df.to_csv(input_folder + 'all_trace.txt', sep='\t', index=False)
        # np.savetxt(input_folder + 'all_trace.txt', trace_np, delimiter='\t')
        reduced_trace_np = np.concatenate(organized['reduced_trace'].to_numpy()).reshape(organized.shape[0], -1)
        reduced_trace_df = pd.DataFrame(reduced_trace_np)
        reduced_trace_df.insert(loc=0, column='ROI_index', value=organized['ROI_index'])
        reduced_trace_df.to_csv(input_folder + 'reduced_trace.txt', sep='\t', index=False)
        # np.savetxt(input_folder + 'reduced_trace.txt', reduced_trace_np, delimiter='\t')
    else:
        for stimulus in stimulus_ls:
            sti_organized = organized[organized['{}_individual_response'.format(stimulus)] == 1]
            stimulus_np = np.concatenate(sti_organized['{}_trace'.format(stimulus)].to_numpy()).reshape(sti_organized.shape[0],-1)
            np.savetxt(input_folder+'{}_trace.txt'.format(stimulus),stimulus_np, delimiter='\t')