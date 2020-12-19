from sys import exit
import numpy as np
import pandas as pd
from video_util import read_tif
from Igor_related_util import read_igor_roi_matrix, read_igor_f_matrix
from itertools import product
from os import listdir, makedirs
from os.path import isfile, join, exists
import cv2
from imageio import imwrite
from util import roi_plot
from scipy.signal import correlate2d
from os import path
from scipy.ndimage.morphology import binary_dilation
import csv


def f_cleanup(root_folder, folder, f_from_roi_flag = 0, interval=0.5, line=1, label_font=2, suggest_mc=0):
    def on_the_edge(ROI_matrix, num_horizontal_cut, num_vertical_cut):
        # Returns the rois that touch the cropping cut line and FOV edges.
        #
        # Input:    ROI_matrix: 2d np matrix, background as 0, roi masks with index number, roi index starts from 1;
        #                       roi matrix containing all roi information from igor output.
        #
        #           num_horizontal_cut: integer; number of horizontal lines drawn to cut the crops,
        #                               e.g. 2 lines crop image to 3 segments
        #
        #           num_vertical_cut: integer; number of vertical lines drawn to cut the crops,
        #                             e.g. 2 lines crop image to 3 segments
        #
        # Output:   discard_ls: list; roi indices that fails the examination, roi index starts from 0.

        FOV_row = ROI_matrix.shape[0]
        FOV_col = ROI_matrix.shape[1]
        coordinate = np.asarray(list(product(range(FOV_row), range(FOV_col))))
        horizontal_line_coordinate = np.array([])
        vertical_line_coordinate = np.array([])

        for horizontal_cut in [0] + (np.asarray(list(range(num_horizontal_cut))) + 1).tolist():
            ratio = horizontal_cut / num_horizontal_cut
            if ratio == 1:
                fixed = np.floor(ratio * FOV_row - 1)
            else:
                fixed = np.floor(ratio * FOV_row)
            fixed_coordinate = coordinate[np.where(coordinate[:, 0] == fixed)[0], :]
            horizontal_line_coordinate = np.vstack(
                [horizontal_line_coordinate, fixed_coordinate]) if horizontal_line_coordinate.size else fixed_coordinate

        for vertical_cut in [0] + (np.asarray(list(range(num_vertical_cut))) + 1).tolist():
            ratio = vertical_cut / num_vertical_cut
            if ratio == 1:
                fixed = np.floor(ratio * FOV_col - 1)
            else:
                fixed = np.floor(ratio * FOV_col)
            fixed_coordinate = coordinate[np.where(coordinate[:, 1] == fixed)[0], :]
            vertical_line_coordinate = np.vstack(
                [vertical_line_coordinate, fixed_coordinate]) if vertical_line_coordinate.size else fixed_coordinate

        line_edge = np.unique(np.concatenate((horizontal_line_coordinate, vertical_line_coordinate), axis=0), axis=0)
        discard_ls = []
        for column in np.unique(ROI_matrix)[0:-1]:
            ROI_coordinate = np.vstack(
                (np.where(ROI_matrix == int(column) + 1)[0], np.where(ROI_matrix == int(column) + 1)[1])).transpose()
            if np.vstack((line_edge, ROI_coordinate)).shape[0] != \
                    np.unique(np.vstack((line_edge, ROI_coordinate)), axis=0).shape[0]:
                discard_ls.append(column)

        return discard_ls

    def immuno_roi_id(ROI_matrix, updated_f):
        files = [f for f in listdir(immuno_folder) if f.split('.')[-1] == 'csv' and 'Immuno' in f]
        if len(files) <= 0:
            return
        if path.exists(immuno_folder + "immuno_targets.txt"):
            immuno_target_df = pd.read_csv(immuno_folder + "immuno_targets.txt", sep='\t').transpose()
        else:
            immuno_target_df = pd.DataFrame(columns=updated_f.columns)
        for file in files:
            immuno_name = file.split('.')[0].split('_')[-1]
            print(immuno_target_df.index.tolist())
            if path.exists(immuno_folder + "immuno_targets.txt"):
                if immuno_name in immuno_target_df.index.tolist():
                    continue

            target_roi = []
            manual_bad_roi = pd.read_csv(immuno_folder + file)[['Type', 'X', 'Y', 'Width', 'Height']]
            manual_bad_roi = manual_bad_roi[manual_bad_roi['Type'] == 'Rectangle']
            manual_mask = np.zeros(ROI_matrix.shape)
            for row_ind, row in manual_bad_roi.iterrows():
                manual_mask[row['Y'] - 1:row['Y'] + row['Height'], row['X'] - 1:row['X'] + row['Width']] = 1
            filtered_mask = manual_mask * ROI_matrix
            filtered_mask = filtered_mask[filtered_mask != 0] - 1
            unique_roi, bad_roi_pixel_counts = np.unique(filtered_mask, return_counts=True)
            for target_roi_index, target_manual_mc_roi in enumerate(unique_roi):
                if bad_roi_pixel_counts[target_roi_index] / np.sum(ROI_matrix == target_manual_mc_roi + 1) > 0.9:
                    target_roi.append(int(target_manual_mc_roi))
            immuno_binary = np.array([int(roi in target_roi) for roi in [int(i) for i in immuno_target_df.columns]])
            immuno_binary = pd.Series(immuno_binary, index=immuno_target_df.columns)
            immuno_binary.name = immuno_name
            immuno_target_df = immuno_target_df.append(immuno_binary)
        immuno_target_df.transpose().to_csv(immuno_folder + 'immuno_targets.txt', sep='\t')

    def manual_failed_mc(ROI_matrix):
        bad_mc = []

        manual_bad_roi = pd.read_csv(folder+'Overlay Elements.csv')[['Type','X', 'Y', 'Width', 'Height']]
        manual_bad_roi = manual_bad_roi[manual_bad_roi['Type'] == 'Rectangle']
        manual_mask = np.zeros(ROI_matrix.shape)
        for row_ind, row in manual_bad_roi.iterrows():
            manual_mask[row['Y']-1:row['Y'] + row['Height'], row['X']-1:row['X'] + row['Width']] = 1
        filtered_mask = manual_mask * ROI_matrix
        filtered_mask = filtered_mask[filtered_mask!=0] - 1
        unique_roi, bad_roi_pixel_counts = np.unique(filtered_mask, return_counts=True)
        for bad_roi_index, bad_manual_mc_roi in enumerate(unique_roi):
            if bad_roi_pixel_counts[bad_roi_index]/np.sum(ROI_matrix == bad_manual_mc_roi+1) > 0.9:
                bad_mc.append(int(bad_manual_mc_roi))
        print("{} ROIs manually discarded".format(len(bad_mc)))
        return bad_mc

    def suggest_failed_mc(ROI_matrix):
        # Returns roi indices that has a bad motion correction results based on 2d cross-correlation inside a minimum tectangle
        # that encloses the roi in each frame against its next frame. If the cumulative offset calculated from the correlation
        # across time is too high, discard roi.
        #
        # Input: ROI_matrix: 2d np matrix, background as 0, roi masks with index number, roi index starts from 1;
        #                    roi matrix containing all roi information from igor output.
        #
        # Output: high_cv:  list; roi indices that fails the examination, roi index starts from 0.

        def offset_calc(corr_result, shape):
            if shape % 2 == 1:
                offset = (shape/2 - 0.5) - corr_result
            if shape % 2 == 0:
                offset = (shape / 2 - 1) - corr_result
            return np.abs(offset)
        offset_sum_ls = []
        # column named after index start from 0
        for column in np.unique(ROI_matrix)[1:]:
            print('suggest_failed_mc: {}'.format(str(int(column)) + '/' + str(len(np.unique(ROI_matrix)[1:]))))
            roi_coord = np.where(ROI_matrix == int(column))
            y_max = np.max(roi_coord[0])
            y_min = np.min(roi_coord[0])
            x_max = np.max(roi_coord[1])
            x_min = np.min(roi_coord[1])
            offset_ls = []
            if (y_max - y_min) * (x_max - x_min) > 1000:
                offset_sum_ls.append(np.inf)
                print('ROI too big, discard')
            else:
                for frame in range(raw_stack.shape[0])[:-1]:
                    curr_frame = raw_stack[frame, :, :]
                    next_frame = raw_stack[frame + 1, :, :]
                    image_2d_curr = curr_frame[y_min:y_max + 1, x_min:x_max + 1]
                    image_2d_next = next_frame[y_min:y_max + 1, x_min:x_max + 1]
                    corr_2d_result = correlate2d(image_2d_curr, image_2d_next, mode='same')
                    max_ind = np.unravel_index(corr_2d_result.argmax(), corr_2d_result.shape)
                    y_offset = offset_calc(max_ind[0], corr_2d_result.shape[0])
                    x_offset = offset_calc(max_ind[1], corr_2d_result.shape[1])
                    total_offset = (y_offset + x_offset)
                    offset_ls.append(total_offset)

                offset_sum_ls.append(np.sum(offset_ls))

        offset_sum_ls = np.asarray(offset_sum_ls)
        threshold = 40
        suggest = np.where(offset_sum_ls >= threshold)[0]
        all_roi = np.unique(igor_roi)[1:] - 1
        not_suggest = all_roi[np.invert(np.isin(all_roi, suggest))].astype(int)
        circle_layer(igor_roi, suggest, "suggest")
        circle_layer(igor_roi, not_suggest, "not_suggest")

    def find_background(ROI_matrix, background_num, background_size):
        roi = ROI_matrix.copy()
        roi[roi != 0] = 1
        roi = binary_dilation(roi, iterations=5)
        seed_init = int(f_matrix.iloc[:,0].mean()*100)
        np.random.seed(seed_init)
        xrand = np.random.randint(low=0, high=ROI_matrix.shape[1] - background_size,  size=10000)
        np.random.seed(seed_init*2)
        yrand = np.random.randint(low=0, high=ROI_matrix.shape[0] - background_size, size=10000)
        coord = list(zip(yrand, xrand))
        background_trace = np.empty((0, f_matrix.shape[0]))
        candidate_count = 0
        while background_trace.shape[0] < background_num:
            temp_y = coord[candidate_count][0]
            temp_x = coord[candidate_count][1]
            temp_background = roi[temp_y:temp_y + background_size, temp_x:temp_x + background_size]
            if 1 not in temp_background.astype(int):
                temp_background = raw_stack[:, temp_y:temp_y + background_size, temp_x:temp_x + background_size].copy()
                std_check = np.std(np.mean(temp_background, axis=(1, 2)).reshape((1,-1)))
                if std_check > 0.7 and std_check < 1.3:
                    background_trace = np.append(background_trace, np.mean(temp_background, axis=(1, 2)).reshape((1,-1)), axis=0)
                    print(std_check)
            candidate_count += 1
        np.savetxt('D:\\Gut Imaging\\Videos\\background_files\\background_{}.txt'.format(folder.split('\\')[-2][1:]), background_trace, delimiter='\t')

    def circle_layer(ROI_matrix, ROI_ls, file_name):
        # Generate a gray scale image depicting edges of rois as circles. Background has values of 0,
        # circles have values of 255. Rois to be drawn are based on input dataframe. The image is saved under the
        # current video folder.
        #
        # Input: ROI_matrix: 2d np matrix, background as 0, roi masks with index number, roi index starts from 1;
        #                    roi matrix containing all roi information from igor output.
        #
        #        f_df: dataframe, index column is time in seconds, columns are roi indices, roi index starts from 0;
        #              circles to be drawn are based on rois shown in this dataframe

        mask = np.copy(ROI_matrix)
        mask[:, :] = 0
        for column in ROI_ls:
            column = int(float(column))

            mask[np.where(ROI_matrix == int(column) + 1)] = 255
        test = mask.copy().astype('uint8')

        # im3 = cv2.findContours(test, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        (contours, _) = cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(test.shape[:2], dtype="uint8") * 255

        cv2.drawContours(mask, contours, -1, 255, 1)

        # mask[:, :] = 0
        # for i in range(len(im3[1])):
        #     for j in range(len(im3[1][i])):
        #         mask[im3[1][i][j][0][1], im3[1][i][j][0][0]] = 255
        # mask = mask.astype(np.uint8)
        cv2.imwrite(folder + 'overlay_{}'.format(file_name) + '.png', mask)
        # imwrite(folder + 'overlay_{}'.format(file_name) + '.png', mask)
        return

    def f_from_roi(ROI_matrix, raw_stack):
        ROI_num = len(np.unique(ROI_matrix)) - 1
        f_np = np.empty((raw_stack.shape[0], ROI_num))

        for ROI_index in range(ROI_num):

            for frame_idx in range(raw_stack.shape[0]):
                f_np[frame_idx,ROI_index] = np.sum(raw_stack[frame_idx, :, :] * (ROI_matrix == ROI_index + 1))/np.sum(ROI_matrix == ROI_index + 1)
        f_df = pd.DataFrame(f_np)
        return f_df

    def delta_f_f_calc(updated_f_matrix):
        def calculate_delta_f_over_f(trace, f_start_frame, f_end_frame):
            f0 = np.mean(trace[f_start_frame:f_end_frame])
            delta_f_f = (trace - f0) / f0
            return delta_f_f

        scheme = pd.read_excel(folder + '\\DeliverySchemes.xlsx')
        temp_dff_df_ls = []
        for sti_index in range(scheme.shape[0]):
            temp_f_df = updated_f_matrix.iloc[
                        scheme.iloc[sti_index]['video_start'] - 1:scheme.iloc[sti_index]['video_end'], :].values
            temp_dff_df_ls.append(pd.DataFrame(
                np.apply_along_axis(calculate_delta_f_over_f, 0, temp_f_df, scheme['F0_start'].iloc[sti_index] -
                                      scheme['video_start'].iloc[sti_index],
                                      scheme['F0_end'].iloc[sti_index] - scheme['video_start'].iloc[
                                          sti_index] + 1)))
        temp_dff_df_ls = pd.concat(temp_dff_df_ls)
        temp_dff_df_ls.columns = updated_f_matrix.columns
        temp_dff_df_ls.index = updated_f_matrix.index
        out_columns = updated_f_matrix.columns
        segment_df = pd.read_csv(folder + 'segment.csv')
        segment_df = segment_df[segment_df['ROI_ID'].isin(updated_f_matrix.columns)]
        out_df = pd.DataFrame(columns=out_columns)
        out_df.loc[0] = segment_df['distance_to_origin'].values
        out_df.index = ['AP']
        out_df = pd.concat((out_df, temp_dff_df_ls))
        out_df = out_df.dropna(axis=1)
        out_df['Ave'] = out_df.mean(axis=1)
        out_df.loc['AP']['Ave'] = 0
        out_df = out_df.sort_values(by='AP', axis=1)
        out_df.columns = ['Ave']+['{}_{}'.format(folder.split('\\')[-2][1:], roi_ind) for roi_ind in out_df.columns[1:]]

        return out_df

    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    video_name = [f for f in onlyfiles if f.split('.')[-1] == 'tif' and f.split('_')[0].lower() != 'MAX'.lower() and 'immuno' not in f]
    assert len(video_name) == 1, "More than one tif file found in" + folder
    video_name = video_name[0]
    raw_stack, video_index = read_tif(folder, video_name)
    video_index = video_index.split('_')[-1]

    igor_roi = read_igor_roi_matrix(folder + 'roi' + '.csv')
    if f_from_roi_flag:
        if igor_roi.shape != raw_stack.shape[1:]:
            return
        f_matrix = f_from_roi(igor_roi, raw_stack)
        f_np = f_matrix.to_numpy()
        f_np = np.append(np.arange(f_np.shape[1]).reshape(1,-1).astype(int), f_np, axis=0)
        f_np = np.append(np.zeros((1, f_np.shape[1])).reshape(1,-1), f_np, axis=0)
        np.savetxt(folder + 'f.csv', f_np, delimiter=',')
    else:
        f_matrix = read_igor_f_matrix(folder + 'f' + '.csv')
    sampling_rate = 1 / interval

    frame_to_time = {}
    for frame_idx in range(f_matrix.shape[0]):
        frame_to_time[frame_idx] = frame_idx / sampling_rate
    f_matrix = f_matrix.rename(frame_to_time, axis='index')

    # if len(raw_stack.shape) > 2:
    #     find_background(igor_roi, 10, 5)

    mc = path.exists(folder + "Overlay Elements.csv")

    if len(raw_stack.shape) == 3:
        if not path.exists(folder + "overlay_suggest.png"):
            if suggest_mc:
                suggest_failed_mc(igor_roi)
        bad_column = []
        bad_edge = []

        if mc:
            bad_column = manual_failed_mc(igor_roi)
            roi_plot(igor_roi, folder, label_font, throw_ls=bad_column)
        else:
            if line:
                bad_edge = on_the_edge(igor_roi, 3, 3)

        if line or mc:
            drop_ls = np.unique(bad_column + bad_edge).astype(int)
            updated_f_matrix = f_matrix.drop(f_matrix.columns[drop_ls], axis=1)
        else:
            updated_f_matrix = f_matrix
    else:
        updated_f_matrix = f_matrix



    if not mc:
        circle_layer(igor_roi, updated_f_matrix, "")
        roi_plot(igor_roi,folder,label_font)

    f_folder = root_folder + root_folder.split('\\')[-2] + '_txt\\f'
    if not exists(f_folder):
        makedirs(f_folder)

    dff_folder = root_folder + root_folder.split('\\')[-2] + '_txt\\dff'
    if not exists(dff_folder):
        makedirs(dff_folder)

    immuno_folder = folder+'Immuno\\'
    if not exists(immuno_folder):
        makedirs(immuno_folder)
    immuno_roi_id(igor_roi, updated_f_matrix)

    if exists(folder + 'segment.csv'):
        dff_df = delta_f_f_calc(updated_f_matrix)
        dff_df.to_csv(dff_folder + '\\DFF_{}_Ave.txt'.format(str(video_index.split('d')[-1])),
                                sep='\t')
        dff_df.to_csv(
            'D:\\Gut Imaging\\Videos\\CommonFiles\\dff_file' + '\\DFF_{}_Ave.txt'.format(str(video_index.split('d')[-1])),
            sep='\t')

    updated_f_matrix.to_csv(f_folder + '\\f_' + str(video_index.split('d')[-1]) + '.txt',
                            sep='\t')

    updated_f_matrix.to_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\f_file' + '\\f_' + str(video_index.split('d')[-1]) + '.txt',
                            sep='\t')

