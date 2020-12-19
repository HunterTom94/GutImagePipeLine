import cv2
import numpy as np
import pandas as pd
import math
import os
from geometry import scaled_line, in_area
from opencv_drawing import LinkPoints, drawSegmentPoints
from Igor_related_util import read_igor_roi_matrix
from ROI_identification import neuron_label_matrix2dataframe
from win32api import GetSystemMetrics
from distutils.dir_util import copy_tree
from video_util import read_tif
import pickle

#                               ONLY_CHANGE_CODES_INSIDE_THE_BOX
###################################################################################################
folder = 'D:\\Gut Imaging\\Videos\\Temp\\d444243\\'  ##
segment_width = 30  ##
refresh_points = 0  ##
redraw = 0  ##
brightness_scale = 1.5  ##
# segment_point_scale = [1.2,1.2]
segment_point_scale = [1,1]
fine_tune_region = 1  ## 1: Manual region specification
fine = 0  ## 1: tune subregion manually; 0: subregion readjusted automatically
KCl_num = '4444k'                                                                                ##
###################################################################################################

def gut_section_exe(folder, segment_width, refresh_points, redraw, brightness_scale, fine_tune_region, fine):
    global mode
    global dummy
    global draw_pad
    global temp_points_holder
    global temp_lines_holder
    global right_points
    global left_points
    global right_lines
    global left_lines
    global left_segment_points
    global right_segment_points
    global mid_points
    global mid_lines
    global src_scale

    roi_np = read_igor_roi_matrix(folder + 'roi' + '.csv')
    # Check if Bright Field exists and if the size of Bright Field is equal to the ROI Matrix
    if os.path.isdir(folder + 'bf') and cv2.imread(folder + 'bf\\' + os.listdir(folder + 'bf')[0]).shape[
                                        :2] == roi_np.shape:
        bright_field_file = os.listdir(folder + 'bf')[0]
        bright_field = cv2.imread(folder + 'bf\\' + bright_field_file)
        bright_field = cv2.normalize(bright_field, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        draw_pad = bright_field * brightness_scale
        draw_pad = cv2.normalize(draw_pad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        r_channel = draw_pad[:, :, 0]
        g_channel = draw_pad[:, :, 1]
        b_channel = draw_pad[:, :, 2]
        r_channel[roi_np != 0] = 255
        g_channel[roi_np != 0] = 154
        b_channel[roi_np != 0] = 229
        draw_pad = np.stack((b_channel, g_channel, r_channel), axis=-1)

    else:
        draw_pad = roi_np.copy()
        draw_pad[draw_pad != 0] = 255
        draw_pad = cv2.normalize(draw_pad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        draw_pad = cv2.cvtColor(draw_pad, cv2.COLOR_GRAY2RGB)

    if draw_pad.shape[0] > draw_pad.shape[1]:
        # src_scale = draw_pad.shape[0] / (GetSystemMetrics(1) * 0.85)  # Make sure the full image can fit in the screen
        src_scale = draw_pad.shape[0] / 918
    else:
        # src_scale = draw_pad.shape[1] / (GetSystemMetrics(0) * 0.85)
        src_scale = draw_pad.shape[1] / 1632
        # if draw_pad.shape[0] / src_scale > (GetSystemMetrics(1) * 0.85):
        if draw_pad.shape[0] / src_scale > 918:
            # src_scale = draw_pad.shape[0] / (GetSystemMetrics(1) * 0.85)
            src_scale = draw_pad.shape[0] / 918

    draw_pad = cv2.resize(draw_pad, (
        np.rint(draw_pad.shape[1] / src_scale).astype(int), np.rint(draw_pad.shape[0] / src_scale).astype(int)))
    dummy = draw_pad.copy()

    temp_points_holder = np.empty((0, 2), int)
    temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])

    left_segment_points = np.empty((0, 2), int)
    right_segment_points = np.empty((0, 2), int)

    left_points = np.empty((0, 2), int)
    right_points = np.empty((0, 2), int)
    mid_points = np.empty((0, 2), int)
    left_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])
    right_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])
    mid_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])

    mode = 'left'

    if fine_tune_region:
        region_select(fine)

    elif (not (os.path.exists(folder + "segment_files\\left_points.txt") and os.path.exists(
            folder + "segment_files\\right_points.txt"))) or refresh_points or redraw:
        setManualPoints(segment_width, redraw)
    else:
        if os.path.exists(folder + "cell_position.pkl"):
            median_positions = pd.read_pickle(folder + 'cell_position.pkl')
        else:
            igor_roi = read_igor_roi_matrix(folder + 'roi' + '.csv')
            labeled_neurons_df = neuron_label_matrix2dataframe(igor_roi)
            median_positions = labeled_neurons_df.groupby('unique_id').median()
            median_positions.to_pickle(folder + 'cell_position.pkl')

        roi_df = pd.DataFrame()
        roi_df['coordinate_x'] = np.rint(median_positions['y'] / src_scale).astype(int)
        roi_df['coordinate_y'] = np.rint(median_positions['x'] / src_scale).astype(int)
        roi_df['ROI_ID'] = np.arange(0, median_positions.shape[0])
        roi_df['segment'] = np.nan

        roi_df = assign_segments(roi_df)
        roi_df = roi_df[['ROI_ID', 'segment', 'distance_to_origin']]
        roi_df['distance_to_origin'] = np.round(roi_df['distance_to_origin'], 1)
        roi_df.to_csv(folder + 'segment.csv', index=False)

def read_points(file, points, lines):
    loaded_points = np.loadtxt(file).astype(int)
    loaded_points[:,0] = loaded_points[:,0] * segment_point_scale[0]
    loaded_points[:, 1] = loaded_points[:, 1] * segment_point_scale[1]
    loaded_points = loaded_points.astype(int)
    for point_index in range(loaded_points.shape[0]):
        points, lines = add_point(loaded_points[point_index, :], points, lines)
    return points, lines


def add_point(coordinate, points, lines):
    coordinate = np.reshape(np.asarray(coordinate), (1, 2))
    points = np.append(points, np.array(coordinate), axis=0)
    if points.shape[0] >= 2:
        dist = math.hypot(points[-1, 0] - points[-2, 0], points[-1, 1] - points[-2, 1])
        lines = lines.append(
            {'line_ID': lines.shape[0], 'point1': points[-2, :], 'point2': points[-1, :], 'length': dist},
            ignore_index=True)
    return points, lines


def remove_point(points, lines):
    if points.shape[0] >= 2:
        lines.drop(lines.tail(1).index, inplace=True)
    points = np.delete(points, -1, 0)
    return points, lines


def setManualPoints(segment_width, redraw):
    global temp_points_holder
    global temp_lines_holder
    global mode
    global draw_pad

    if not os.path.exists(folder + 'segment_files'):
        os.makedirs(folder + 'segment_files')
    if os.path.exists(folder + "segment_files\\left_points.txt"):
        if redraw:
            os.remove(folder + "segment_files\\left_points.txt")
    else:
        redraw = 1
    if os.path.exists(folder + "segment_files\\right_points.txt"):
        if redraw:
            os.remove(folder + "segment_files\\right_points.txt")
    else:
        redraw = 1

    temp_points_holder = np.empty((0, 2), int)
    temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length'])

    def refresh():
        global temp_points_holder
        global temp_lines_holder
        global mode
        global draw_pad
        global dummy

        np.savetxt(folder + "segment_files\\{}_points.txt".format(mode), np.asarray(temp_points_holder))
        if mode == 'left':
            draw_pad = dummy.copy()
            temp_points_holder = np.empty((0, 2), int)
            temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length'])
            mode = 'right'
            read_saved_points()
            temp_points_holder = np.empty((0, 2), int)
            temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length'])
            mode = 'left'
            read_saved_points()
        elif mode == 'right':
            draw_pad = dummy.copy()
            temp_points_holder = np.empty((0, 2), int)
            temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length'])
            mode = 'left'
            read_saved_points()
            temp_points_holder = np.empty((0, 2), int)
            temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length'])
            mode = 'right'
            read_saved_points()

    def read_saved_points():
        global temp_points_holder
        global temp_lines_holder
        if os.path.exists(folder + "segment_files\\{}_points.txt".format(mode)):
            temp_read_np = np.loadtxt(folder + "segment_files\\{}_points.txt".format(mode)).astype(int).tolist()
            for p in temp_read_np:
                if mode == 'left':
                    cv2.circle(draw_pad, (p[0], p[1]), 2, (0, 255, 0), 2, cv2.LINE_AA)
                elif mode == 'right':
                    cv2.circle(draw_pad, (p[0], p[1]), 2, (255, 0, 0), 2, cv2.LINE_AA)
                temp_points_holder, temp_lines_holder = add_point(p, temp_points_holder, temp_lines_holder)
                if temp_lines_holder.shape[0] > 0:
                    LinkPoints(draw_pad, temp_lines_holder.iloc[-1]['point1'], temp_lines_holder.iloc[-1]['point2'])

    def drawAll(segment_width):
        global right_points
        global left_points
        global right_lines
        global left_lines
        global left_segment_points
        global right_segment_points
        global mid_points
        global mid_lines

        left_points = np.empty((0, 2), int)
        right_points = np.empty((0, 2), int)
        mid_points = np.empty((0, 2), int)

        left_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])
        right_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])
        mid_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])

        left_segment_points = np.empty((0, 2), int)
        right_segment_points = np.empty((0, 2), int)

        right_points, right_lines = read_points(folder + 'segment_files\\right_points.txt', right_points, right_lines)
        left_points, left_lines = read_points(folder + 'segment_files\\left_points.txt', left_points, left_lines)

        pt_num = np.min([left_lines.shape[0], right_lines.shape[0]])
        for line_index in range(pt_num):
            temp_left_segment_points = np.empty((0, 2), int)
            temp_right_segment_points = np.empty((0, 2), int)
            temp_left_segments = pd.DataFrame(columns=['segment_ID', 'point1', 'point2', 'length'])
            temp_right_segments = pd.DataFrame(columns=['segment_ID', 'point1', 'point2', 'length'])

            norm_length = left_lines.iloc[line_index]['length'] + right_lines.iloc[line_index]['length']
            num_segments = int(norm_length / segment_width)

            temp_left_segment_points, temp_left_segments = singleline_segmentation(left_lines.iloc[line_index],
                                                                                   temp_left_segment_points,
                                                                                   temp_left_segments,
                                                                                   num_segments)
            temp_right_segment_points, temp_right_segments = singleline_segmentation(right_lines.iloc[line_index],
                                                                                     temp_right_segment_points,
                                                                                     temp_right_segments, num_segments)
            drawSegmentPoints(draw_pad, temp_left_segment_points)
            drawSegmentPoints(draw_pad, temp_right_segment_points)

            if line_index > 0:
                temp_left_segment_points = np.delete(temp_left_segment_points, (0), axis=0)
                temp_right_segment_points = np.delete(temp_right_segment_points, (0), axis=0)
            left_segment_points = np.append(left_segment_points, temp_left_segment_points, axis=0)
            right_segment_points = np.append(right_segment_points, temp_right_segment_points, axis=0)

        assert left_segment_points.shape[0] == right_segment_points.shape[
            0], 'Number of Points on Each Side is not Equal.'
        np.save(folder + 'segment_files\\left_segment_points.npy', left_segment_points)
        np.save(folder + 'segment_files\\right_segment_points.npy', right_segment_points)

        for segment_point_index in range(left_segment_points.shape[0]):
            if segment_point_index == 0:
                RGB = (0, 255, 255)
            else:
                RGB = (255, 200, 0)
            LinkPoints(draw_pad, left_segment_points[segment_point_index, :],
                       right_segment_points[segment_point_index, :], RGB=RGB)
            mid_points, mid_lines = add_point(
                scaled_line(left_segment_points[segment_point_index, :], right_segment_points[segment_point_index, :],
                            0.5),
                mid_points, mid_lines)
            if mid_lines.shape[0] > 0:
                LinkPoints(draw_pad, mid_lines.iloc[-1]['point1'].astype(int), mid_lines.iloc[-1]['point2'].astype(int),
                           RGB=(0, 0, 255))
        mid_lines.to_pickle(folder + 'segment_files\\mid_lines.pkl')
        cv2.imwrite(folder + 'segment_image.jpeg', draw_pad)

    def drawCircle(action, x, y, flags, userdata):
        global temp_points_holder
        global temp_lines_holder
        global draw_pad

        if action == cv2.EVENT_LBUTTONUP:
            temp_points_holder, temp_lines_holder = add_point([x, y], temp_points_holder, temp_lines_holder)
            if mode == 'left':
                cv2.circle(draw_pad, (x, y), 2, (0, 255, 0), 2, cv2.LINE_AA)
            elif mode == 'right':
                cv2.circle(draw_pad, (x, y), 2, (255, 0, 0), 2, cv2.LINE_AA)
            if temp_lines_holder.shape[0] > 0:
                LinkPoints(draw_pad, temp_lines_holder.iloc[-1]['point1'], temp_lines_holder.iloc[-1]['point2'])
        elif action == cv2.EVENT_RBUTTONUP:
            temp_points_holder, temp_lines_holder = remove_point(temp_points_holder, temp_lines_holder)
            refresh()

    cv2.namedWindow("DrawPad")
    if redraw:
        cv2.setMouseCallback("DrawPad", drawCircle)
        k = 0
        while k != 27:

            cv2.imshow("DrawPad", draw_pad)

            k = cv2.waitKey(20) & 0xFF
            if k == 99:
                draw_pad = dummy.copy()
            if k == 32:
                np.savetxt(folder + "segment_files\\{}_points.txt".format(mode), np.asarray(temp_points_holder))
                draw_pad = dummy.copy()
                drawAll(segment_width)
            if k == 9:
                if mode == 'left':
                    refresh()
                    temp_points_holder = np.empty((0, 2), int)
                    temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length'])
                    mode = 'right'
                    read_saved_points()
                elif mode == 'right':
                    refresh()
                    temp_points_holder = np.empty((0, 2), int)
                    temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length'])
                    mode = 'left'
                    read_saved_points()
    else:
        drawAll(segment_width)
        cv2.imshow("DrawPad", draw_pad)
        cv2.waitKey(-1)
    cv2.destroyAllWindows()


def multilines_segmentation(lines, segment_points, segments, num_segments):
    lines['segment_point'] = lines['segment_point'].astype(object)
    previous_line_index = 0
    current_segment_indices = []
    segment_points, segments = add_point(lines.iloc[0]['point1'], segment_points, segments)
    current_segment_indices.append(0)

    lines_length = lines['length']
    total_length = lines_length.sum()
    section_length = total_length / num_segments
    for segment_index in range(num_segments - 1):
        cumulative_length = section_length * (segment_index + 1)
        for line_index in range(lines.shape[0]):
            if lines_length[0:line_index + 1].sum() > cumulative_length:
                if line_index > previous_line_index:
                    lines.at[previous_line_index, 'segment_point'] = np.asarray(current_segment_indices).astype(int)
                    current_segment_indices = []
                    previous_line_index = line_index
                line = lines.iloc[line_index]
                scale = (cumulative_length - lines_length[0:line_index].sum()) / line['length']
                segment_point = scaled_line(line['point1'], line['point2'], scale)
                segment_points, segments = add_point(segment_point, segment_points, segments)
                current_segment_indices.append(segment_index + 1)
                break
    segment_points, segments = add_point(lines.iloc[-1]['point2'], segment_points, segments)
    current_segment_indices.append(num_segments)
    lines.at[lines.shape[0] - 1, 'segment_point'] = np.asarray([19, 20]).astype(int)
    segment_points = segment_points.astype(int)
    return lines, segment_points, segments


def singleline_segmentation(line, segment_points, segments, num_segments):
    segment_points, segments = add_point(line['point1'], segment_points, segments)
    total_length = line['length']
    section_length = total_length / num_segments
    for segment_index in range(num_segments - 1):
        cumulative_length = section_length * (segment_index + 1)
        scale = cumulative_length / line['length']
        segment_point = scaled_line(line['point1'], line['point2'], scale)
        segment_points, segments = add_point(segment_point, segment_points, segments)
    segment_points, segments = add_point(line['point2'], segment_points, segments)
    segment_points = segment_points.astype(int)
    return segment_points, segments


def assign_segments(roi_df):
    left_segment_points = np.load(folder + 'segment_files\\left_segment_points.npy')
    right_segment_points = np.load(folder + 'segment_files\\right_segment_points.npy')
    mid_lines = pd.read_pickle(folder + 'segment_files\\mid_lines.pkl')
    roi_on_boundary = {}

    for segment_index in range(left_segment_points.shape[0] - 1):
        section_points = np.empty((0, 2), int)
        section_points = np.append(section_points, left_segment_points[segment_index, :].reshape(1, 2), axis=0)
        section_points = np.append(section_points, left_segment_points[segment_index + 1, :].reshape(1, 2), axis=0)
        section_points = np.append(section_points, right_segment_points[segment_index + 1, :].reshape(1, 2), axis=0)
        section_points = np.append(section_points, right_segment_points[segment_index, :].reshape(1, 2), axis=0)

        test_df = roi_df[roi_df.isnull().any(axis=1)]
        test_points = np.concatenate((test_df['coordinate_x'].values, test_df['coordinate_y'].values)).reshape(2,
                                                                                                               -1).transpose()
        results = in_area(draw_pad, section_points, test_points)
        for result_index in range(results.shape[0]):
            roi_id = int(test_df.iloc[result_index]['ROI_ID'])
            if results[result_index, :] == 1:
                roi_df.at[roi_df['ROI_ID'] == roi_id, 'segment'] = segment_index
            elif results[result_index, :] == 0:

                if str(roi_id) in roi_on_boundary:
                    updated_list = roi_on_boundary[str(roi_id)]
                    updated_list.append(segment_index)
                    roi_on_boundary.update({str(roi_id): updated_list})
                else:
                    roi_on_boundary.update({str(roi_id): [segment_index]})

    for key, value in roi_on_boundary.items():
        roi_df.at[roi_df['ROI_ID'] == int(key), 'segment'] = np.asarray(value).mean()

    roi_df['distance_to_origin'] = np.nan
    origin_section = 0

    for index, row in roi_df.iterrows():
        if not np.isnan(row['segment']):
            if str(row['segment']).split('.')[1][0] == '0':
                distance = mid_lines.iloc[int(row['segment'])]['length'] * 0.5 + \
                           mid_lines.iloc[origin_section:int(row['segment'])][
                               'length'].sum()
            else:
                distance = mid_lines.iloc[origin_section:int(row['segment']) + 1]['length'].sum()
            roi_df.at[index, 'distance_to_origin'] = distance
    return roi_df


def region_select(fine=0):
    global temp_points_holder
    global temp_lines_holder
    global temp_region_coord_holder
    global mode
    global draw_pad
    global current_manual_segment
    global seg_num
    global AP
    global region_start_track

    def segment_update(folder, cumulative_AP):
        # def distance2origin_new(segment):
        #     return mid_lines.iloc[int(segment)]['length'] * 0.5 + mid_lines.iloc[:int(segment)]['length'].sum()

        segment_df = pd.read_csv(folder + 'segment.csv').dropna().reset_index(drop=True)
        # AP = np.apply_along_axis(distance2origin_new, 0, segment_df['segment'].values.reshape(1,-1))
        AP = segment_df['distance_to_origin'].values

        segment_df['cum_AP'] = seg2AP(AP)
        segment_df['region'] = np.nan
        for key_ind in range(len(cumulative_AP)):
            if key_ind == 0:
                start = 0
            else:
                start = list(cumulative_AP.values())[key_ind - 1]
            end = list(cumulative_AP.values())[key_ind]
            segment_df.at[(segment_df['cum_AP'] >= start ) & (segment_df['cum_AP'] < end ), 'region'] = \
            list(cumulative_AP.keys())[key_ind]
        segment_df.at[segment_df['cum_AP'] >= end, 'region'] = list(cumulative_AP.keys())[key_ind]
        segment_df.to_csv(folder + '{}_manual_segment.csv'.format(folder.split('\\')[-2][1:]), index=False)
        segment_df.to_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\manual_segment\\{}_manual_segment.csv'.format(folder.split('\\')[-2][1:]), index=False)


    def seg2AP(a):
        mid_lines = pd.read_pickle(folder + 'segment_files\\mid_lines.pkl')
        # AP_min = (mid_lines.iloc[1]['length'] * 0.5)
        AP_min = 0
        # AP_max = (mid_lines.iloc[-1]['length'] * 0.5 + mid_lines.iloc[:-1]['length'].sum())
        AP_max = mid_lines['length'].sum()

        a = a/(AP_max-AP_min)
        return a


    assert os.path.exists(folder + "segment_files\\left_segment_points.npy") and os.path.exists(
        folder + "segment_files\\right_segment_points.npy"), 'No Segment Points Files'

    temp_region_coord_holder = np.empty((0, 2), int)
    cell_position = pd.read_pickle(folder + 'cell_position.pkl')

    references = pd.read_excel('D:\\Gut Imaging\\Videos\\CommonFiles\\AP_reference.xlsx', index_col=0).iloc[1:,
                 1].values/100
    noncum_references = pd.read_excel('D:\\Gut Imaging\\Videos\\CommonFiles\\AP_reference.xlsx', index_col=0).iloc[1:,
                 0].values
    segment_df = pd.read_csv(folder + 'segment.csv').dropna().reset_index(drop=True)
    AP = segment_df['distance_to_origin'].values
    AP = seg2AP(AP)
    for reference in references:
        temp_region_coord_holder = np.append(temp_region_coord_holder, cell_position.iloc[segment_df.iloc[np.argmin(
            np.abs(AP - reference)), 0], :2].values.reshape(1, 2), axis=0)
    colored_position = np.array([0,2,3,6,7])
    colored_position = np.append(colored_position, colored_position+10)
    colored = np.zeros((20,))
    colored[colored_position] = 1

    cumulative_AP = dict.fromkeys(['R5b', 'R5a', 'R4c', 'R4b', 'R4a', 'R3', 'R2c', 'R2b', 'R2a', 'R1'])
    dict_key_index = np.arange(len(cumulative_AP))
    dict_key_index = np.append(dict_key_index, dict_key_index)
    temp_region_coord_holder = temp_region_coord_holder[:,[1,0]] / src_scale


    seg_num = temp_region_coord_holder.shape[0]
    current_manual_segment = 0
    temp_points_holder = np.empty((0, 2), int)
    temp_lines_holder = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length'])
    region_start_track = [0]

    left_segment_points = np.load(folder + 'segment_files\\left_segment_points.npy')
    right_segment_points = np.load(folder + 'segment_files\\right_segment_points.npy')

    edge_pad = 2

    left_segment_points[left_segment_points == 0] = edge_pad
    right_segment_points[right_segment_points == 0] = edge_pad
    left_segment_points[left_segment_points == 0] = edge_pad
    right_segment_points[right_segment_points == 0] = edge_pad

    left_segment_points[:, 0][left_segment_points[:, 0] == draw_pad.shape[1] - 1] = draw_pad.shape[1] - 1 - edge_pad
    right_segment_points[:, 0][right_segment_points[:, 0] == draw_pad.shape[1] - 1] = draw_pad.shape[1] - 1 - edge_pad
    left_segment_points[:, 1][left_segment_points[:, 1] == draw_pad.shape[0] - 1] = draw_pad.shape[0] - 1 - edge_pad
    right_segment_points[:, 1][right_segment_points[:, 1] == draw_pad.shape[0] - 1] = draw_pad.shape[0] - 1 - edge_pad

    def drawAll(segment_width):
        global right_points
        global left_points
        global right_lines
        global left_lines
        global left_segment_points
        global right_segment_points
        global mid_points
        global mid_lines

        left_points = np.empty((0, 2), int)
        right_points = np.empty((0, 2), int)
        mid_points = np.empty((0, 2), int)

        left_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])
        right_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])
        mid_lines = pd.DataFrame(columns=['line_ID', 'point1', 'point2', 'length', 'segment_point'])

        left_segment_points = np.empty((0, 2), int)
        right_segment_points = np.empty((0, 2), int)

        right_points, right_lines = read_points(folder + 'segment_files\\right_points.txt', right_points, right_lines)
        left_points, left_lines = read_points(folder + 'segment_files\\left_points.txt', left_points, left_lines)

        pt_num = np.min([left_lines.shape[0], right_lines.shape[0]])
        for line_index in range(pt_num):
            temp_left_segment_points = np.empty((0, 2), int)
            temp_right_segment_points = np.empty((0, 2), int)
            temp_left_segments = pd.DataFrame(columns=['segment_ID', 'point1', 'point2', 'length'])
            temp_right_segments = pd.DataFrame(columns=['segment_ID', 'point1', 'point2', 'length'])

            norm_length = left_lines.iloc[line_index]['length'] + right_lines.iloc[line_index]['length']
            num_segments = int(norm_length / segment_width)

            temp_left_segment_points, temp_left_segments = singleline_segmentation(left_lines.iloc[line_index],
                                                                                   temp_left_segment_points,
                                                                                   temp_left_segments,
                                                                                   num_segments)
            temp_right_segment_points, temp_right_segments = singleline_segmentation(right_lines.iloc[line_index],
                                                                                     temp_right_segment_points,
                                                                                     temp_right_segments, num_segments)
            drawSegmentPoints(draw_pad, temp_left_segment_points)
            drawSegmentPoints(draw_pad, temp_right_segment_points)

            if line_index > 0:
                temp_left_segment_points = np.delete(temp_left_segment_points, (0), axis=0)
                temp_right_segment_points = np.delete(temp_right_segment_points, (0), axis=0)
            left_segment_points = np.append(left_segment_points, temp_left_segment_points, axis=0)
            right_segment_points = np.append(right_segment_points, temp_right_segment_points, axis=0)

        assert left_segment_points.shape[0] == right_segment_points.shape[
            0], 'Number of Points on Each Side is not Equal.'
        np.save(folder + 'segment_files\\left_segment_points.npy', left_segment_points)
        np.save(folder + 'segment_files\\right_segment_points.npy', right_segment_points)

        for segment_point_index in range(left_segment_points.shape[0]):
            if segment_point_index == 0:
                RGB = (0, 255, 255)
            else:
                RGB = (255, 200, 0)
            LinkPoints(draw_pad, left_segment_points[segment_point_index, :],
                       right_segment_points[segment_point_index, :], RGB=RGB)
            mid_points, mid_lines = add_point(
                scaled_line(left_segment_points[segment_point_index, :], right_segment_points[segment_point_index, :],
                            0.5),
                mid_points, mid_lines)
            if mid_lines.shape[0] > 0:
                LinkPoints(draw_pad, mid_lines.iloc[-1]['point1'].astype(int), mid_lines.iloc[-1]['point2'].astype(int),
                           RGB=(0, 0, 255))
        mid_lines.to_pickle(folder + 'segment_files\\mid_lines.pkl')
        cv2.imwrite(folder + 'segment_image.jpeg', draw_pad)

    def findSegment(x,y):
        for segment_index in range(left_segment_points.shape[0] - 1):
            section_points = np.empty((0, 2), int)
            section_points = np.append(section_points, left_segment_points[segment_index, :].reshape(1, 2),
                                       axis=0)
            section_points = np.append(section_points,
                                       left_segment_points[segment_index + 1, :].reshape(1, 2), axis=0)
            section_points = np.append(section_points,
                                       right_segment_points[segment_index + 1, :].reshape(1, 2), axis=0)
            section_points = np.append(section_points, right_segment_points[segment_index, :].reshape(1, 2),
                                       axis=0)



            test_points = np.array([x,y]).reshape((1, 2))
            results = in_area(draw_pad, section_points, test_points)
            if results == 1:
                return segment_index, section_points
        return None

    def fillSegment():
        global temp_region_coord_holder
        global current_manual_segment
        global seg_num
        global draw_pad
        global mid_lines

        draw_pad = drawed_dummy.copy()
        for region_coord_index in range(temp_region_coord_holder.shape[0])[
                                  current_manual_segment:seg_num + current_manual_segment]:
            segment_index, section_points = findSegment(temp_region_coord_holder[region_coord_index,0], temp_region_coord_holder[region_coord_index,1])
            if isinstance(segment_index, int):
                distance = seg2AP((mid_lines.iloc[segment_index]['length'] * 0.5 + mid_lines.iloc[:segment_index]['length'].sum()))
                cumulative_AP.update({list(cumulative_AP.keys())[dict_key_index[region_coord_index]]: distance})
                if colored[region_coord_index]:
                    cv2.fillPoly(draw_pad, pts=[section_points], color=(8,122,236))
                    # pass
                else:
                    cv2.fillPoly(draw_pad, pts=[section_points], color=(255, 255, 255))

        for key_ind, key in enumerate(cumulative_AP):
            if cumulative_AP[key] <= 1:
                cv2.putText(draw_pad, text='{}: {}%'.format(key, np.round(cumulative_AP[key]*100, 1)), org=(750, 50 + key_ind*30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color=(255, 255, 255))
            else:
                cv2.putText(draw_pad, text='{}: {}%'.format(key, np.round(cumulative_AP[key]*100, 1)),
                            org=(750, 50 + key_ind * 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(255, 255, 255))

    def region_select_callback(action, x, y, flags, userdata):
        global temp_region_coord_holder
        global draw_pad
        global current_manual_segment
        global AP
        global mid_lines
        global region_start_track

        def auto_append(x,y, region_start, region_ratio):
            global temp_region_coord_holder
            global current_manual_segment

            segment_index, section_points = findSegment(x, y)
            clicked_length = seg2AP((mid_lines.iloc[segment_index]['length'] * 0.5 +
                               mid_lines.iloc[:segment_index]['length'].sum()))
            distance = clicked_length - region_start
            for ratio_index in range(len(region_ratio)-1):
                target = distance / np.sum(region_ratio) * region_ratio[ratio_index] + region_start
                region_start = target
                coordinate = cell_position.iloc[segment_df.iloc[np.argmin(np.abs(AP - target)), 0], :2].values.reshape(1, 2)
                coordinate = ((1 / src_scale) * coordinate[0, ::-1]).reshape((1, 2))
                temp_region_coord_holder = np.append(temp_region_coord_holder, np.array(coordinate), axis=0)
                current_manual_segment += 1
            coordinate = np.reshape(np.asarray([x, y]), (1, 2))
            temp_region_coord_holder = np.append(temp_region_coord_holder, np.array(coordinate), axis=0)
            current_manual_segment += 1
            return clicked_length

        if action == cv2.EVENT_LBUTTONUP:
            if current_manual_segment < 10:
                if fine:
                    coordinate = np.reshape(np.asarray([x, y]), (1, 2))
                    temp_region_coord_holder = np.append(temp_region_coord_holder, np.array(coordinate), axis=0)
                    current_manual_segment += 1
                else:
                    if current_manual_segment == 0:
                        region_start_track.append(auto_append(x, y, region_start_track[-1], noncum_references[0:2]))
                    elif current_manual_segment == 2:
                        region_start_track.append(auto_append(x, y, region_start_track[-1], noncum_references[2:5]))
                    elif current_manual_segment == 5:
                        coordinate = np.reshape(np.asarray([x, y]), (1, 2))
                        temp_region_coord_holder = np.append(temp_region_coord_holder, np.array(coordinate), axis=0)
                        current_manual_segment += 1
                        segment_index, section_points = findSegment(x, y)
                        region_start_track.append(seg2AP((mid_lines.iloc[segment_index]['length'] * 0.5 +
                                                 mid_lines.iloc[:segment_index]['length'].sum())))
                    elif current_manual_segment == 6:
                        region_start_track.append(auto_append(x, y, region_start_track[-1], noncum_references[6:9]))
                    elif current_manual_segment == 9:
                        coordinate = np.reshape(np.asarray([x, y]), (1, 2))
                        temp_region_coord_holder = np.append(temp_region_coord_holder, np.array(coordinate), axis=0)
                        current_manual_segment += 1
                        segment_index, section_points = findSegment(x, y)
                        region_start_track.append(seg2AP((mid_lines.iloc[segment_index]['length'] * 0.5 +
                                                          mid_lines.iloc[:segment_index]['length'].sum())))

            fillSegment()

        elif action == cv2.EVENT_RBUTTONUP:
            if current_manual_segment > 0:
                if fine:
                    temp_region_coord_holder = np.delete(temp_region_coord_holder, -1, 0)
                    current_manual_segment -= 1
                else:
                    if current_manual_segment == 10:
                        temp_region_coord_holder = temp_region_coord_holder[:-1,:]
                        region_start_track = region_start_track[:-1]
                        current_manual_segment -= 1
                    elif current_manual_segment == 9:
                        temp_region_coord_holder = temp_region_coord_holder[:-3,:]
                        region_start_track = region_start_track[:-1]
                        current_manual_segment -= 3
                    elif current_manual_segment == 6:
                        temp_region_coord_holder = temp_region_coord_holder[:-1, :]
                        region_start_track = region_start_track[:-1]
                        current_manual_segment -= 1
                    elif current_manual_segment == 5:
                        temp_region_coord_holder = temp_region_coord_holder[:-3,:]
                        region_start_track = region_start_track[:-1]
                        current_manual_segment -= 3
                    elif current_manual_segment == 2:
                        temp_region_coord_holder = temp_region_coord_holder[:-2,:]
                        region_start_track = region_start_track[:-1]
                        current_manual_segment -= 2
            fillSegment()

    cv2.namedWindow("DrawPad")
    cv2.setMouseCallback("DrawPad", region_select_callback)
    drawAll(segment_width)

    drawed_dummy = draw_pad.copy()
    fillSegment()

    k = 0
    while k != 27:
        cv2.imshow("DrawPad", draw_pad)
        k = cv2.waitKey(20)

    cv2.destroyAllWindows()

    segment_update(folder, cumulative_AP)
    pd.DataFrame.from_dict(cumulative_AP, orient='index').to_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\region_boundary\\{}_region_boundary.csv'.format(folder.split('\\')[-2][1:]))
    cv2.imwrite(folder + 'manual_segment_reference.png', draw_pad)

    KCl_dir = '\\'.join(folder.split('\\')[:-2])+ '\\d{}\\'.format(KCl_num)
    if os.path.isdir(KCl_dir):
        fromDirectory = folder + '\\segment_files'
        toDirectory = KCl_dir + 'segment_files'
        copy_tree(fromDirectory, toDirectory)
        gut_section_exe(KCl_dir, segment_width, 0, 0, brightness_scale, 0, 0)
        segment_update(KCl_dir, cumulative_AP)
        KCl_ROI_df = pd.read_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\manual_segment\\{}_manual_segment.csv'.format(KCl_num))
        ROI_df = pd.read_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\manual_segment\\{}_manual_segment.csv'.format(folder.split('\\')[-2][1:]))

        # MC_ROI_df = pd.read_csv('D:\\Gut Imaging\\Videos\\temp_f_file\\f_{}.txt'.format(folder.split('\\')[-2][1:]), sep='\t')
        MC_ROI_df = pd.read_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\dff_file\\DFF_{}_Ave.txt'.format(folder.split('\\')[-2][1:]),
                                sep='\t')
        # MC_ROI_index = np.array(list(MC_ROI_df.columns)[1:]).astype(int).flatten()
        MC_ROI_index = [ROI_index.split('_')[-1] for ROI_index in MC_ROI_df.columns[2:]]
        KCL_count = KCl_ROI_df.groupby(['region'])['ROI_ID'].count()
        # MC_throw_ROI_df = ROI_df[~ROI_df['ROI_ID'].isin(MC_ROI_index)]
        MC_keep_ROI_df = ROI_df[ROI_df['ROI_ID'].isin(MC_ROI_index)]

        # if MC_throw_ROI_df.shape[0] == 0:
        #     MC_count = np.zeros((len(KCL_count),))
        #     count_df = pd.DataFrame(dict(KCl=KCL_count)).reset_index()
        #     count_df['MC_throw'] = MC_count
        # else:
        #     MC_count = MC_throw_ROI_df.groupby(['region'])['ROI_ID'].count()
        #     count_df = pd.DataFrame(dict(KCl=KCL_count, MC_throw=MC_count)).reset_index()
        MC_count = MC_keep_ROI_df.groupby(['region'])['ROI_ID'].count()
        count_df = pd.DataFrame(dict(KCl=KCL_count, kept=MC_count)).reset_index()
        try:
            count_df = count_df.set_index('region')
        except KeyError:
            count_df = count_df.set_index('index')
        count_df = count_df.fillna(0)
        # if count_df['MC_throw'].isnull().values.any() and count_df['KCl'].isnull().values.any():
        #     count_df.set_index('index')
        #     count_df = count_df.fillna(0)
        # elif count_df['MC_throw'].isnull().values.any():
        #     count_df.set_index('index')
        #     count_df['MC_throw'] = count_df['MC_throw'].fillna(0)
        # elif count_df['KCl'].isnull().values.any():
        #     count_df.set_index('index')
        #     count_df['KCl'] = count_df['KCl'].fillna(0)
        # else:
        #     count_df.set_index('region')
        # count_df['kept'] = count_df['KCl'] - count_df['MC_throw']
        # count_df = count_df[['KCl', 'MC_throw', 'kept']].astype(int)
        count_df['MC+dim'] = count_df['KCl'] - count_df['kept']
        count_df = count_df[['KCl', 'MC+dim', 'kept']].astype(int)
        count_df.to_csv(folder + '\\{}_region_count.csv'.format(folder.split('\\')[-2][1:]))
        count_df.to_csv('D:\\Gut Imaging\\Videos\\CommonFiles\\region_count\\{}_region_count.csv'.format(folder.split('\\')[-2][1:]))
    else:
        print('KCl folder not found. No cell count')

if __name__ == '__main__':
    gut_section_exe(folder, segment_width, refresh_points, redraw, brightness_scale, fine_tune_region, fine)