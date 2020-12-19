from video_util import read_tif
import numpy as np
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac
from skimage import io
import cv2
from win32api import GetSystemMetrics
from skimage.util import img_as_uint
import pandas as pd
import os

folder = 'D:\\Gut Imaging\\Videos\\Alignment\\'


def align_video(stack_src, stack_dst):
    global draw_pad
    global temp_points_holder

    small_window_size = 450

    stack_src_max = np.clip(np.max(stack_src, axis=0), 300, 700)
    stack_dst_max = np.clip(np.max(stack_dst, axis=0), 300, 700)

    assert stack_src_max.shape == stack_dst_max.shape

    stack_src_max = cv2.normalize(stack_src_max, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    stack_dst_max = cv2.normalize(stack_dst_max, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    zeros = cv2.normalize(np.zeros(stack_src_max.shape).astype(int), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    draw_pad = np.stack((zeros, stack_src_max, stack_dst_max))
    draw_pad = np.swapaxes(draw_pad, 0, 2)

    if draw_pad.shape[0] > draw_pad.shape[1]:
        src_scale = draw_pad.shape[0] / (GetSystemMetrics(1) * 0.85)  # Make sure the full image can fit in the screen
    else:
        src_scale = draw_pad.shape[1] / (GetSystemMetrics(0) * 0.85)
        if draw_pad.shape[0] / src_scale > (GetSystemMetrics(1) * 0.85):
            src_scale = draw_pad.shape[0] / (GetSystemMetrics(1) * 0.85)
    draw_pad = cv2.resize(draw_pad, (
        np.rint(draw_pad.shape[1] / src_scale).astype(int), np.rint(draw_pad.shape[0] / src_scale).astype(int)))

    dummy = draw_pad.copy()

    temp_points_holder = np.empty((0, 2), int)
    pair_holder = np.empty((2, 2, 0), int)
    warped_stack = np.empty(stack_dst.shape)

    def MouseCallback(action, x, y, flags, userdata):
        global temp_points_holder
        global draw_pad

        zoom_scale = 10

        if action == cv2.EVENT_LBUTTONUP:
            if temp_points_holder.shape[0] < 2:
                coordinate = (np.reshape(np.asarray([x, y]), (1, 2))).astype(int)
                temp_points_holder = np.append(temp_points_holder, np.array(coordinate), axis=0)
                print('The pair currently has {} point'.format(temp_points_holder.shape[0]))
            else:
                print('The pair already has two points')

        elif action == cv2.EVENT_RBUTTONUP:
            if temp_points_holder.shape[0] == 0:
                print('There is no point left to be removed')
            else:
                temp_points_holder = np.delete(temp_points_holder, -1, 0)
            print('The pair currently has {} point'.format(temp_points_holder.shape[0]))

        elif action == cv2.EVENT_MOUSEMOVE:
            try:
                rescaled_crop = cv2.resize(draw_pad[y - int(small_window_size / 2 / zoom_scale):y + int(
                    small_window_size / 2 / zoom_scale) + 1,
                                           x - int(small_window_size / 2 / zoom_scale):x + int(
                                               small_window_size / 2 / zoom_scale) + 1],
                                           (small_window_size, small_window_size))
                cv2.circle(rescaled_crop, (int(small_window_size / 2), int(small_window_size / 2)), 2, (255, 255, 255),
                           1, cv2.LINE_AA)
                cv2.imshow("Zoom", rescaled_crop)
            except cv2.error:
                pass

    cv2.namedWindow("DrawPad")
    cv2.moveWindow("DrawPad", 950, 20)
    cv2.namedWindow("Align")
    cv2.moveWindow("Align", 10, 20)
    cv2.resizeWindow('Align', 700, 700)
    cv2.namedWindow("Zoom")
    cv2.moveWindow("Zoom", 400, 30)
    cv2.resizeWindow('Zoom', small_window_size, small_window_size)
    cv2.setMouseCallback("DrawPad", MouseCallback)
    k = 0
    while k != 27:
        draw_pad = dummy.copy()
        if pair_holder.shape[2] > 0:
            for i in range(pair_holder.shape[2]):
                cv2.circle(draw_pad, (pair_holder[0, 0, i], pair_holder[0, 1, i]), 2, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(draw_pad, (pair_holder[1, 0, i], pair_holder[1, 1, i]), 2, (255, 255, 255), 1, cv2.LINE_AA)
        if temp_points_holder.shape[0] > 0:
            for i in range(temp_points_holder.shape[0]):
                if i == 0:
                    color = (255, 0, 0)
                elif i == 1:
                    color = (255, 255, 255)
                cv2.circle(draw_pad, (temp_points_holder[i, 0], temp_points_holder[i, 1]), 2, color, 1, cv2.LINE_AA)
        cv2.imshow("DrawPad", draw_pad)

        k = cv2.waitKey(100) & 0xFF
        if k == 99:
            draw_pad = dummy.copy()
        if k == 61:
            if temp_points_holder.shape[0] == 2:
                pair = (np.reshape(temp_points_holder, (2, 2, 1))).astype(int)
                pair_holder = np.append(pair_holder, np.array(pair), axis=2)
                temp_points_holder = np.empty((0, 2), int)
            else:
                print('The pair does not have enough points')
        if k == 45:
            if pair_holder.shape[2] > 0:
                pair_holder = np.delete(pair_holder, -1, 2)
            else:
                print('No more pair to be removed')

        if pair_holder.shape[2] > 3:
            src = np.empty((0, 2), int)
            dst = np.empty((0, 2), int)
            for i in range(pair_holder.shape[2]):
                src = np.append(src, pair_holder[0, :, i].reshape(1, 2), axis=0)
                dst = np.append(dst, pair_holder[1, :, i].reshape(1, 2), axis=0)
            src = (src * src_scale).astype(int)
            src[:, [0, 1]] = src[:, [1, 0]]
            dst = (dst * src_scale).astype(int)
            dst[:, [0, 1]] = dst[:, [1, 0]]

            # robustly estimate affine transform model with RANSAC
            model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                           residual_threshold=2, max_trials=100)
            outliers = inliers == False
            tform = AffineTransform(scale=model_robust.scale, rotation=model_robust.rotation,
                                    translation=model_robust.translation)

            img_warped = warp(stack_dst_max, tform)
            img_warped = cv2.normalize(img_warped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

            align_pad = np.stack((zeros, stack_src_max, img_warped))
            align_pad = np.swapaxes(align_pad, 0, 2)
            align_pad = cv2.resize(align_pad, (700, 700))
            cv2.imshow("Align", align_pad)
        if k == 13:
            print('Aligning...')
            cv2.destroyAllWindows()
            for frame_index in range(stack_dst.shape[0]):
                frame_warped = warp(stack_dst[frame_index, :, :], tform)
                warped_stack[frame_index, :, :] = frame_warped
                print('{} %'.format(np.round(frame_index / stack_dst.shape[0] * 100, 1)))
            warped_stack = img_as_uint(warped_stack)
            concat_stack = np.append(stack_src, warped_stack, axis=0)
            return concat_stack


video_name_ls = []
directory = os.fsencode(folder)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".tif"):
        video_name_ls.append(filename)

if video_name_ls[0][:2] == video_name_ls[-1][:2]:
    excel_file_name = video_name_ls[0].split('.')[0] + video_name_ls[-1].split('.')[0][-2:] + '.xlsx'
else:
    excel_file_name = video_name_ls[0].split('.')[0] + video_name_ls[-1].split('.')[0] + '.xlsx'
meta = pd.read_excel(folder + excel_file_name, header=None)
meta['segment'] = np.nan

video_index_ls = np.array([int(video_index.split('.')[0]) for video_index in video_name_ls])
assert np.array_equal(video_index_ls, meta.iloc[:, 0].values), 'Meta File and Video names does not match'

video_ls = []
for video_name_ls_index in range(len(video_name_ls)):
    video_ls.append(read_tif(folder + video_name_ls[video_name_ls_index]))

assert len(np.unique(np.array([video.shape[1] for video in video_ls]))) == 1, 'Video dimensions do not agree'
assert len(np.unique(np.array([video.shape[2] for video in video_ls]))) == 1, 'Video dimensions do not agree'

segment = 0
meta.at[0, 'segment'] = segment
for video_ls_index in range(len(video_ls) - 1):
    cv2.namedWindow("AlignCheck")
    cv2.moveWindow("AlignCheck", 950, 20)

    check_stack_src_max = np.clip(np.max(video_ls[video_ls_index], axis=0), 300, 700)
    check_stack_dst_max = np.clip(np.max(video_ls[video_ls_index + 1], axis=0), 300, 700)

    assert check_stack_src_max.shape == check_stack_dst_max.shape

    check_stack_src_max = cv2.normalize(check_stack_src_max, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    check_stack_dst_max = cv2.normalize(check_stack_dst_max, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    zeros = cv2.normalize(np.zeros(check_stack_src_max.shape).astype(int), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    check_draw_pad = np.stack((zeros, check_stack_src_max, check_stack_dst_max))
    check_draw_pad = np.swapaxes(check_draw_pad, 0, 2)

    if check_draw_pad.shape[0] > check_draw_pad.shape[1]:
        src_scale = check_draw_pad.shape[0] / (
                GetSystemMetrics(1) * 0.85)  # Make sure the full image can fit in the screen
    else:
        src_scale = check_draw_pad.shape[1] / (GetSystemMetrics(0) * 0.85)
        if check_draw_pad.shape[0] / src_scale > (GetSystemMetrics(1) * 0.85):
            src_scale = check_draw_pad.shape[0] / (GetSystemMetrics(1) * 0.85)
    check_draw_pad = cv2.resize(check_draw_pad, (
        np.rint(check_draw_pad.shape[1] / src_scale).astype(int),
        np.rint(check_draw_pad.shape[0] / src_scale).astype(int)))
    k = 0
    while k != 27:
        cv2.imshow("AlignCheck", check_draw_pad)
        k = cv2.waitKey(500)
        if k == 13:
            segment += 1
            meta.at[video_ls_index + 1, 'segment'] = segment
            cv2.destroyAllWindows()
            break
        elif k == 32:
            meta.at[video_ls_index + 1, 'segment'] = segment
            cv2.destroyAllWindows()
            break

segment_ls = []
for segment in meta['segment'].unique():
    temp_segment_holder = np.empty((0, video_ls[0].shape[1], video_ls[0].shape[2])).astype('uint16')
    segment_meta = meta[meta['segment'] == segment]
    for segment_video_index in segment_meta.index:
        temp_video = np.reshape(video_ls[segment_video_index], (-1, video_ls[0].shape[1], video_ls[0].shape[2]))
        temp_segment_holder = np.append(temp_segment_holder, temp_video, axis=0)
    segment_ls.append(temp_segment_holder)

if len(segment_ls) == 1:
    aligned_video = segment_ls
else:
    segmeng_count = 0
    prev_video = segment_ls[segmeng_count]
    while segmeng_count + 1 < len(segment_ls):
        next_video = segment_ls[segmeng_count + 1]
        a = align_video(prev_video, next_video)
        prev_video = a
        segmeng_count += 1
    aligned_video = prev_video

time_stamp_ls = np.insert(np.cumsum(meta.iloc[:, 2].values), 0, 0)
video_ls = []
for time_stamp_index in range(len(time_stamp_ls) - 1):
    video_ls.append(aligned_video[time_stamp_ls[time_stamp_index]:time_stamp_ls[time_stamp_index + 1]])

output_video = np.empty((0, video_ls[0].shape[1], video_ls[0].shape[2])).astype('uint16')
for final_order_index in meta.iloc[:, 1].values - 1:
    temp_video = np.reshape(video_ls[final_order_index], (-1, video_ls[0].shape[1], video_ls[0].shape[2]))
    output_video = np.append(output_video, np.array(temp_video), axis=0)

io.imsave(folder + excel_file_name.split('.')[0] + '.tif', output_video)
