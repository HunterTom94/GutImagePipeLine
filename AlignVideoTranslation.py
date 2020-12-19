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

if __name__ == '__main__':
    folder = 'D:\\Gut Imaging\\Videos\\Alignment\\'

    concatenate_flag = 0

    def x_move(src, offset):
        offset = int(np.rint(offset))
        pad = np.zeros(src.shape)
        if offset>0:
            pad[:,offset:] = src[:,:-offset]
        elif offset < 0:
            pad[:, :offset] = src[:, -offset:]
        else:
            pad = src
        return pad

    def y_move(src, offset):
        offset = int(np.rint(offset))
        offset = -offset
        pad = np.zeros(src.shape)
        if offset>0:
            pad[offset:,:] = src[:-offset,:]
        elif offset < 0:
            pad[:offset,:] = src[-offset:,:]
        else:
            pad = src
        return pad

    def rotation(src, rot):
        (h, w) = src.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rot, 1)
        return cv2.warpAffine(src, M, (w, h))


    def align_video(stack_src, stack_dst):
        x_tracker = 0
        y_tracker = 0
        rot_tracker = 0

        stack_src_max = np.clip(np.max(stack_src, axis=0), 300, 700).transpose()
        stack_dst_max = np.clip(np.max(stack_dst, axis=0), 300, 700).transpose()

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
        src_dummy = draw_pad[:, :, 1].copy()

        warped_stack = np.empty(stack_dst.shape)

        cv2.namedWindow("DrawPad")
        cv2.moveWindow("DrawPad", 950, 20)
        k = 0
        while k != 27:
            moved_src = src_dummy.copy()
            draw_pad = dummy.copy()

            moved_src = x_move(moved_src, x_tracker)
            moved_src = y_move(moved_src, y_tracker)
            moved_src = rotation(moved_src, rot_tracker)

            draw_pad[:, :, 1] = moved_src

            cv2.imshow("DrawPad", draw_pad)

            k = cv2.waitKeyEx(500)
            if k == 2424832:
                x_tracker -= 1
            if k == 2555904:
                x_tracker += 1
            if k == 2490368:
                y_tracker += 1
            if k == 2621440:
                y_tracker -= 1
            if k == 113:
                rot_tracker -= 0.2
            if k == 101:
                rot_tracker += 0.2

            if k == 13:
                print('Aligning...')
                cv2.destroyAllWindows()
                for frame_index in range(stack_dst.shape[0]):

                    frame_warped = rotation(y_move(x_move(stack_dst[frame_index, :, :], -x_tracker*src_scale), -y_tracker*src_scale), -rot_tracker)
                    print([x_tracker, y_tracker])
                    warped_stack[frame_index, :, :] = frame_warped
                    print('{} %'.format(np.round(frame_index / stack_dst.shape[0] * 100, 1)))
                warped_stack = img_as_uint(warped_stack.astype('uint16'))
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
    meta = meta.sort_values(by=0)
    meta['segment'] = np.nan

    video_index_ls = np.array([int(video_index.split('.')[0]) for video_index in video_name_ls])
    assert np.array_equal(video_index_ls, meta.iloc[:, 0].values), 'Meta File and Video names does not match'

    video_len_ls =[]
    video_ls = []
    for video_name_ls_index in range(len(video_name_ls)):
        video_ls.append(read_tif(folder + video_name_ls[video_name_ls_index]))
        video_len_ls.append(video_ls[-1].shape[0])
    video_len_ls = np.array(video_len_ls)

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

    time_stamp_ls = np.insert(np.cumsum(video_len_ls), 0, 0)
    video_ls = []
    for time_stamp_index in range(len(time_stamp_ls) - 1):
        video_ls.append(aligned_video[time_stamp_ls[time_stamp_index]:time_stamp_ls[time_stamp_index + 1]])

    output_video = np.empty((0, video_ls[0].shape[1], video_ls[0].shape[2])).astype('uint16')
    video_ls = [video_ls[i] for i in np.argsort(meta.iloc[:, 1] .values)]
    for temp_video in video_ls:
        temp_video = np.reshape(temp_video, (-1, video_ls[0].shape[1], video_ls[0].shape[2]))
        output_video = np.append(output_video, np.array(temp_video), axis=0)

    # for final_order_index in meta.iloc[:, 1] .values - 1:
    #     temp_video = np.reshape(video_ls[final_order_index], (-1, video_ls[0].shape[1], video_ls[0].shape[2]))
    #     output_video = np.append(output_video, np.array(temp_video), axis=0)

    io.imsave(folder + excel_file_name.split('.')[0] + '.tif', output_video)
