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
from time import time
from sys import exit
from os import listdir
from os.path import isfile, join

folder = 'D:\\Gut Imaging\\Videos\\Temp\\d214246\\'


def x_move(src, offset):
    pad = np.zeros(src.shape)
    if offset>0:
        pad[:,offset:] = src[:,:-offset]
    elif offset < 0:
        pad[:, :offset] = src[:, -offset:]
    else:
        pad = src
    return pad

def y_move(src, offset):
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

    stack_src_max = np.clip(stack_src, 0, 80).transpose()
    stack_dst_max = np.clip(stack_dst, 0, 80).transpose()

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
    src_dummy = draw_pad[:,:,1].copy()

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
            return [cv2.resize(moved_src, (stack_dst.shape[1], stack_dst.shape[0])).astype('uint8'), draw_pad.astype('uint8')]

immuno_image_ls = []
max = np.max(read_tif(folder + folder.split('\\')[-2] + '.tif'), axis=0)
immuno_image_ls.append(max)
folder += 'Immuno\\'
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
immuno_names = [f for f in onlyfiles if f.split('.')[-1] == 'tif' and 'Immuno' in f]
for immuno_name in immuno_names:
    immuno_image = read_tif(folder + immuno_name)[0:max.shape[0], 0:max.shape[1]]
    immuno_name = immuno_name.split('_')[-1]
    immuno_image_ls.append(immuno_image)

    assert len(np.unique(np.array([video.shape[0] for video in immuno_image_ls]))) == 1, 'Video dimensions do not agree'
    assert len(np.unique(np.array([video.shape[1] for video in immuno_image_ls]))) == 1, 'Video dimensions do not agree'

    aligned_immuno, merged = align_video(immuno_image, max)
    io.imsave(folder + 'Aligned_immuno_{}'.format(immuno_name), aligned_immuno)
    io.imsave(folder + 'Merged_immuno_{}'.format(immuno_name), merged)