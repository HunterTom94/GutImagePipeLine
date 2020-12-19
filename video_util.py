import numpy as np
from skimage import io
import cv2


def read_tif(source_folder, file_name=[]):
    #  Read tif video file and return numpy array and video filename
    #
    #  Arg:    source_folder -- string, absolute address of folder; if only one argument is provided, this argument indicates full file path
    #          file_name -- string, file name containing suffix
    #  Return: raw_stack -- np array, [frame_number, row_number, column_number]
    #          video_name -- string, file name without suffix
    if file_name:
        stack_filepath = source_folder + file_name
        video_name = file_name.split('.')[0]
    else:
        stack_filepath = source_folder
    print(stack_filepath)
    raw_stack = io.imread(stack_filepath)
    print(raw_stack.shape)
    if file_name:
        return raw_stack, video_name
    else:
        return raw_stack

def get_peak_frame(source):
    #  Return the frame and the index of that frame in an video that has the overall summed pixel value
    #
    #  Arg:    source -- 3darray, input video
    #  Return: peak   -- 2darray, peak frame of the video
    #          ind    -- int, frame index of the peak frame
    frame_sum = np.sum(source, axis=(1, 2))
    ind = np.argmax(frame_sum)
    peak = source[ind]
    return peak, ind


def deltaF_video(source, f0_frames):
    #  Return a video that is subtracted from average of a series of frames (f0)
    #
    #  Arg:    source     -- string, absolute address of input video file
    #                     -- or 3darray, input video
    #          f0_frames  -- 1darray, array of frames to be averaged as f0
    #  Return: df         -- 3darray, output deltaF video
    if isinstance(source, str):
        stack, _ = read_tif(source)
    elif isinstance(source, np.ndarray):
        stack = source

    average_frame = np.mean(stack[f0_frames], axis=0)
    df = stack - average_frame
    return df


def invert(source, bit):
    #  Invert the pixel values of a video based on its bit depth
    #
    #  Arg:    source     -- ndarray, input image/video
    #          bit        -- int, bit depth of the source to be inverted
    #  Return:            -- inverted image/video
    return -(source - np.power(2, bit))


def allgin_zoom(source, stimuli_frame, rectangle, rectangle_specs=[], circle_specs=[]):
    #  Concatenate a video and a zoomed region in the video side by side. Zoomed region is indicated by a rectangle.
    #  Frames where the stimuli are presented will be indicated with a circle at the lower left of the output video.
    #  Input video is expected to be color-coded in RGB.
    #
    #  Arg:    source               -- string, absolute address of folder; if only one argument is provided, this argument indicates full file path
    #                               -- or 4darray, input video with RGB channels (frame_num, height, width, RGB)
    #          stimuli_frame        -- 1darray, array of frames when stimilus is on
    #          rectangle            -- tuple, (top, left, width) of zoom rectangle
    #          rectangle_specs      -- tuple, ((R,G,B), lineThickness) of the rectangle specifying zoom in area
    #          circle_specs         -- tuple, ((center_x, center_y), radius, (R,G,B)) of the circle indicating on-stimulus
    #  Return: canvas               -- 4darray, video of raw input and concatenated zoomed rectangle on the right side
    #                                  (frame_num, height, width, RGB)
    if isinstance(source, str):
        stack, _ = read_tif(source)
    elif isinstance(source, np.ndarray):
        stack = source

    if not circle_specs:
        circle_center = (39, 722)
        circle_radius = 20
        circle_color = (255, 128, 0)

    if not rectangle_specs:
        rectangle_color = (255, 255, 255)
        rectangleThickness = 2

    top = rectangle[0]
    left = rectangle[1]
    width = rectangle[2]
    height = np.round(stack.shape[1] * width / stack.shape[2]).astype(int)

    canvas = np.ones((stack.shape[0], stack.shape[1], stack.shape[2] * 2 + 5, stack.shape[3]))
    canvas[:, 0:stack.shape[1], 0:stack.shape[2], :] = stack
    for frame_index in range(stack.shape[0]):
        for color_axis in range(3):
            zoom = cv2.resize(stack[frame_index, top:top + height, left:left + width, color_axis],
                              (stack.shape[2], stack.shape[1]))
            canvas[frame_index, 0:stack.shape[1], canvas.shape[2] - stack.shape[2]:canvas.shape[2], color_axis] = zoom
        cv2.rectangle(canvas[frame_index, :, :, :], (left, top), (left + width, top + height), rectangle_color,
                      rectangleThickness)

    for ind in stimuli_frame:
        cv2.circle(canvas[ind, :, :, :], circle_center, circle_radius, circle_color, cv2.FILLED)

    return canvas.astype('uint8')
