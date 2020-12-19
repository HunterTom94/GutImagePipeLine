import numpy as np
from video_util import read_tif
from os import listdir
from os.path import isfile, join, isdir
from scipy.signal import correlate2d
from Igor_related_util import read_igor_roi_matrix
from matplotlib import pyplot as plt


def offset_calc(corr_result, shape):
    if shape % 2 == 1:
        offset = (shape / 2 - 0.5) - corr_result
    if shape % 2 == 0:
        offset = (shape / 2 - 1) - corr_result
    return np.abs(offset)

root_folder = 'D:\\Gut Imaging\\Videos\\MC_pre\\'

video_folders = [folder for folder in listdir(root_folder) if isdir(join(root_folder, folder))]

sample_indices = []
for folder in video_folders:
    sample_indices.append(folder[1:7])

for sample in np.unique(sample_indices):
    sample_folders = [folder for folder in video_folders if sample in folder]
    for folder in sample_folders:
        files = listdir(root_folder+ folder)
        video_ls = [file for file in files if file.endswith('.tif')]
        assert len(video_ls) == 1
        raw_stack = read_tif(root_folder+ folder+'\\'+video_ls[0])
        ROI_matrix = read_igor_roi_matrix(root_folder + folder + '\\' + [file for file in files if file == 'roi.csv'][0])

        offset_mx = np.empty((len(np.unique(ROI_matrix)) - 1, raw_stack.shape[0]))

        offset_sum_ls = []
        # column named after index start from 0
        for column in np.unique(ROI_matrix)[1:]:
            column = int(column)
            roi_coord = np.where(ROI_matrix == column)
            y_max = np.max(roi_coord[0])
            y_min = np.min(roi_coord[0])
            x_max = np.max(roi_coord[1])
            x_min = np.min(roi_coord[1])
            if (y_max - y_min) * (x_max - x_min) > 1000:
                offset_mx[column-1, :] = np.nan
                print('ROI too big, discard')
            else:
                print('{}/{}'.format(column, len(np.unique(ROI_matrix))))
                mean_frame = np.mean(raw_stack, axis=0)
                image_2d_mean = mean_frame[y_min:y_max + 1, x_min:x_max + 1]
                for frame in range(raw_stack.shape[0]):#[:-1]:
                    curr_frame = raw_stack[frame, :, :]
                    # next_frame = raw_stack[frame + 1, :, :]
                    image_2d_curr = curr_frame[y_min:y_max + 1, x_min:x_max + 1]
                    # image_2d_next = next_frame[y_min:y_max + 1, x_min:x_max + 1]
                    # corr_2d_result = correlate2d(image_2d_curr, image_2d_next, mode='same')
                    corr_2d_result = correlate2d(image_2d_curr, image_2d_mean, mode='same')
                    max_ind = np.unravel_index(corr_2d_result.argmax(), corr_2d_result.shape)
                    y_offset = offset_calc(max_ind[0], corr_2d_result.shape[0])
                    x_offset = offset_calc(max_ind[1], corr_2d_result.shape[1])
                    total_offset = np.linalg.norm([y_offset, x_offset])
                    offset_mx[column-1, frame] = total_offset
        np.save('offset_mx_{}.npy'.format(sample), offset_mx)
        mean_offset = np.nanmean(offset_mx, axis=0)
        plt.plot(mean_offset)
    plt.title(sample)
    plt.savefig(root_folder + 'mean_cell_offset_{}.png'.format(sample))
    plt.show()
