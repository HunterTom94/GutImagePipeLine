from f_cleanup import f_cleanup
from os.path import isfile, join, isdir
from os import listdir, makedirs
from time import time
# from joblib import Parallel, delayed
import multiprocessing as mp


def execute_folder(video_folder):
    time_track = time()
    f_cleanup(root_folder, video_folder + '\\', f_from_roi_flag=0, line=0, interval=4, label_font=10, suggest_mc=1)  # 1 for execution
    print('{} finished. Time elapsed: {}'.format(video_folder ,time() - time_track))

# root_folder = 'D:\\Gut Imaging\\Videos\\Temp\\IndAA\\'
root_folder = 'D:\\Gut Imaging\\Videos\\Temp\\'
folders = [join(root_folder, f) for f in listdir(root_folder) if f[0] == 'd' and isdir(join(root_folder, f))]
# common_name = '150309b'
# folders = [join(root_folder, f) for f in listdir(root_folder) if common_name in f]

num_cores = mp.cpu_count()

if __name__ == '__main__':
    pool = mp.Pool(num_cores)
    pool.map(execute_folder, folders)

    pool.close()
    pool.join()

    # for video_folder in folders:
    #     files = [f for f in listdir(video_folder) if isfile(join(video_folder, f))]
    #     time_track = time()
    #     f_cleanup(root_folder, video_folder + '\\', f_from_roi_flag = 0, line=0, interval=4, label_font=10)  # 1 for execution
    #     print(time() - time_track)