import cv2
from video_util import read_tif
from os import listdir
import numpy as np
from Igor_related_util import read_igor_roi_matrix

folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\SARFIA_drawROI\\d24982504\\'
files = listdir(folder)

video_name = [ind for ind in files if ind.endswith('.tif')][0]
params = np.unique([ind.split('_')[1].rsplit('.',1)[0] for ind in files if ind.endswith('.csv')])
# print(params)
# params = ['f15t1.5' 'f20t1.5' 'f5t1.5' 'f8t0.5' 'f8t1.5' 'f8t3']
# F
params = ['f8t1.5', 'f5t1.5',  'f15t1.5']
# T
params = ['f8t1.5', 'f8t0.5', 'f8t3']
max_proj = read_tif(folder+video_name)
norm_max = np.empty_like(max_proj)
cv2.normalize(max_proj,dst=norm_max,  alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
norm_max = norm_max
BGR_max = cv2.cvtColor(norm_max, cv2.COLOR_GRAY2BGR)*5
BGR_max = np.clip(BGR_max, 0,255).astype('uint8')

# color_ls = ['2EAC66', '4C8E55','6B7144','8A5333','A93622','C81912']
# color_ls = ['2EAC66', '6B7144','8A5333','C81912']
# color_ls = ['2EAC66', '8A5333', 'C81912']
f_color_ls = ['00bbf0', 'd9faff',  '005792']
t_color_ls = ['fca180', 'fffe9f',  'd92027']
color_dict = dict(zip(params, t_color_ls))

cv2.namedWindow('FOV')
pt1 = (0,0)
pt2 = (0,0)

for param in params[::-1]:
    print(param)
    rois = read_igor_roi_matrix(folder+'roi_{}.csv'.format(param))
    bin_rois = np.zeros_like(rois).astype('uint8')
    bin_rois[rois != 0] = 1

    contours, hierarchy = cv2.findContours(bin_rois, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(BGR_max, contours, -1, tuple(int(color_dict[param][i:i+2], 16) for i in (0, 2, 4))[::-1], 1)

def crop_save(action, x, y, flags, userdata):
    global pt1, pt2
    if action == cv2.EVENT_LBUTTONDOWN:
        pt1 = (x, y)

    elif action == cv2.EVENT_LBUTTONUP:
        pt2 = (x, y)
        # cropped = BGR_max[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        cropped = BGR_max[pt2[1]-100:pt2[1], pt2[0]-100:pt2[0]]
        cropped = cv2.resize(cropped, (int(cropped.shape[1] * 5), int(cropped.shape[0] * 5)))
        cv2.imwrite('cropped.png', cropped)
        cv2.imshow("Crop", cropped)
        # key = cv2.waitKeyEx(1)
cv2.setMouseCallback('FOV', crop_save)
cv2.imshow('FOV',BGR_max)
    # norm_max[rois != 0] = 255
    # cv2.imshow('1', norm_max)
cv2.waitKey(-1)
