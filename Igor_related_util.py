import numpy as np
import pandas as pd


def read_igor_roi_matrix(file):
    #  Arg:    file -- absolute address of igor roi matrix as csv matrix, containing suffix
    #  Return: ROI_matrix -- np array, 2d matrix compatible with output of read_tif()
    ROI_matrix = np.loadtxt(open(file, "rb"),
                            delimiter=",", skiprows=1)
    ROI_matrix = np.delete(ROI_matrix, 0, axis=0)
    ROI_matrix = -ROI_matrix
    ROI_matrix[np.where(ROI_matrix == -1)] = 0
    ROI_matrix = np.transpose(ROI_matrix)
    return ROI_matrix

def read_igor_f_matrix(file):
    #  Arg:    file -- absolute address of igor f matrix as csv matrix, containing suffix.
    #  Return: dataframe, each column is one ROI
    return pd.read_csv(file, header=1)

