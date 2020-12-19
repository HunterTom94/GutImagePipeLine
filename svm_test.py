import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sys import exit
from matplotlib.patches import Rectangle
from util import low_pass_filter
import itertools
from sklearn import svm
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump, load
#
#
prepare_data = 0

if prepare_data:
    input_folder = 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\Map_ShortDura_NoFilter\\'
    # input_folder = 'D:\\#Yinan\\untitled folder\\'
    # input_folder = 'D:\\#Yinan\\Dh31_AA\\'
    scheme = pd.read_excel(input_folder + '\\DeliverySchemes.xlsx')

    stimulus_ls = scheme['stimulation'].to_list()
    frame_ls = scheme['video_end'].to_numpy() - scheme['video_start'].to_numpy() + 1
    stimulus_length_ls = scheme['stimulus_end'].to_numpy() - scheme['stimulus_start'].to_numpy() + 1
    time_stamp_ls = np.insert(np.cumsum(frame_ls), 0, 0)
    organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
    raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')

    organized = organized[~organized["ROI_index"].str.contains('_b')]

    raw_organized = raw_organized[raw_organized['ROI_index'].isin(organized['ROI_index'])]
    raw_organized = raw_organized.sort_values(by=['ROI_index'], ascending=True)
    organized = organized.sort_values(by=['ROI_index'], ascending=True)
    raw_organized.index = range(raw_organized.shape[0])
    organized.index = range(organized.shape[0])
    assert raw_organized.shape[0] == organized.shape[0]

    sti_df = pd.DataFrame()
    for sample in raw_organized['sample_index'].unique():
        raw_copy = raw_organized.copy()
        org_copy = organized.copy()
        raw_copy = raw_copy[raw_copy['sample_index'] == sample]
        org_copy = org_copy[org_copy['sample_index'] == sample]

        temp_sti_df = pd.DataFrame()
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['df'] = raw_copy['{}_average_peak'.format('KCl')] - raw_copy['{}_f0_mean'.format('KCl')]
        temp_sti_df['basal'] = raw_copy['{}_f0_mean'.format('KCl')]
        temp_sti_df['average_response'] = org_copy['{}_average_response'.format('KCl')]
        temp_sti_df['average_basal'] = temp_sti_df['basal'].mean()
        temp_sti_df['median_basal'] = temp_sti_df['basal'].median()
        temp_sti_df['std_basal'] = temp_sti_df['basal'].std()
        temp_sti_df['n_mean_basal'] = temp_sti_df['basal']/temp_sti_df['basal'].mean()
        temp_sti_df['n_median_basal'] = temp_sti_df['basal']/temp_sti_df['basal'].median()
        temp_sti_df['std_n_mean_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].mean())/temp_sti_df['basal'].std()
        temp_sti_df['std_n_median_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].median())/temp_sti_df['basal'].std()

        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

    # X = np.hstack((sti_df['basal'].to_numpy().reshape((-1, 1)), sti_df['average_basal'].to_numpy().reshape((-1, 1)),
    #                sti_df['median_basal'].to_numpy().reshape((-1, 1)), sti_df['std_basal'].to_numpy().reshape((-1, 1)),
    #                sti_df['n_mean_basal'].to_numpy().reshape((-1, 1)), sti_df['n_median_basal'].to_numpy().reshape((-1, 1)),
    #                sti_df['std_n_mean_basal'].to_numpy().reshape((-1, 1)), sti_df['std_n_median_basal'].to_numpy().reshape((-1, 1))))

    X = np.hstack((sti_df['n_mean_basal'].to_numpy().reshape((-1, 1)),
                   sti_df['std_n_mean_basal'].to_numpy().reshape((-1, 1)),
                   sti_df['std_n_median_basal'].to_numpy().reshape((-1, 1))))

    # X = np.hstack((sti_df['n_mean_basal'].to_numpy().reshape((-1, 1)),
    #                sti_df['n_median_basal'].to_numpy().reshape((-1, 1)),
    #                sti_df['std_n_mean_basal'].to_numpy().reshape((-1, 1)),
    #                sti_df['std_n_median_basal'].to_numpy().reshape((-1, 1))))

    y = sti_df['average_response'].to_numpy().flatten()

    np.save('D:\\#Yinan\\svm_test\\AA_x.npy', X)
    np.save('D:\\#Yinan\\svm_test\\AA_y.npy', y)
    exit()

AA_x = np.load('D:\\#Yinan\\svm_test\\AA_x.npy')
AA_y = np.load('D:\\#Yinan\\svm_test\\AA_y.npy')

dosage_x = np.load('D:\\#Yinan\\svm_test\\dosage_x.npy')
dosage_y = np.load('D:\\#Yinan\\svm_test\\dosage_y.npy')

NEAA_x = np.load('D:\\#Yinan\\svm_test\\NEAA_x.npy')
NEAA_y = np.load('D:\\#Yinan\\svm_test\\NEAA_y.npy')

pros1_x = np.load('D:\\#Yinan\\svm_test\\pros1_x.npy')
pros1_y = np.load('D:\\#Yinan\\svm_test\\pros1_y.npy')

pros2_x = np.load('D:\\#Yinan\\svm_test\\pros2_x.npy')
pros2_y = np.load('D:\\#Yinan\\svm_test\\pros2_y.npy')

pros3_x = np.load('D:\\#Yinan\\svm_test\\pros3_x.npy')
pros3_y = np.load('D:\\#Yinan\\svm_test\\pros3_y.npy')

pros4_x = np.load('D:\\#Yinan\\svm_test\\pros4_x.npy')
pros4_y = np.load('D:\\#Yinan\\svm_test\\pros4_y.npy')

sti_pros1_x = np.load('D:\\#Yinan\\svm_test\\sti_pros1_x.npy')
sti_pros1_y = np.load('D:\\#Yinan\\svm_test\\sti_pros1_y.npy')

sti_pros2_x = np.load('D:\\#Yinan\\svm_test\\sti_pros2_x.npy')
sti_pros2_y = np.load('D:\\#Yinan\\svm_test\\sti_pros2_y.npy')

sti_pros3_x = np.load('D:\\#Yinan\\svm_test\\sti_pros3_x.npy')
sti_pros3_y = np.load('D:\\#Yinan\\svm_test\\sti_pros3_y.npy')

def train(x_ls, y_ls):
    X = np.vstack(tuple(x_ls))
    y = np.vstack(tuple(y_ls))

    print('Train Size: {}'.format(X.shape[0]))
    print()

    clf = svm.SVC(gamma='scale', class_weight='balanced', kernel='linear')
    clf.fit(X, y)
    return clf

def test(x_ls, y_ls, clf):
    X = np.vstack(tuple(x_ls))
    y = np.vstack(tuple(y_ls))

    print('Test Size: {}'.format(X.shape[0]))
    print()

    y_pred = clf.predict(X)
    print('precision: {}'.format(precision_score(y, y_pred)))
    print('recall: {}'.format(recall_score(y, y_pred)))
    print('F1: {}'.format(f1_score(y, y_pred)))
    print('confusion matrix: {}'.format(confusion_matrix(y, y_pred)))
    print()
    return y_pred

def train_test(x_ls, y_ls):
    X = np.vstack(x_ls)
    y = np.vstack(y_ls)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

    print('Train Size: {}'.format(X_train.shape[0]))
    print('Test Size: {}'.format(X_test.shape[0]))
    print()

    clf = svm.SVC(gamma='scale', class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('precision: {}'.format(precision_score(y_test,y_pred)))
    print('recall: {}'.format(recall_score(y_test,y_pred)))
    print('F1: {}'.format(f1_score(y_test,y_pred)))
    print('confusion matrix: {}'.format(confusion_matrix(y_test, y_pred)))
    print()

train_x = []
# train_x.append(dosage_x)
# train_x.append(NEAA_x)
# train_x.append(AA_x)
train_x.append(pros1_x)
train_x.append(pros2_x)
train_x.append(pros3_x)
train_x.append(pros4_x)
# train_x.append(sti_pros1_x)
# train_x.append(sti_pros2_x)
# train_x.append(sti_pros3_x)

train_y = []
# train_y.append(dosage_y.reshape(-1,1))
# train_y.append(NEAA_y.reshape(-1,1))
# train_y.append(AA_y.reshape(-1,1))
train_y.append(pros1_y.reshape(-1,1))
train_y.append(pros2_y.reshape(-1,1))
train_y.append(pros3_y.reshape(-1,1))
train_y.append(pros4_y.reshape(-1,1))
# train_y.append(sti_pros1_y.reshape(-1,1))
# train_y.append(sti_pros2_y.reshape(-1,1))
# train_y.append(sti_pros3_y.reshape(-1,1))

test_x = []
# test_x.append(dosage_x)
# test_x.append(NEAA_x)
test_x.append(AA_x)
# test_x.append(pros1_x)
# test_x.append(pros2_x)
# test_x.append(pros3_x)
# test_x.append(pros4_x)
# test_x.append(sti_pros1_x)
# test_x.append(sti_pros2_x)
# test_x.append(sti_pros3_x)
#
test_y = []
# test_y.append(dosage_y.reshape(-1,1))
# test_y.append(NEAA_y.reshape(-1,1))
test_y.append(AA_y.reshape(-1,1))
# test_y.append(pros1_y.reshape(-1,1))
# test_y.append(pros2_y.reshape(-1,1))
# test_y.append(pros3_y.reshape(-1,1))
# test_y.append(pros4_y.reshape(-1,1))
# test_y.append(sti_pros1_y.reshape(-1,1))
# test_y.append(sti_pros2_y.reshape(-1,1))
# test_y.append(sti_pros3_y.reshape(-1,1))

# clf = train(train_x, train_y)
# y_pred = test(test_x, test_y, clf)

train_test(train_x, train_y)
# print(clf.coef_)
# print(clf.intercept_)

# X = np.vstack(tuple(test_x))
# y = np.vstack(tuple(test_y))
#
# formula_pred = np.array(np.matmul(X, clf.coef_.reshape((3,1))) + clf.intercept_ > 0).astype(int).flatten()