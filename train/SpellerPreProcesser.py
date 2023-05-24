import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset

import SpellerConstant
from SpellerConstant import SENSOR_AXES
from SpellerConstant import SUMMARY_STATISTICS_SUFFIX
from RealTimePeakDetector import RealTimePeakDetector

def extract_X_y(data_frame: pd.DataFrame()):
    data = data_frame['data'].tolist()
    X = pad_data(data)
    y = data_frame['target'].to_numpy()
    return X, y


def standardize(data: [], min, max):
    _data = TimeSeriesScalerMinMax((min, max)).fit_transform(data)
    dim = _data.shape
    _data = _data.reshape(dim[0], -1)
    return _data


def pad_data(data: []):
    data = to_time_series_dataset(data)
    dim = data.shape
    data = data.reshape(dim[0], -1)
    return data


def extract_X_y_summ_sta(data_frame: pd.DataFrame()):
    for axes in SENSOR_AXES:
        for suffix in SUMMARY_STATISTICS_SUFFIX:
            col_name = '{}_{}'.format(axes, suffix)
            # print(type(data_frame[col_name]))
            data_frame['data_summ_stat'] = data_frame['data_summ_stat'].combine(data_frame[col_name],
                                                                                (lambda x1, x2: x1 + [x2]))

    data = data_frame['data_summ_stat'].tolist()
    X = standardize(data)
    y = data_frame['target'].to_numpy()
    return X, y

def count_peak_dip(data_array:[]):
    peak = sum(1 for i in data_array if i > 0)
    dip = sum(1 for i in data_array if i < 0)
    return peak, dip
# TODO add peak and dip numbers
def create_summary_statistics(data_frame: pd.DataFrame()):
    lag = 18
    threshold = 3
    influence = 1
    for axes in SENSOR_AXES:
        column = data_frame[axes]
        mean_array = np.array([])
        std_array = np.array([])
        var_array = np.array([])
        peak_array = np.array([])
        dip_array = np.array([])
        data_summ_stat = []
        for column_elem in column:
            mean_array = np.append(mean_array, float(np.mean(column_elem)))
            std_array = np.append(std_array, float(np.std(column_elem)))
            var_array = np.append(var_array, float(np.var(column_elem)))
            data_summ_stat.append([])
            realTimePeakDetector = RealTimePeakDetector(column_elem[0:lag], lag, threshold,
                                                                             influence)
            for data_point in column_elem[lag:]:
                realTimePeakDetector.thresholding_algo(data_point)

            peak, dip = count_peak_dip(realTimePeakDetector.signals)
            peak_array = np.append(peak_array, peak)
            dip_array = np.append(dip_array, dip)
        data_frame['{}_peak'.format(axes)] = peak_array
        data_frame['{}_dip'.format(axes)] = dip_array
        data_frame['{}_mean'.format(axes)] = mean_array
        data_frame['{}_std'.format(axes)] = std_array
        data_frame['{}_var'.format(axes)] = var_array
        data_frame['data_summ_stat'] = data_summ_stat
    return data_frame


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
