import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset

import SpellerConstant
from SpellerConstant import SENSOR_AXES
from SpellerConstant import SUMMARY_STATISTICS_SUFFIX


def extract_X_y(data_frame: pd.DataFrame()):
    data = data_frame['data'].tolist()
    X = standardize(data)
    y = data_frame['target'].to_numpy()
    return X, y


def standardize(data: []):
    data = to_time_series_dataset(data)
    data = TimeSeriesScalerMinMax((-1.,1.)).fit_transform(data)
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


def create_summary_statistics(data_frame: pd.DataFrame()):
    for axes in SENSOR_AXES:
        column = data_frame[axes]
        mean_array = np.array([])
        std_array = np.array([])
        var_array = np.array([])
        data_summ_stat = []
        for column_elem in column:
            mean_array = np.append(mean_array, float(np.mean(column_elem)))
            std_array = np.append(std_array, float(np.std(column_elem)))
            var_array = np.append(var_array, float(np.var(column_elem)))
            data_summ_stat.append([])
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