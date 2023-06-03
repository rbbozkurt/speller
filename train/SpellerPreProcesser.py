import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
import antropy as ant
import SpellerConstant
from SpellerConstant import SENSOR_AXES
from SpellerConstant import SUMMARY_STATISTICS_FEATURES, PEAK_DETECTOR_LAG
from train.RealTimePeakDetector import RealTimePeakDetector
import scipy.stats as stats
from scipy.fft import fft, fftfreq


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
        for feature in SUMMARY_STATISTICS_FEATURES:
            col_name = '{}_{}'.format(axes, feature)
            # print(type(data_frame[col_name]))
            data_frame['data_summ_stat'] = data_frame['data_summ_stat'].combine(data_frame[col_name],
                                                                                (lambda x1, x2: x1 + [x2]))

    data = data_frame['data_summ_stat'].tolist()
    X = standardize(data, -1, 1)
    y = data_frame['target'].to_numpy()
    return X, y


def count_peak_dip(data_array: []):
    peak = sum(1 for i in data_array if i > 0)
    dip = sum(1 for i in data_array if i < 0)
    return peak, dip


def create_peak_dip_arrays(lag, threshold, signal_data):
    realTimePeakDetector0 = RealTimePeakDetector(signal_data[0:lag], PEAK_DETECTOR_LAG, threshold,
                                                 0)
    realTimePeakDetector1 = RealTimePeakDetector(signal_data[0:lag], PEAK_DETECTOR_LAG, threshold,
                                                 1)

    for data_point in signal_data[lag:]:
        realTimePeakDetector0.thresholding_algo(data_point)
        realTimePeakDetector1.thresholding_algo(data_point)

    return realTimePeakDetector0.signals, realTimePeakDetector1.signals


# TODO add peak and dip numbers
def create_summary_statistics(data_frame: pd.DataFrame()):
    for axes in SENSOR_AXES:
        column = data_frame[axes]
        Min = []
        Max = []
        hj_mob_array = []
        hj_com_array = []
        peak_array0 = []
        dip_array0 = []
        peak_array1 = []
        dip_array1 = []
        Mean = []
        Rms = []
        Var = []
        Std = []
        Power = []
        Peak = []
        Skew = []
        Kurtosis = []
        P2p = []
        CrestFactor = []
        FormFactor = []
        PulseIndicator = []
        Max_f = []
        Sum_f = []
        Mean_f = []
        Var_f = []
        Peak_f = []
        Skew_f = []
        Kurtosis_f = []
        data_summ_stat = []
        for X in column:
            ## TIME DOMAIN ##
            Min.append(np.min(X))
            Max.append(np.max(X))
            Mean.append(np.mean(X))
            Rms.append(np.sqrt(np.mean(X ** 2)))
            Var.append(np.var(X))  # Hjorth's activity
            Std.append(np.std(X))
            Power.append(np.mean(X ** 2))
            Peak.append(np.max(np.abs(X)))
            P2p.append(np.ptp(X))
            CrestFactor.append(np.max(np.abs(X)) / np.sqrt(np.mean(X ** 2)))
            Skew.append(stats.skew(X))
            Kurtosis.append(stats.kurtosis(X))
            FormFactor.append(np.sqrt(np.mean(X ** 2)) / np.mean(X))
            PulseIndicator.append(np.max(np.abs(X)) / np.mean(X))
            ## FREQ DOMAIN ##
            ft = fft(X)
            S = np.abs(ft ** 2) / len(X)
            Max_f.append(np.max(S))
            Sum_f.append(np.sum(S))
            Mean_f.append(np.mean(S))
            Var_f.append(np.var(S))
            Peak_f.append(np.max(np.abs(S)))
            Skew_f.append(stats.skew(X))
            Kurtosis_f.append(stats.kurtosis(X))

            ## calculate Hjorth Parameters mobility & complexity ##
            mob, com = ant.hjorth_params(X)
            hj_mob_array.append(mob)
            hj_com_array.append(com)


            ## peak and dips ##
            threshold = (np.max(X) - np.min(X)) / 2
            signal0, signal1 = create_peak_dip_arrays(PEAK_DETECTOR_LAG, threshold, X)
            peak, dip = count_peak_dip(signal0)
            peak_array0.append(peak)
            dip_array0.append(dip)
            peak, dip = count_peak_dip(signal1)
            peak_array1.append(peak)
            dip_array1.append(dip)
            data_summ_stat.append([])

        data_frame['{}_MIN'.format(axes)] = Min
        data_frame['{}_MAX'.format(axes)] = Min
        data_frame['{}_PEAK_INF0'.format(axes)] = peak_array0
        data_frame['{}_DIP_INF0'.format(axes)] = dip_array0
        data_frame['{}_PEAK_INF1'.format(axes)] = peak_array1
        data_frame['{}_DIP_INF1'.format(axes)] = dip_array1
        data_frame['{}_HJ_MOB'.format(axes)] = hj_mob_array
        data_frame['{}_HJ_COM'.format(axes)] = hj_com_array
        data_frame['{}_MEAN'.format(axes)] = Mean
        data_frame['{}_RMS'.format(axes)] = Rms
        data_frame['{}_VAR'.format(axes)] = Var
        data_frame['{}_STD'.format(axes)] = Std
        data_frame['{}_POWER'.format(axes)] = Power
        data_frame['{}_PEAK'.format(axes)] = Peak
        data_frame['{}_P2P'.format(axes)] = P2p
        data_frame['{}_CREST_FACTOR'.format(axes)] = CrestFactor
        data_frame['{}_SKEW'.format(axes)] = Skew
        data_frame['{}_KURTOSIS'.format(axes)] = Kurtosis
        data_frame['{}_MAX_f'.format(axes)] = Max_f
        data_frame['{}_SUM_f'.format(axes)] = Sum_f
        data_frame['{}_MEAN_f'.format(axes)] = Mean_f
        data_frame['{}_VAR_f'.format(axes)] = Var_f
        data_frame['{}_PEAK_f'.format(axes)] = Peak_f
        data_frame['{}_SKEW_f'.format(axes)] = Skew_f
        data_frame['{}_KURTOSIS_f'.format(axes)] = Kurtosis

        data_frame['data_summ_stat'] = data_summ_stat
    return data_frame


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
