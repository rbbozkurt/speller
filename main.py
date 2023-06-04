# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw
import string
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, cross_val_score
from database import SpellerDatabase
from database.SpellerDatabase import *
import plot.SpellerPlotter as Plotter
import SpellerConstant
from train import SpellerModel
from train import SpellerPreProcesser
from train import SpellerTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

TARGET_LETTERS = list(string.ascii_uppercase)
TOP_K_VALUES = [1, 2, 3, 4, 5]
N_SPLITS = 5


def create_label_dict(letter_list: []):
    labels = {}
    for num, letter in enumerate(letter_list, start=1):
        labels[letter] = float(num)
    return labels


def plot_graphs_from_data_frame(data_frame, label_dic):
    Plotter.plot_letter_sensors(TARGET_LETTERS, data_frame, label_dic, "Before standardized")
    Plotter.plot_letter_sensor(data_frame, label_dic)


def get_summ_stats_X_y_from_data_frame(data_frame):
    # create summary statistics from sensor datas
    data_frame_summ = SpellerPreProcesser.create_summary_statistics(data_frame)
    # extract X and y values from summ stat data
    X, y = SpellerPreProcesser.extract_X_y_summ_sta(data_frame_summ)
    return X, y


def train_session(model, X, y, session_desc):
    # fold scores and descriptions
    fold_scores = []
    fold_descs = []

    # validation score and descriptions
    valid_score = None
    valid_desc = None

    # min and max values for scaler
    min = -1
    max = 1

    # pad and scale data
    print("------------ Standardizing data with ({},{}) ({}) ------------".format(min, max, session_desc))
    _X = standardize_data(X, max, min)

    # split the data and store X_valid and y_valid for leave out validation
    print("------------ Splitting test,train set ({}) ------------".format(session_desc))
    _X, X_valid, y, y_valid = train_test_split(_X, y, test_size=0.2, random_state=42, stratify=y)

    # get train and test indeces
    print("------------ Training SciKit Model ({}) ------------".format(session_desc))
    model = fold_test_train(_X, fold_descs, fold_scores, model, y)

    # test model on validation set
    print("------------ Testing SciKit Model ({}) ------------".format(session_desc))
    valid_desc, valid_score, y_pred = model_predict(model, X_valid, y_valid)
    valid_desc = '{}-{}'.format(valid_desc, session_desc)

    if 'HGBR' not in session_desc:
        calculate_top_k_accuracy_scores(_X, model, session_desc, y)

    model_desc = "Test-{}".format(session_desc)
    title = 'Confusion {}'.format(model_desc)
    plot_confusion_matrix(title, y_pred, y_valid)

    y_pred = model.predict(X_valid)
    y_pred = np.around(y_pred)

    model_desc = "Train-{}".format(session_desc)
    title = 'Confusion {}'.format(model_desc)
    plot_confusion_matrix(title, y_pred, y_valid)

    # Plot performance of two models
    print("------------ Plotting results ------------")
    Plotter.plot_bar(fold_scores, fold_descs, "Balance Accuracy Scores of Folds {}".format(model_desc))

    return valid_desc, valid_score, model


def plot_confusion_matrix(title, y_pred, y_valid):
    conf_matrix_scikit = confusion_matrix(y_valid, y_pred)
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', title,
                             TARGET_LETTERS)


def model_predict(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    y_pred = np.around(y_pred)
    valid_score = balanced_accuracy_score(y_valid, y_pred)
    valid_desc = 'Validation'
    return valid_desc, valid_score, y_pred


def fold_test_train(_X, fold_descs, fold_scores, model, y):
    skf = StratifiedKFold(n_splits=N_SPLITS)
    for i, (train_index, test_index) in enumerate(skf.split(_X, y)):
        X_train = _X[train_index]
        y_train = y[train_index]
        X_test = _X[test_index]
        y_test = y[test_index]
        # fit model
        model = model.fit(X_train, y_train)
        # get predictions
        y_pred = model.predict(X_test)
        y_pred = np.around(y_pred)
        fold_descs.append('Fold {}'.format(i))
        fold_scores.append(balanced_accuracy_score(y_test, y_pred))
    mean_of_folds = np.mean(fold_scores)
    fold_scores.append(mean_of_folds)
    fold_descs.append('Mean')
    return model


def standardize_data(X, max, min):
    _X = np.nan_to_num(X, copy=True)
    _X = SpellerPreProcesser.standardize(_X, min, max)
    return _X


def calculate_top_k_accuracy_scores(_X, model, session_desc, y):
    # calculate top-k accuracy scores
    print("------------ Calculating Top-K-Accuray SciKit Model ({}) ------------".format(session_desc))
    k_values = []
    score = []
    for k in TOP_K_VALUES:
        k_values.append(k)
        score.append(SpellerTrainer.calculate_top_k_accuracy(model, _X, y, k))
    print("------------ Plotting results of Top-K-Accuracy ------------")
    Plotter.plot_bar(score, k_values, "Top K Accuracy Scores Sk {}".format(session_desc))


def init_and_train_models(X, y, desc):
    model_descs = []
    model_scos = []
    lr_model = LogisticRegression(random_state=0, multi_class='multinomial')
    knn_model = KNeighborsClassifier(n_neighbors=len(TARGET_LETTERS))
    hgbr_model = HistGradientBoostingRegressor()
    # start train sessions
    knn_desc, knn_sc, _ = train_session(knn_model, X, y, "{}_KNN".format(desc))
    lr_desc, lr_sc, _ = train_session(lr_model, X, y, "{}_LR".format(desc))
    hgbr_desc, hgbr_sc, _ = train_session(hgbr_model, X, y, "{}_HGBR".format(desc))
    model_descs.extend([knn_desc, lr_desc, hgbr_desc])
    model_scos.extend([knn_sc, lr_sc, hgbr_sc])

    return model_descs, model_scos


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    label_dict = create_label_dict(TARGET_LETTERS)
    data_frame = SpellerDatabase.read_letters_from_database(SpellerConstant.FIREBASE_REFERENCE, tuple(label_dict))

    # plot_graphs_from_data_frame(data_frame,label_dict)
    X, y = SpellerPreProcesser.extract_X_y(data_frame)

    # init and train models with X, y
    sd_descs, sd_scos = init_and_train_models(X, y, "SD")

    X_summ_stat, y_summ_stat = get_summ_stats_X_y_from_data_frame(data_frame)
    # X, y = read_letters_from_npy('X_timeseries.npy', 'y_timeseries.npy')

    # init and train model with summary stats
    ss_descs, ss_scos = init_and_train_models(X_summ_stat, y_summ_stat, "SS")

    score = sd_scos + ss_scos
    desc = sd_descs + ss_descs

    print("------------ Plotting results ------------")
    Plotter.plot_bar(sd_scos, sd_descs,
                     "Balance Accuracy Scores Sk SD")
    Plotter.plot_bar(ss_descs, ss_scos,
                     "Balance Accuracy Scores Sk SS")
    Plotter.plot_bar(score, desc,
                     "Balance Accuracy Scores Comparison")

# balance acc score list for different datasets

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
