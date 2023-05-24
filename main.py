# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tslearn.metrics import dtw
import string

from database import SpellerDatabase
from database.SpellerDatabase import *
import plot.SpellerPlotter as Plotter
import SpellerConstant
from train import SpellerModel
from train import SpellerPreProcesser
from train import SpellerTrainer
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

TARGET_LETTERS = list(string.ascii_uppercase)


def create_label_dict(letter_list: []):
    labels = {}
    for num, letter in enumerate(letter_list, start=1):
        labels[letter] = float(num)
    return labels


def temp():
    balance_acc_score = []

    # names of the datasets
    models = ['Summary statistics with SciKit', 'Sensor data with SciKit']

    # summary statistics with SciKit

    # create summary statistics from sensor datas
    data_frame_summ = SpellerPreProcesser.create_summary_statistics(data_frame)
    # extract X and y values from summ stat data
    X, y = SpellerPreProcesser.extract_X_y_summ_sta(data_frame_summ)
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # create kn model
    sklearn_knn = KNeighborsClassifier(n_neighbors=len(TARGET_LETTERS))
    # train model
    trained_model = SpellerTrainer.train_model(X_train, y_train, sklearn_knn)
    # test model
    y_pred = trained_model.predict(X_test)
    # create conf matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('--------------- Summary Statistics ---------------')
    balance_acc_score.append(balanced_accuracy_score(y_test, y_pred))
    print(SpellerPreProcesser.multiclass_roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=TARGET_LETTERS))
    Plotter.plot_conf_matrix(conf_matrix, 'Prediction', 'Actual', 'Confusion Matrix Summary Statistics', TARGET_LETTERS)
    print("Shape: {}".format(X_train.shape[1]))
    initial_type = [
        ('input', FloatTensorType([None, X_train.shape[1]]))
    ]
    SpellerTrainer.convert_to_onnx(trained_model, "sklearn_summ_model", initial_type)

    # sensor data with SciKit
    X, y = SpellerPreProcesser.extract_X_y(data_frame)
    X = np.nan_to_num(X, copy=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sklearn_knn = KNeighborsClassifier(n_neighbors=len(TARGET_LETTERS))
    trained_model = SpellerTrainer.train_model(X_train, y_train, sklearn_knn)
    y_pred = trained_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print('--------------- Sensor Statistics ---------------')
    balance_acc_score.append(balanced_accuracy_score(y_test, y_pred))
    print(SpellerPreProcesser.multiclass_roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=TARGET_LETTERS))
    Plotter.plot_conf_matrix(conf_matrix, 'Prediction', 'Actual', 'Confusion Matrix with SciKit', TARGET_LETTERS)

    print("Shape: {}".format(X_train.shape[1]))
    initial_type = [
        ('input', FloatTensorType([None, X_train.shape[1]]))
    ]
    SpellerTrainer.convert_to_onnx(trained_model, "sklearn_model", initial_type)

    # sensor data with TsLearn
    tslearn_knn = KNeighborsTimeSeriesClassifier(n_neighbors=len(TARGET_LETTERS), metric="dtw")
    tslearn_knn.fit(X_train, y_train)
    y_pred = tslearn_knn.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    balance_acc_score.append(balanced_accuracy_score(y_test, y_pred))
    print(SpellerPreProcesser.multiclass_roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=TARGET_LETTERS))
    Plotter.plot_conf_matrix(conf_matrix, 'Prediction', 'Actual', 'Confusion Matrix with TsLearn', TARGET_LETTERS)

    Plotter.plot_bar([0.0, 1.0], balance_acc_score, models, "Balance Accuracy Scores")

    print(SpellerModel.predict_using_model("sklearn_model.onnx", X_test))

def plot_graphs_from_data_frame(data_frame, label_dic):
    Plotter.plot_letter_sensors(TARGET_LETTERS, data_frame, label_dic, "Before standardized")
    Plotter.plot_letter_sensor(data_frame,label_dic)
def get_summ_stats_X_y_from_data_frame(data_frame):
    # create summary statistics from sensor datas
    data_frame_summ = SpellerPreProcesser.create_summary_statistics(data_frame)
    # extract X and y values from summ stat data
    X, y = SpellerPreProcesser.extract_X_y_summ_sta(data_frame_summ)
    return X, y
def tslearn_train_session(X,y):
    # balance acc score list for different datasets
    balance_acc_score = []
    # names of the datasets
    models = []


    # train model with scikitlearn
    shuffles = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    conf_matrices = []
    min = -1
    max = 1
    print("------------ Standardizing data with ({},{}) ------------".format(min, max))
    _X = np.nan_to_num(X, copy=True)
    _X = SpellerPreProcesser.standardize(_X, min, max)
    tslearn_knn = KNeighborsTimeSeriesClassifier(n_neighbors=len(TARGET_LETTERS))
    print("------------ Splitting test,train set ------------")
    X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.2, random_state=42, stratify=y)
    print("------------ Training TsL Model ------------")
    tslearn_knn = tslearn_knn.fit(X_train, y_train)
    print("------------ Testing TsL Model ------------")
    y_pred = tslearn_knn.predict(X_test)
    conf_matrix_scikit = confusion_matrix(y_test, y_pred)
    model_desc = "TsL Test Data "
    models.append(model_desc)
    conf_matrices.append(conf_matrix_scikit)
    balance_acc_score.append(balanced_accuracy_score(y_test, y_pred))
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', 'Confusion {}'.format(model_desc),
                             TARGET_LETTERS)

    y_pred = tslearn_knn.predict(X_train)
    conf_matrix_scikit = confusion_matrix(y_train, y_pred)
    model_desc = "TsL Train Data"
    models.append(model_desc)
    conf_matrices.append(conf_matrix_scikit)
    balance_acc_score.append(balanced_accuracy_score(y_train, y_pred))
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', 'Confusion {}'.format(model_desc),
                             TARGET_LETTERS)
    # cross-validate model
    tslearn_knn = tslearn_knn.fit(X_test, y_test)
    y_pred = tslearn_knn.predict(X_train)
    conf_matrix_scikit = confusion_matrix(y_train, y_pred)
    model_desc = "TsL Cross-Train "
    models.append(model_desc)
    conf_matrices.append(conf_matrix_scikit)
    balance_acc_score.append(balanced_accuracy_score(y_train, y_pred))
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', 'Confusion {}'.format(model_desc),
                             TARGET_LETTERS)

    y_pred = tslearn_knn.predict(X_test)
    conf_matrix_scikit = confusion_matrix(y_test, y_pred)
    model_desc = "TsL Cross-Test "
    models.append(model_desc)
    conf_matrices.append(conf_matrix_scikit)
    balance_acc_score.append(balanced_accuracy_score(y_test, y_pred))
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', 'Confusion {}'.format(model_desc),
                             TARGET_LETTERS)

    # Plot performance of two models
    print("------------ Plotting results ------------")
    Plotter.plot_bar(np.arange(start=0.0, stop=len(models), step=1.0), balance_acc_score, models,
                     "Balance Accuracy Scores")

    return model_desc, balance_acc_score
def sklearn_train_session(X, y):
    # balance acc score list for different datasets
    balance_acc_score = []
    # names of the datasets
    models = []


    # train model with scikitlearn
    shuffles = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    conf_matrices = []
    min = -1
    max = 1
    print("------------ Standardizing data with ({},{}) ------------".format(min, max))
    _X = np.nan_to_num(X, copy=True)
    _X = SpellerPreProcesser.standardize(_X, min, max)
    sklearn_knn = KNeighborsClassifier(n_neighbors=len(TARGET_LETTERS))
    print("------------ Splitting test,train set ------------")
    X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.2, random_state=42, stratify=y)
    print("------------ Training SciKit Model ------------")
    sklearn_knn = sklearn_knn.fit(X_train, y_train)
    print("------------ Testing SciKit Model ------------")
    y_pred = sklearn_knn.predict(X_test)
    conf_matrix_scikit = confusion_matrix(y_test, y_pred)
    model_desc = "SciKit Test Data "
    models.append(model_desc)
    conf_matrices.append(conf_matrix_scikit)
    balance_acc_score.append(balanced_accuracy_score(y_test, y_pred))
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', 'Confusion {}'.format(model_desc),
                             TARGET_LETTERS)

    y_pred = sklearn_knn.predict(X_train)
    conf_matrix_scikit = confusion_matrix(y_train, y_pred)
    model_desc = "SciKit Train Data"
    models.append(model_desc)
    conf_matrices.append(conf_matrix_scikit)
    balance_acc_score.append(balanced_accuracy_score(y_train, y_pred))
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', 'Confusion {}'.format(model_desc),
                             TARGET_LETTERS)
    # cross-validate model
    sklearn_knn = sklearn_knn.fit(X_test, y_test)
    y_pred = sklearn_knn.predict(X_train)
    conf_matrix_scikit = confusion_matrix(y_train, y_pred)
    model_desc = "SciKit Cross-Train "
    models.append(model_desc)
    conf_matrices.append(conf_matrix_scikit)
    balance_acc_score.append(balanced_accuracy_score(y_train, y_pred))
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', 'Confusion {}'.format(model_desc),
                             TARGET_LETTERS)

    y_pred = sklearn_knn.predict(X_test)
    conf_matrix_scikit = confusion_matrix(y_test, y_pred)
    model_desc = "SciKit Cross-Test "
    models.append(model_desc)
    conf_matrices.append(conf_matrix_scikit)
    balance_acc_score.append(balanced_accuracy_score(y_test, y_pred))
    Plotter.plot_conf_matrix(conf_matrix_scikit, 'Prediction', 'Actual', 'Confusion {}'.format(model_desc),
                             TARGET_LETTERS)

    # Plot performance of two models
    print("------------ Plotting results ------------")
    Plotter.plot_bar(np.arange(start=0.0, stop=len(models), step=1.0), balance_acc_score, models,
                     "Balance Accuracy Scores")

    return model_desc, balance_acc_score
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    label_dict = create_label_dict(TARGET_LETTERS)
    data_frame = SpellerDatabase.read_letters_from_database(SpellerConstant.FIREBASE_REFERENCE,tuple(label_dict))
    #plot_graphs_from_data_frame(data_frame,label_dict)
    X, y = get_summ_stats_X_y_from_data_frame(data_frame)
    #X, y = read_letters_from_npy('X_timeseries.npy', 'y_timeseries.npy')
    sk_model_desc, sk_model_sc = sklearn_train_session(X, y)
    ts_model_desc, ts_model_sc = tslearn_train_session(X, y)
    desc = ts_model_desc + sk_model_desc
    scores = ts_model_sc + ts_model_desc
    print("------------ Plotting results ------------")
    Plotter.plot_bar(np.arange(start=0.0, stop=len(desc), step=1.0), scores, desc,
                     "Balance Accuracy Scores Ts-Sk")

# balance acc score list for different datasets

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
