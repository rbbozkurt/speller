# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tslearn.metrics import dtw
import string
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

TARGET_LETTERS = list(string.ascii_uppercase)


def create_label_dict(letter_list: []):
    labels = {}
    for num, letter in enumerate(letter_list, start=1):
        labels[letter] = float(num)
    return labels


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # read data from database
    speller_database = SpellerDatabase(SpellerConstant.FIREBASE_REFERENCE, SpellerConstant.SENSOR_AXES)
    data_frame = speller_database.read_letters_from_database(TARGET_LETTERS)

    # balance acc score list for different datasets
    balance_acc_score = []

    # names of the datasets
    models = ['Summary statistics with SciKit', 'Sensor data with SciKit']

    # summary statistics with SciKit

    # #Plotter.plot_letter_sensors(TARGET_LETTERS, data_frame, create_label_dict(TARGET_LETTERS), "Before standardized")
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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
