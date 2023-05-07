# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tslearn.metrics import dtw
import string
from database.SpellerDatabase import *
import plot.SpellerPlotter as Plotter
import SpellerConstant
from train import SpellerStandardizer
from train import SpellerTrainer
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skl2onnx.common.data_types import FloatTensorType

TARGET_LETTERS = list(string.ascii_uppercase)


def create_label_dict(letter_list: []):
    labels = {}
    for num, letter in enumerate(letter_list, start=1):
        labels[letter] = float(num)
    return labels


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    speller_database = SpellerDatabase(SpellerConstant.FIREBASE_REFERENCE, SpellerConstant.SENSOR_AXES)
    data_frame = speller_database.read_letters_from_database(TARGET_LETTERS)
    # summary statistics
    # #Plotter.plot_letter_sensors(TARGET_LETTERS, data_frame, create_label_dict(TARGET_LETTERS), "Before standardized")
    data_frame_summ = SpellerStandardizer.create_summary_statistics(data_frame)
    X, y = SpellerStandardizer.extract_X_y_summ_sta(data_frame_summ)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sklearn_knn = KNeighborsClassifier(n_neighbors=len(TARGET_LETTERS))
    trained_model = SpellerTrainer.train_model(X_train, y_train, sklearn_knn)
    y_pred = trained_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('--------------- Summary Statistics ---------------')
    print(SpellerStandardizer.multiclass_roc_auc_score(y_test,y_pred))
    print(classification_report(y_test, y_pred,target_names=TARGET_LETTERS))
    Plotter.plot_conf_matrix(conf_matrix, 'Prediction', 'Actual', 'Confusion Matrix Summ Stat')
    print(type(trained_model))
    initial_type = [
        ('input', FloatTensorType([None, 1200]))
    ]
    SpellerTrainer.convert_to_onnx(trained_model, "sklearn_summ_model", initial_type)

    # sensor data
    X, y = SpellerStandardizer.extract_X_y(data_frame)
    X = np.nan_to_num(X, copy=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sklearn_knn = KNeighborsClassifier(n_neighbors=len(TARGET_LETTERS))
    trained_model = SpellerTrainer.train_model(X_train, y_train, sklearn_knn)
    y_pred = trained_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('--------------- Sensor Statistics ---------------')
    print(SpellerStandardizer.multiclass_roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=TARGET_LETTERS))
    Plotter.plot_conf_matrix(conf_matrix, 'Prediction', 'Actual', 'Confusion Matrix')
    print(type(trained_model))
    initial_type = [
        ('input', FloatTensorType([None, 1200]))
    ]
    SpellerTrainer.convert_to_onnx(trained_model, "sklearn_model", initial_type)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
