from tslearn.metrics import dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from skl2onnx import convert_sklearn
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_helper import save_onnx_model

def train_model(X_train, y_train, knn):
    knn.fit(X_train, y_train)
    return knn

def convert_to_onnx(model, model_name, init_types):

    # Write the ONNX model to disk
    converted_model = convert_sklearn(model, initial_types=init_types)
    save_onnx_model(converted_model,'{}.onnx'.format(model_name))
    return model_name

def convert_to_onnx(model, model_name, X):
    converted_model = to_onnx(model, X, model_name)
    save_onnx_model(converted_model,'{}.onnx'.format(model_name))
    return model_name