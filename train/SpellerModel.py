import numpy
import onnxruntime as rt


def predict_using_model(model_path, X_test):
    sess = rt.InferenceSession(
        model_path, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {input_name: X_test.astype(numpy.float32)})[0]
    return pred_onnx