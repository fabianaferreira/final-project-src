from DataGenerator import DataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import top_k_categorical_accuracy
import numpy as np
import pandas as pd

BATCH_SIZE = 64
SUBSET = True
MODELS_DIR_PATH = './models/'
METRICS_DIR = './Metrics/'
LOGS_DIR = './logs/'


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def generate_metrics(model_name):
    # Loading labels and data
    if SUBSET:
        labels = np.load('./Plots/labels_CM_subset.npy', allow_pickle=True)
        X_test = np.load('./datasets/CNN/X_test_no_edge_frames_subset.npy')
        y_test = np.load('./datasets/CNN/y_test_no_edge_frames_subset.npy')
    else:
        labels = np.load('./Plots/labels_CM.npy', allow_pickle=True)
        X_test = np.load('./datasets/CNN/X_test_no_edge_frames.npy')
        y_test = np.load('./datasets/CNN/y_test_no_edge_frames.npy')
    # Loading model
    print('Loading model...')
    dependencies = {'top_2_accuracy': top_2_accuracy,
                    'top_3_accuracy': top_3_accuracy}
    model = load_model(model_name, custom_objects=dependencies)

    print('Creating an instance of DataGenerator for X_test and y_test...')
    # Using image size as default so as to be able to use them in models that
    # contains pre-trained networks that requires specific image size such
    # as VGG and InceptionV3

    # test_gen = DataGenerator(X_test, y_test, BATCH_SIZE, img_rows=176, img_cols=240, channel=1)
    test_gen = DataGenerator(X_test, y_test, BATCH_SIZE)

    print('Predicting...')
    result = model.predict_generator(test_gen)

    aux_model_name = model_name.rsplit('/', 1)[1][:-3]

    report_name = 'classification_report_' + aux_model_name + ('_subset' if SUBSET else '')
    history_name = 'history_' + aux_model_name + ('_subset' if SUBSET else '')

    print('Calculating other metrics...')
    report = classification_report(target_names=labels, y_true=np.argmax(result, axis=1),
                                   y_pred=np.argmax(y_test, axis=1), output_dict=True)

    print('Saving report...')
    df = pd.DataFrame(report).transpose()
    df.to_csv(METRICS_DIR + report_name + '.csv')

    print('Saving history...')
    df = pd.read_csv(LOGS_DIR + aux_model_name + '.log')
    df.to_csv(METRICS_DIR + history_name + '.csv')


model = MODELS_DIR_PATH + 'fine_tune_VGG16_no_edge_frames_2020-01-28-23:26:54.h5'
generate_metrics(model)
