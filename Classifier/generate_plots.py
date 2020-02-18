from DataGenerator import DataGenerator
from ConfusionMatrix import ConfusionMatrix
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import top_k_categorical_accuracy
import numpy as np
import pandas as pd

BATCH_SIZE = 64
FRAMES = 5
SUBSET = True
MODELS_DIR_PATH = './models/'
CONFUSION_MATRIX_DIR = './Confusion_Matrix/'


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def generate_plot(model_name):
    # Loading labels and data
    if SUBSET:
        labels = np.load('./Plots/labels_new_subset.npy', allow_pickle=True)
        X_test = np.load('./datasets/CNN/X_test_no_edge_frames_new_subset_' + str(FRAMES) + '.npy')
        y_test = np.load('./datasets/CNN/y_test_no_edge_frames_new_subset_' + str(FRAMES) + '.npy')
    else:
        labels = np.load('./Plots/labels_CM_others.npy', allow_pickle=True)
        X_test = np.load('./datasets/CNN/X_test_no_edge_frames_' + str(FRAMES) + '.npy')
        y_test = np.load('./datasets/CNN/y_test_no_edge_frames_' + str(FRAMES) + '.npy')
    # Loading model
    print('Loading model...')
    dependencies = {'top_2_accuracy': top_2_accuracy,
                    'top_3_accuracy': top_3_accuracy}
    model = load_model(model_name, custom_objects=dependencies)

    print('Creating an instance of DataGenerator for X_test and y_test...')
    # Using image size as default so as to be able to use them in models that
    # contains pre-trained networks that requires specific image size such
    # as VGG and InceptionV3

    test_gen = DataGenerator(X_test, y_test, BATCH_SIZE)

    print('Predicting...')
    result = model.predict_generator(test_gen)

    fig_name = 'confusion_matrix_model_' + model_name.rsplit('/', 1)[1][:-3] + '_' + str(FRAMES) + ('_subset' if SUBSET else '')
    cm = ConfusionMatrix(labels, y_true=y_test, y_pred=result)

    print('Saving matrix...')
    cm.save_matrix(filename=fig_name)

model = MODELS_DIR_PATH + 'fine_tune_VGG16_no_edge_frames_2020-02-16-19:46:21.h5'
generate_plot(model)
