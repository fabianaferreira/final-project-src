from DataGenerator import DataGenerator
from ConfusionMatrix import ConfusionMatrix
from tensorflow.keras.models import load_model
import numpy as np

BATCH_SIZE = 64
SUBSET = True
MODELS_DIR_PATH = './models/'


def generate_plot(model_name):
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
    model = load_model(model_name)

    print('Creating an instance of DataGenerator for X_test and y_test...')
    # Using image size as default so as to be able to use them in models that
    # contains pre-trained networks that requires specific image size such
    # as VGG and InceptionV3
    test_gen = DataGenerator(X_test, y_test, BATCH_SIZE, img_rows=176, img_cols=240, channel=1)

    print('Predicting...')
    result = model.predict_generator(test_gen)

    fig_name = 'confusion_matrix_model_' + model_name.rsplit('/', 1)[1][:-3] + ('_subset' if SUBSET else '')
    cm = ConfusionMatrix(labels, y_pred=y_test, y_true=result)

    print('Saving matrix...')
    cm.save_matrix(filename=fig_name)

    # print('Saving plot...')
    # cm.plot_figure(normalize=True, show_annotations=True, save_fig=True, figname=fig_name)


model = MODELS_DIR_PATH + 'CNN_no_edge_frames_data_augmentation_2020_01_21_10:30.h5'
generate_plot(model)
