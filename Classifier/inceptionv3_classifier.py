from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, BatchNormalization, MaxPooling2D, \
    GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

import numpy as np
import pandas as pd
from DataGenerator import DataGenerator

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K


def createModel():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(100, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(48, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model


# Loading data
X_train = np.load('./datasets/CNN/X_train_no_edge_frames.npy')
X_test = np.load('./datasets/CNN/X_test_no_edge_frames.npy')
y_train = np.load('./datasets/CNN/y_train_no_edge_frames.npy')
y_test = np.load('./datasets/CNN/y_test_no_edge_frames.npy')

# Creating model
model = createModel()

train_gen = DataGenerator(X_train, y_train, 128, grayscale=False)
test_gen = DataGenerator(X_test, y_test, 128, grayscale=False)

# Defining callbacks
# TODO: Check if folder exist and create them if not
epochs_to_wait_for_improvement = 10
logging_path = './logs'
models_path = './models'
model_name = 'InceptionV3_no_edge_frames'

early_stopping = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improvement)
checkpoint = ModelCheckpoint(f'{models_path}/{model_name}.h5', monitor='val_loss', save_best_only=True, mode='min')
csv_logger = CSVLogger(f'{logging_path}/{model_name}.log')

callbacks = [early_stopping, checkpoint, csv_logger]

print('Training model... You should get a coffee...')
# Fit the model
model.fit_generator(
    generator=train_gen,
    steps_per_epoch=len(train_gen),
    epochs=1000,
    verbose=1,
    validation_data=test_gen,
    validation_steps=len(test_gen),
    callbacks=callbacks
)