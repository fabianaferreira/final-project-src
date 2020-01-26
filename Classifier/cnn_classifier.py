from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LambdaCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

import pandas as pd
import numpy as np

BATCH_SIZE = 64


def createModel():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation='elu', input_shape=(176, 240, 1), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128, activation='elu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


# Loading data
X_train = np.load('./datasets/CNN/X_train_no_edge_frames_subset.npy')
X_test = np.load('./datasets/CNN/X_test_no_edge_frames_subset.npy')
y_train = np.load('./datasets/CNN/y_train_no_edge_frames_subset.npy')
y_test = np.load('./datasets/CNN/y_test_no_edge_frames_subset.npy')

# Creating model
model = createModel()

y_train = np.argmax(y_train, axis=1).astype(str)
y_test = np.argmax(y_test, axis=1).astype(str)

df_train = pd.DataFrame({'X': X_train, 'y': y_train})
df_test = pd.DataFrame({'X': X_test, 'y': y_test})

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    samplewise_center=True,
    rotation_range=40,
    horizontal_flip=False,
    vertical_flip=False,
    validation_split=0.2,
    fill_mode='nearest'
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='X',
    y_col='y',
    target_size=(176, 240),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training')  # set as training data

validation_gen = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='X',
    y_col='y',
    target_size=(176, 240),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation')

# Defining callbacks
# TODO: Check if folder exist and create them if not
epochs_to_wait_for_improvement = 20
logging_path = './logs'
models_path = './models'
model_name = 'CNN_no_edge_frames_' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
early_stopping = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improvement)
checkpoint = ModelCheckpoint(f'{models_path}/{model_name}.h5', monitor='val_loss', save_best_only=True, mode='min')
csv_logger = CSVLogger(f'{logging_path}/{model_name}.log')

callbacks = [early_stopping, checkpoint, csv_logger]

print('Training model... You should get a coffee...')
# Fit the model
print(model.summary())
# print(model_name)
# exit(1)
model.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    epochs=1000,
    verbose=1,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples // BATCH_SIZE,
    callbacks=callbacks,
    # class_weight=[12.39411284, 5.43687231, 11.48333333, 4.59194184, 9.109375, 5.06617647, 8.10588235]
    # class_weight=[2, 1, 2, 1, 1.5, 1, 1.5]
    class_weight=[1.76699708, 0.7763886, 1.63858549, 0.65533498, 1.30395869,
                  0.72539257, 1.15690616]
)
