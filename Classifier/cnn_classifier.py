from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, BatchNormalization, MaxPooling2D, \
    AveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

import pandas as pd
import numpy as np

# from DataGenerator import DataGenerator

BATCH_SIZE = 64


def createModel():
    model = Sequential()
    # model.add(Conv2D(filters=16, kernel_size=(16, 16), input_shape=(176, 240, 1), kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(filters=32, kernel_size=(9, 9), kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(filters=32, kernel_size=(9, 9), kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(filters=64, kernel_size=(5, 5), kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(filters=64, kernel_size=(5, 5), kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(3, 3), strides=(3, 3)))

    # model.add(GlobalAveragePooling2D())

    # model.add(Flatten())
    # model.add(Dense(40, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dropout(0.5))
    # # model.add(Dense(48, activation='softmax'))
    # model.add(Dense(7, activation='softmax'))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(176, 240, 1), kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))

    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
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

# df_train = pd.concat([pd.DataFrame({'X': X_train}), pd.DataFrame(y_train)], axis=1)
# df_test = pd.concat([pd.DataFrame({'X': X_test}), pd.DataFrame(y_test)], axis=1)
df_train = pd.DataFrame({'X': X_train, 'y': y_train})
df_test = pd.DataFrame({'X': X_test, 'y': y_test})
# class_columns = [str(x) for x in (set(df_train.columns) - {'X'})]

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2
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

# train_gen = DataGenerator(X_train, y_train, 64)
# test_gen = DataGenerator(X_test, y_test, 64)

# Defining callbacks
# TODO: Check if folder exist and create them if not
epochs_to_wait_for_improvement = 10
logging_path = './logs'
models_path = './models'
model_name = 'CNN_no_edge_frames_' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

early_stopping = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improvement)
checkpoint = ModelCheckpoint(f'{models_path}/{model_name}.h5', monitor='val_loss', save_best_only=True, mode='min')
csv_logger = CSVLogger(f'{logging_path}/{model_name}.log')

callbacks = [early_stopping, checkpoint, csv_logger]

print('Training model... You should get a coffee...')
# Fit the model
print(model.summary())
print(model_name)
# exit(1)
model.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    epochs=1000,
    verbose=1,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples // BATCH_SIZE,
    callbacks=callbacks,
    class_weight=[12.39411284, 5.43687231, 11.48333333, 4.59194184, 9.109375, 5.06617647, 8.10588235]
)

# train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator,
#     validation_steps = validation_generator.samples // batch_size,
