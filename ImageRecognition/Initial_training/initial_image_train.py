 # Much of the code used here is based on: https://keras.io/examples/keras_recipes/tfrecord/
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, MaxPooling2D, Flatten, Dense, Dropout
from get_dataset import get_dataset
tf.compat.v1.enable_eager_execution()


def get_file_list(directory):
    file_list = []
    # Loading training dataset:
    for absolute_path in list(os.walk(directory))[0][2]:
        file_list.append(directory + absolute_path)
    return file_list

# Features as provided by MARCO
feature_description = {
    "image/height": tf.io.VarLenFeature(tf.int64),  # image height in pixels
    "image/width": tf.io.VarLenFeature(tf.int64),  # image width in pixels
    "image/colorspace": tf.io.VarLenFeature(tf.string),  # specifying the colorspace, always 'RGB'
    "image/channels": tf.io.VarLenFeature(tf.int64),  # specifying the number of channels, always 3
    "image/class/label": tf.io.VarLenFeature(tf.int64),  # specifying the index in a normalized classification layer
    "image/class/raw": tf.io.VarLenFeature(tf.int64),  # specifying the index in the raw (original) classification layer
    "image/class/source": tf.io.VarLenFeature(tf.int64),  # specifying the index of the source (creator of the image)
    "image/class/text": tf.io.VarLenFeature(tf.string),  # specifying the human-readable version of the normalized label
    "image/format": tf.io.VarLenFeature(tf.string),  # specifying the format, always 'JPEG'
    "image/filename": tf.io.VarLenFeature(tf.string),  # containing the basename of the image file
    "image/id": tf.io.VarLenFeature(tf.int64),  # specifying the unique id for the image
    "image/encoded": tf.io.VarLenFeature(tf.string),  # containing JPEG encoded image in RGB colorspace
}


def get_model():
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=1)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_SIZE)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(640, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # if Binary = True or something...
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    return model


train_dir = './train-jpg/'
test_dir = './test-jpg/'
valid_dir = './test-jpg/'

train_list = get_file_list(train_dir)
test_list = get_file_list(test_dir)
valid_list = get_file_list(valid_dir)

# Tuning and training params:
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 100
IMAGE_SIZE = [100, 100, 3]
keras_batch_size = BATCH_SIZE
learning_rate = 0.0001
epochs = 1

print("Train Files: ", len(train_list))
print("Test Files: ", len(test_list))

train_dataset = get_dataset(train_list, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
valid_dataset = get_dataset(test_list, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
test_dataset = get_dataset(test_list, labeled=False, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
model = get_model()
model.fit(train_dataset, validation_data=valid_dataset, batch_size=keras_batch_size, epochs=epochs)
model.save('./csgid-image-rec.h5')
