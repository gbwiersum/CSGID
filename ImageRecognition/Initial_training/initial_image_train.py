# Much of the code used here is based on: https://keras.io/examples/keras_recipes/tfrecord/
import tensorflow as tf
from ImageRecognition.utils.DataGenerator import get_dataset
from ImageRecognition.utils.Inceptionv4 import create_model
import glob

train_list = glob.glob('../train-jpg/*')[0:2]
test_list = glob.glob('../test-jpg/*')[0:1]
valid_list = test_list

# Tuning and training params:
BATCH_SIZE = 1
IMAGE_SIZE = [299, 299, 3]
keras_batch_size = BATCH_SIZE
learning_rate = 0.095
epochs = 1

print("Train Files: ", len(train_list))
print("Test Files: ", len(test_list))

train_dataset = get_dataset(train_list, batch_size=BATCH_SIZE, im_size=IMAGE_SIZE)
valid_dataset = get_dataset(test_list, batch_size=BATCH_SIZE, im_size=IMAGE_SIZE)
test_dataset = get_dataset(test_list, labeled=False, batch_size=BATCH_SIZE, im_size=IMAGE_SIZE)

model = create_model(num_classes=1)
model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy,
              metrics=tf.keras.metrics.binary_accuracy)

model.fit(train_dataset, validation_data=valid_dataset, batch_size=keras_batch_size, epochs=epochs)
model.save('../Models/0000csgid-image-rec.h5')
