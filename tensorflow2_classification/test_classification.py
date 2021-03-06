import tensorflow as tf
import numpy as np
import glob
import os
import datetime
import time
import matplotlib.pyplot as plt

from model import resnet_model

data_path = '/data/prota/coco_12_realsize'
data_path_validation = data_path + '_val'

CLASS_NAMES = np.array([item for item in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, item)) is True])

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

BATCH_SIZE  = 32
IMG_HEIGHT  = 224
IMG_WIDTH   = 224
N_EPOCHS    = 50

# check here: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10)
train_data_gen = image_generator.flow_from_directory(directory=str(data_path), batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), classes=list(CLASS_NAMES))
test_data_gen = image_generator.flow_from_directory(directory=str(data_path_validation), batch_size=BATCH_SIZE, shuffle=False, target_size=(IMG_HEIGHT, IMG_WIDTH), classes=list(CLASS_NAMES))

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

model = resnet_model(IMG_SHAPE)

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
writer = tf.summary.create_file_writer("/tmp/mylogs/test_classification_dataaugmentation_{}".format(timestamp))

@tf.function
def train_step(images, labels, step):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    eq = tf.equal(tf.argmax(labels, -1), tf.argmax(predictions, -1))
    accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))
    with writer.as_default():
        tf.summary.scalar("training_accuracy", accuracy, step=step)
    return predictions, loss

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    eq = tf.equal(tf.argmax(labels, -1), tf.argmax(predictions, -1))
    accuracy = tf.reduce_mean(tf.cast(eq, tf.float32))
    return predictions, t_loss, accuracy

epoch = 0
step = 1
t0 = time.time()
for batch, label in train_data_gen:
    if train_data_gen.batch_index == 1:
        print('test for epoch {}'.format(epoch))
        test_accuracy_batch_list = []
        for test_batch, test_label in test_data_gen:
            if test_data_gen.batch_index == 0:
                break
            predictions, t_loss, batch_accuracy = test_step(test_batch, test_label)
            test_accuracy_batch_list.append(batch_accuracy.numpy())
        test_accuracy = np.asarray(test_accuracy_batch_list).mean()
        with writer.as_default():
            tf.summary.scalar("test_accuracy", test_accuracy * 100, step=epoch + 1)
        print('Test Accuracy at epoch [{0:03}]: {1:0.04}'.format(epoch, test_accuracy * 100))

        epoch += 1
        if epoch == N_EPOCHS:
            break
        print('Starting epoch {}/{}'.format(epoch, N_EPOCHS))

    train_preds, train_loss = train_step(batch, label, tf.convert_to_tensor(step, tf.int64))
    print('[{0:03}] loss: {1:0.05}'.format(train_data_gen.batch_index, train_loss.numpy()))
    step += 1

print('training time: {} sec'.format(time.time() - t0))