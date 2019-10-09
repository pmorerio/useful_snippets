import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import matplotlib.pyplot as plt


class resnet_model(Model):
    def __init__(self, img_shape):
        super(resnet_model, self).__init__()
        self.resnet_output = tf.keras.applications.ResNet50(input_shape=img_shape,
                                                            include_top=False,
                                                            weights='imagenet')
        self.avgpool = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_out = tf.keras.layers.Dense(12, activation='softmax')

    def call(self, x):
        self.features = self.avgpool(self.resnet_output(x))
        x = self.flatten(self.features)
        return self.dense_out(x)