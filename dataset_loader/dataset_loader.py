# see datasets here: https://www.tensorflow.org/datasets/catalog/overview

import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import cv2
import numpy as np

data, info = tfds.load("cats_vs_dogs", with_info=True)

train_data = data['train']
# test_data = data['test']

train_data = train_data.batch(1)

i = 1
for sample in train_data:
    print('{0:04}) {1}'.format(i, sample['image'].numpy().shape))
    img = np.squeeze(sample['image'].numpy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    i += 1