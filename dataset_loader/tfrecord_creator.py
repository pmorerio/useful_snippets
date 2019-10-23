import tensorflow as tf
import os
import numpy as np
import random as rd
import cv2
import warnings
import tqdm
import sys
import time
import glob
import pickle
from matplotlib import pyplot as plt


#
# def load_image(addr):
#     img = cv2.imread(addr)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32)
#     img /= 255
#     return img
#
# def load_encoded(addr):
#     with open(addr, 'rb') as f:
#         png_bytes = f.read()
#     return png_bytes


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_from_dirs(data_dir, class_list):
    class_dirs = []
    for i, name in enumerate(class_list):
        class_dirs.append(os.path.join(data_dir, name))
    class_dirs.sort()
    files = []
    class_id = []
    for i, mydir in enumerate(class_dirs):
        imgs = glob.glob1(mydir, '*.png')
        imgs = [os.path.join(mydir, n) for n in imgs]
        files += imgs
        labels = np.ones((len(imgs),), dtype=np.float32) * i
        class_id = class_id + labels.tolist()
    return files, class_id, class_dirs


def serialize_example(image_path, label):
    """
    :param image_path: path of the image
    :param label: label of the image
    :return: serialized example
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    image_string = open(image_path, 'rb').read()
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
        'image_path': _bytes_feature(image_path.encode('utf-8')),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




def write_TFRecords(directory, filename, class_list, imgs_for_file=1000):
    def chunks(l, n):
        n = max(1, n)
        return [l[i:i + n] for i in range(0, len(l), n)]

    files, class_id, class_names = read_from_dirs(directory, class_list)

    # shuffle in order to make more files
    # ord = list(range(len(crops_files)))
    rd.Random(4).shuffle(files)
    rd.Random(4).shuffle(class_id)
    # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    file_chunk = chunks(files, imgs_for_file)
    label_chunk = chunks(class_id, imgs_for_file)
    counter = list(range(len(label_chunk)))
    for im, id, count in zip(file_chunk, label_chunk, counter):
        print('processing chunk: {}/{}'.format(count + 1, len(counter)))
        tmp_filename = '{0}_{1:03}.tfrecords'.format(filename, count)
        writer = tf.io.TFRecordWriter(tmp_filename)
        for i in tqdm.tqdm(range(len(im)), desc='processing chunk'):

            # cathegorical_label = np.zeros((len(class_names),), np.int)
            # cathegorical_label[int(id[i])] = 1

            # Create an example protocol buffer and serialize to string
            label = np.int64(id[i])
            image = im[i]
            example = serialize_example(image, label)

            # write on the file
            writer.write(example)
        writer.close()
        sys.stdout.flush()



def main():
    DATA_DIR = '/data/datasets/stl10'
    SPLIT = 'test'
    tfrecords_filename = os.path.join(DATA_DIR, 'stl10_{}'.format(SPLIT))
    write_TFRecords(os.path.join(DATA_DIR, SPLIT), filename=tfrecords_filename, class_list=os.listdir(os.path.join(DATA_DIR, SPLIT)))

if __name__ == '__main__':
    main()
