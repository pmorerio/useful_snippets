import tensorflow as tf
import os
import time
import glob


SPLIT = 'train'
list_file = glob.glob1('/data/datasets/stl10/', '*{}*.tfrecords'.format(SPLIT))
list_file = [os.path.join('/data/datasets/stl10', i) for i in list_file]
raw_image_dataset = tf.data.TFRecordDataset(list_file)



def _parse_example(example_proto):
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'image_path': tf.io.VarLenFeature(tf.string),
    }

    # Parse the input tf.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.cast(tf.io.decode_jpeg(example['image_raw']), tf.uint8)
    label = tf.cast(example['label'], tf.int64)
    return {
        'image': image,
        'label': label
    }



raw_example = raw_image_dataset.map(_parse_example)

for e in range(5):
    count = 1
    t0 = time.time()
    for example in raw_example:
        print('{}/{} - {}'.format(e + 1, 5, count))
        count += 1
    print('Time for epoch: {}'.format(time.time() - t0))
