import cv2
import os
import numpy as np
import glob
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt


def readTFRecord(serialized, categorical=False):

    data = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'mask': tf.io.FixedLenFeature((), tf.string),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'channelsImg': tf.io.FixedLenFeature((), tf.int64),
        'channelsMask': tf.io.FixedLenFeature((), tf.int64)
        }

    example = tf.io.parse_single_example(serialized, data)
    image = tf.image.decode_png(example['image'])
    mask = tf.image.decode_png(example['mask'])
    height = example['height']
    width = example['width']
    channelsImg = example['channelsMask']
    channelsMask = example['channelsMask']
    mask = mask[:,:,0]

    image = tf.reshape(image, (224, 224, 3))
    image = tf.cast(image, tf.float32)
    mask = tf.reshape(mask, (224, 224))
    mask =  tf.cast(mask, tf.float32)

    if categorical:
        mask = tf.keras.utils.to_categorical(mask)

    return image, mask


def getRecordNumber(tfrecords):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=4)
    dataset = dataset.map(readTFRecord, num_parallel_calls=4)

    for i, d in enumerate(dataset):
        num=i

    return num

def getShards(tfrecords):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    #print('HEYHEYHEYHEYHEYHEY', tfRecordsPaths, flush=True)
    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=4)
    dataset = dataset.map(readTFRecord, num_parallel_calls=4)

    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(16, drop_remainder=True)
    dataset = dataset.prefetch(4)

    return dataset


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--tfrecordpath', required=True, help='path to tfrecord')
    ap.add_argument('-c', '--categorical', help='binary or categorical - default is binary')
    ap.add_argument('-n', '--number', help='get the number of records')
    args = vars(ap.parse_args())

    tfRecordPaths = os.path.join(args['tfrecordpath'],'*.tfrecords')
    print(tfRecordsPaths)
    if args['number'] is not None:
        number = getRecordNumber(tfrecords)
        print('The number is: {}'.format(number), flush=True)

    dataset = getShards(tfRecordPaths)
