import cv2
import os
import random
import numpy as np
import glob
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt


class Augment():

    def getRotate90(self, x, y):

        rand = tf.random.uniform(shape=[],minval=0,maxval=4,dtype=tf.int32)
        x= tf.image.rot90(x, rand)
        y= tf.image.rot90(y, rand)

        return x, y


    def getRotate(self, x, y):

        if tf.random.uniform(())>0.5:
            degree = tf.random.normal([])*360
            x = tfa.image.rotate(x, degree * math.pi / 180, interpolation='BILINEAR')
            y = tfa.image.rotate(y, degree * math.pi / 180, interpolation='BILINEAR')

        return x, y


    def getFlip(self, x, y):

        if tf.random.uniform(())> 0.5:
            x=tf.image.flip_left_right(x)
            y=tf.image.flip_left_right(y)

        if tf.random.uniform(())> 0.5:
            x=tf.image.flip_up_down(x)
            y=tf.image.flip_up_down(y)

        return x, y


    def getColor(self, x, y):

        if tf.random.uniform(()) > 0.5:
            x = tf.image.random_hue(x, 0.05)
        if tf.random.uniform(()) > 0.5:
            x = tf.image.random_saturation(x, 0.5, 1.2)
        if tf.random.uniform(()) > 0.5:
            x = tf.image.random_brightness(x, 0.25)
        if tf.random.uniform(()) > 0.5:
            x = tf.image.random_contrast(x, 0.8, 1.5)

        return x, y


    def getCrop(self, x, y):

        rand = tf.random.uniform((), minval=0.6, maxval=1)
        x = tf.image.central_crop(x, central_fraction=rand)
        y = tf.image.central_crop(y, central_fraction=rand)
        return x, y


    def standardized(self, x, y):
        x = tf.image.per_image_standardization(x)
        return x, y


    def convertMask(self, x, y):
        y = y[:,:,0]
        return x, y


    def categoricalMask(self, x, y):
        y = tf.keras.utils.to_categorical(y)
        print('categorixal babbbyyyyy',y.shape)
        return x, y



def readTFRecord(serialized, imgDims=256):

    data = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'mask': tf.io.FixedLenFeature((), tf.string)
        #'imagename': tf.io.FixedLenFeature((), tf.string),
        #'maskname': tf.io.FixedLenFeature((), tf.string)
        #'dims': tf.io.FixedLenFeature((), tf.int64)
        }

    example = tf.io.parse_single_example(serialized, data)
    image = tf.image.decode_png(example['image'])
    mask = tf.image.decode_png(example['mask'])

    return image, mask


def getRecordNumber(tfrecords):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=4)
    dataset = dataset.map(readTFRecord, num_parallel_calls=4)

    for i, d in enumerate(dataset):
        num=i

    return num


def getShards(tfrecords, dataSize, batchSize, imgDims, test=False, augmentations=[], categorical=False):

    aug = Augment()
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=4)
    dataset = dataset.map(readTFRecord, num_parallel_calls=4)
    dataset = dataset.map(lambda x, y: (tf.reshape(x,(imgDims, imgDims, 3)), tf.reshape(y,(imgDims, imgDims, 3))))
    dataset = dataset.map(lambda x, y: (tf.cast(x,tf.float32), tf.cast(y,tf.float32)))
    
    if len(augmentations)>0:
        print('Applying following Augmentations')
        for i, a in enumerate(augmentations):
            print('{}: {}'.format(i, a))
        for f in augmentations:
            dataset = dataset.map(getattr(aug, 'get'+f), num_parallel_calls=4)
            #dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y),  num_parallel_calls=4)
    else:
        print('No data augmentation being applied')
    
    dataset = dataset.map(aug.convertMask, num_parallel_calls=4)
    
    if categorical:
        dataset = dataset.map(aug.categoricalMask)
  
    if not test:
        dataset = dataset.cache()
        dataset = dataset.shuffle(dataSize)
        dataset = dataset.repeat()
        dataset = dataset.batch(batchSize, drop_remainder=True)
        dataset = dataset.prefetch(4)

    return dataset


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--tfrecordpath', required=True, help='path to tfrecord')
    ap.add_argument('-c', '--categorical', help='binary or categorical - default is binary')
    ap.add_argument('-n', '--number', help='get the number of records')
    ap.add_argument('-a', '--augment', help='augmentation flag')
    args = vars(ap.parse_args())

    tfRecordPaths = os.path.join(args['tfrecordpath'],'*.tfrecords')

    if args['number'] is not None:
        number = getRecordNumber(tfrecords)
        print('The number is: {}'.format(number), flush=True)

    dataset = getShards(tfRecordPaths, augment)
