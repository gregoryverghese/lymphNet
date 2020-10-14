import cv2
import os
import random
import numpy as np
import glob
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import math
from prettytable import PrettyTable


class Augment():
    def __init__(self, hueLimits, saturationLimits, contrastLimits, brightnessLimits,
                 rotateProb=0.5, flipProb=0.5, colorProb=0.5, channelMeans=[0,0,0], channelStd=[1,1,1]): 

        self.hueLimits = hueLimits
        self.saturationLimits = saturationLimits
        self.contrastLimits = contrastLimits
        self.brightnessLimits = brightnessLimits
        self.rotateProb = rotateProb
        self.flipProb = flipProb
        self.colorProb = colorProb
        self.channelMeans = channelMeans
        self.channelStd = channelStd
    

    def getRotate90(self, x, y):

        rand = tf.random.uniform(shape=[],minval=0,maxval=4,dtype=tf.int32)
        x= tf.image.rot90(x, rand)
        y= tf.image.rot90(y, rand)

        return x, y


    def getRotate(self, x, y):

        if tf.random.uniform(())>self.rotateProb:
            degree = tf.random.normal([])*360
            x = tfa.image.rotate(x, degree * math.pi / 180, interpolation='BILINEAR')
            y = tfa.image.rotate(y, degree * math.pi / 180, interpolation='BILINEAR')

        return x, y


    def getFlip(self, x, y):

        if tf.random.uniform(())> self.flipProb:
            x=tf.image.flip_left_right(x)
            y=tf.image.flip_left_right(y)

        if tf.random.uniform(())> self.flipProb:
            x=tf.image.flip_up_down(x)
            y=tf.image.flip_up_down(y)

        return x, y


    def getColor(self, x, y):

        if tf.random.uniform(()) > self.colorProb:
            x = tf.image.random_hue(x, self.hueLimits)
        if tf.random.uniform(()) > self.colorProb:
            x = tf.image.random_saturation(x, self.saturationLimits[0],
                                           self.saturationLimits[1])
        if tf.random.uniform(()) > self.colorProb:
            x = tf.image.random_brightness(x, self.brightnessLimits)
        if tf.random.uniform(()) > self.colorProb:
            x = tf.image.random_contrast(x, self.contrastLimits[0],
                                         self.contrastLimits[1])

        return x, y


    def getCrop(self, x, y):

        rand = tf.random.uniform((), minval=0.6, maxval=1)
        x = tf.image.central_crop(x, central_fraction=rand)
        y = tf.image.central_crop(y, central_fraction=rand)
        return x, y


    def getStandardizeImage(self, x, y):
        x = tf.image.per_image_standardization(x)
        return x, y


    def getStandardizeDataset(self, x, y):

        #xnew = (x - self.channelMeans)/self.channelStd
        #xnew = tf.clip_by_value(xnew,-1.0, 1.0)
        #xnew = (xnew+1.0)/2.0
        x=x
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

    print(image)

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


def getShards(tfrecords, dataSize, batchSize, imgDims, augParams={},
              augmentations=[], test=False, taskType='binary',
              channelMeans=[1,1,1],
              channelStd=[1,1,1]):	
    
    AUTO = tf.data.experimental.AUTOTUNE
    ignoreDataOrder = tf.data.Options()
    ignoreDataOrder.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = dataset.with_options(ignoreDataOrder)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=16, num_parallel_calls=AUTO)

    dataset = dataset.map(readTFRecord, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x, y: (tf.reshape(x,(imgDims, imgDims, 3)), tf.reshape(y,(imgDims, imgDims, 3))))
    dataset = dataset.map(lambda x, y: (tf.cast(x,tf.float16), tf.cast(y,tf.float16)))
    
    datasetName='train' if not test else 'test'
    print(datasetName+' dataset')
    print('-'*15)

    if len(augmentations)>0:

        flipProb = augParams['flipProb']
        rotateProb = augParams['rotateProb']
        colorProb = augParams['colorProb']
        hueLimits = augParams['hue']
        saturationLimits = augParams['saturation']
        contrastLimits = augParams['contrast']
        brightnessLimits = augParams['brightness']
        channelMeans = augParams['channelMeans']
        channelStd = augParams['channelStd']

        aug = Augment(hueLimits, 
                      saturationLimits, 
                      contrastLimits, 
                      brightnessLimits, 
                      rotateProb, 
                      flipProb, 
                      colorProb,
                      channelMeans,
                      channelStd)

        print('\n'*2+'Applying following Augmentations for'+datasetName+' dataset \n')
        for i, a in enumerate(augmentations):
            print('{}: {}'.format(i, a))

        columns = [c for c in list(augParams.keys())]
        values = [v for v in list(augParams.values())]
        table = PrettyTable(columns)
        table.add_row(values)
        print(table)
        print('\n')

        for f in augmentations:
            dataset = dataset.map(getattr(aug, 'get'+f), num_parallel_calls=4)
            #dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y),  num_parallel_calls=4)
    else:
        print('No data augmentation being applied')
    
    #channelMeans = augParams['channelMeans']
    #channelStd = augParams['channelStd']
    dataset = dataset.map(lambda x, y: (x, y[:,:,0:1]), num_parallel_calls=4)
    dataset = dataset.map(lambda x, y: (x/255.0, y), num_parallel_calls=4)
    dataset = dataset.map(lambda x, y: ((x-channelMeans)/channelStd,y), num_parallel_calls=4)

    if taskType=='multi':
       dataset = dataset.map(lambda x, y: (x, tf.one_hot(tf.cast(y[:,:,0], tf.int32), depth=3, dtype=tf.float32)), num_parallel_calls=4)
  
    if not test:
        dataset = dataset.cache()
        #dataset = dataset.repeat()
        dataset = dataset.shuffle(dataSize, reshuffle_each_iteration=True)
        #dataset = dataset.repeat()
        dataset = dataset.batch(batchSize, drop_remainder=True)
        dataset = dataset.prefetch(AUTO)
    else:
        dataset = dataset.batch(1)

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
