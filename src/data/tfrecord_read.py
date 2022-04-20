#!/usr/bin/env python3

import os
import random
import glob
import argparse
import math

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from utilities.augmentation import Augment, Normalize 

__author__ = 'Gregory Verghese'
__email__ ='gregory.verghese@kcl.ac.uk'


def readTFRecord(serialized, imgDims=256):
    '''
    read tfrecord file containing image
    and mask data
    Args:
        serialized: tfrecord file
    Returns:
        image: image tensor
        mask: mask tensor
    '''   
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


def readTF2(serialized):

    dataMap = {'image': tf.io.FixedLenFeature((), tf.string),
               'mask': tf.io.FixedLenFeature((), tf.string),
               'xDim': tf.io.FixedLenFeature((), tf.int64),
               'yDim': tf.io.FixedLenFeature((), tf.int64), 
               'label': tf.io.FixedLenFeature((),
               tf.string)}

    example = tf.io.parse_single_example(serialized, dataMap)
    image = tf.image.decode_png(example['image'])
    mask = tf.image.decode_png(example['mask'])
    xDim = example['xDim']
    yDim = example['yDim']
    label = example['label']

    print('xDim: {}, yDim:{}'.format(xDim, yDim))

    image = tf.reshape(image, (xDim, yDim, 3)) 
    mask = tf.reshape(mask, (xDim, yDim, 3)) 
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    return image, mask, label


def getRecordNumber(tfrecords):
    '''
    return the number of images across
    all tfrecord files (whole dataset)
    Args:
        tfrecords: tfrecord file paths
    Return:
        num: number of images in files
    '''
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=4)
    dataset = dataset.map(readTFRecord, num_parallel_calls=4)

    for i, d in enumerate(dataset):
        num=i

    return num


def getShards(tfrecords, dataSize, batchSize, imgDims, datasetName,augParams={}, augmentations=[], taskType='binary', normalize=[], normalizeParams={}):	
    '''
    return tf.record.dataset containing image mask tensor along with requested 
transfomations/augmentations. Info on https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    Args:
        tfrecords: tfrecord file paths
        dataSize: size of dataset
        batchSize: batch size
        imgDims: tensor dimensions
        augParams: dictionary of augmentation parameters
        augmentations: list of augmentation methods
        test: boolean flag for test set
        tastType: string multi or binary
    Returns:
        dataset: tfrecord.data.dataset
    '''

    AUTO = tf.data.experimental.AUTOTUNE
    ignoreDataOrder = tf.data.Options()
    ignoreDataOrder.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = dataset.with_options(ignoreDataOrder)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=16, num_parallel_calls=AUTO)

    dataset = dataset.map(readTFRecord, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda x, y: (tf.reshape(x,(imgDims, imgDims, 3)), tf.reshape(y,(imgDims, imgDims, 3))))
    dataset = dataset.map(lambda x, y: (tf.cast(x,tf.float16), tf.cast(y,tf.float16)))
    
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

        aug = Augment(hueLimits, 
                      saturationLimits, 
                      contrastLimits, 
                      brightnessLimits, 
                      rotateProb, 
                      flipProb, 
                      colorProb)

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
        print('No data augmentation being applied to data')

    if len(normalize)>0:

        channelMeans = normalizeParams['channelMeans']
        channelStd = normalizeParams['channelStd']

        norm = Normalize(channelMeans, channelStd)
        print('\n'*2+'Applying following normalization methods for '+
              datasetName+' dataset \n')
        for i, n in enumerate(normalize):
            print('{}','{}'.format(i,n))
            dataset = dataset.map(getattr(norm, 'get'+ n), num_parallel_calls=4)
        if 'StandardizeDataset' in normalize:
            columns=['means', 'std']
            values=[channelMeans, channelStd]
            table = PrettyTable(columns)
            table.add_row(values)
            print(table)
            print('\n')
    
    else:
        print('No normalization being applied to data')
    
    dataset = dataset.map(lambda x, y: (x, y[:,:,0:1]), num_parallel_calls=4)
    if taskType=='multi':
       dataset = dataset.map(lambda x, y: (x, tf.one_hot(tf.cast(y[:,:,0], tf.int32), 
                            depth=3, dtype=tf.float32)), num_parallel_calls=4)

    #batch train and validation datasets (do not use dataset.repeat())
    #since we build our own custom training loop as opposed to model.fit
    #if model.fit used order of shuffle,cache and batch important
    if datasetName!='Test':
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
