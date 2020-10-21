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

__author__ = 'Gregory Verghese'
__email__ ='gregory.verghese@kcl.ac.uk'


class Augment():
    '''
    class for applying different augmentations to tf.data.dataset
    on the fly before training
    '''

    def __init__(self, hueLimits, saturationLimits, contrastLimits, brightnessLimits,
                 rotateProb=0.5, flipProb=0.5, colorProb=0.5): 

        self.hueLimits = hueLimits
        self.saturationLimits = saturationLimits
        self.contrastLimits = contrastLimits
        self.brightnessLimits = brightnessLimits
        self.rotateProb = rotateProb
        self.flipProb = flipProb
        self.colorProb = colorProb
    

    def getRotate90(self, x, y):
        '''
        Randomly apply 90 degree rotation
        Args:
            x: image tensor
            y: mask tensor
        Returns:
            x: transformed image tensor
            y : transformed mask tensor
        '''

        rand = tf.random.uniform(shape=[],minval=0,maxval=4,dtype=tf.int32)
        x= tf.image.rot90(x, rand)
        y= tf.image.rot90(y, rand)

        return x, y


    def getRotate(self, x, y):
        '''
        Randomly apply random degree rotation
        Args:
            x: image tensor
            y: mask tensor
        Returns:
            x: transformed image tensor
            y : transformed mask tensor
        '''
        if tf.random.uniform(())>self.rotateProb:
            degree = tf.random.normal([])*360
            x = tfa.image.rotate(x, degree * math.pi / 180, interpolation='BILINEAR')
            y = tfa.image.rotate(y, degree * math.pi / 180, interpolation='BILINEAR')

        return x, y


    def getFlip(self, x, y): 
        '''
        Randomly applies horizontal and
        vertical flips
        Args:
            x: image tensor
            y: mask tensor
        Returns:
            x: transformed image tensor
            y : transformed mask tensor
        '''

        if tf.random.uniform(())> self.flipProb:
            x=tf.image.flip_left_right(x)
            y=tf.image.flip_left_right(y)

        if tf.random.uniform(())> self.flipProb:
            x=tf.image.flip_up_down(x)
            y=tf.image.flip_up_down(y)

        return x, y


    def getColor(self, x, y):

        '''
        Randomly transforms either hue, saturation
        brightness and contrast
        Args:
            x: image tensor
            y: mask tensor
        Returns:
            x: transformed image tensor
            y : transformed mask tensor
        '''
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
        '''
        Randomly crops tensor
        Args:
            x: image tensor
            y: mask tensor
        Returns:
            x: transformed image tensor
            y : transformed mask tensor
        '''
        rand = tf.random.uniform((), minval=0.6, maxval=1)
        x = tf.image.central_crop(x, central_fraction=rand)
        y = tf.image.central_crop(y, central_fraction=rand)
        return x, y


class Normalize():
    '''
    class to normalize tensor pixel values
    '''
    def __init__(self, channelMeans, channelStd):
        self.channelMeans = channelMeans
        self.channelStd = channelStd


    def getStandardizeImage(self, x, y):
        '''
        applies image level standardization
        Args:
            x: image tensor
            y: mask tensor
        Returns:
            x: normalized image tensor
            y: normalized mask tensor
        '''
        x = tf.image.per_image_standardization(x)
        return x, y


    def getStandardizeDataset(self, x, y):
        '''
        applies dataset level standardization
        to each individual image
        Args:
            x: image tensor
            y: mask tensor
        Returns:
            x: transformed image tensor
            y : transformed mask tensor
        '''
        xnew = (x - self.channelMeans)/self.channelStd
        xnew = tf.clip_by_value(xnew,-1.0, 1.0)
        xnew = (xnew+1.0)/2.0
        return x, y


    def getScale(self, x, y):
        '''
        Scale image data between 0-1
        Args:
            x: image tensor
            y: mask tensor
        Returns:
            x: transformed image tensor
            y : transformed mask tensor
        '''
        return x/255.0, y
        



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


def getShards(tfrecords, dataSize, batchSize, imgDims, augParams={}, augmentations=[], 
              test=False, taskType='binary', normalize=[], normalizeParams={}):	
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
        for n in normalize:
            dataset = dataset.map(getattr(norm, 'get'+ n), num_parallel_calls=4)

        columns=['means', 'std']
        values=[channelMeans, channelStd]
        table = PrettyTable(columns)
        table.add_row(values)
        print(table)
        print('\n')
    
        for n in normalize:
            dataset = dataset.map(getattr(norm, 'get'+f), num_parallel_calls=4)
        else:
            print('No normalization being applied to data')
    
    dataset = dataset.map(lambda x, y: (x, y[:,:,0:1]), num_parallel_calls=4)
    if taskType=='multi':
       dataset = dataset.map(lambda x, y: (x, tf.one_hot(tf.cast(y[:,:,0], tf.int32), 
                            depth=3, dtype=tf.float32)), num_parallel_calls=4)

    #batch train and validation datasets (do not use dataset.repeat())
    #since we build our own custom training loop as opposed to model.fit
    #if model.fit used order of shuffle,cache and batch important
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
