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


class TFRecordLoader():
    def __init__(self,tfrecords,name,tile_dims,task_type,batch_size):
 
        self.tfrecords=tfrecords
        self.tile_dims=tile_dims
        self.name=name
        self.task_type=task_type
        self.batch_size=batch_size
        self.dataset=None
        self.tile_nums=None
        print(self.name+' dataset')
        #print('-'*15)


    @property
    def steps(self):
        if self.tile_nums>self.batch_size:
            steps=np.floor(self.tile_nums/self.batch_size)
        else:
            steps=1
        return steps
      

    def _read_tfr_record(self, serialized):
        '''
        read tfrecord image/mask files
        :param serialized: tfrecord file
        :return image: image tensor (HxWxC)
        :return mask: mask tensor (HxWxC)
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


    def record_size(self):
        '''
        return total image count across all tfrecord files (whole dataset)
        :param tfrecords: tfrecord file paths
        :return num: int file count
        '''
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False
        dataset = tf.data.Dataset.list_files(self.tfrecords)
        dataset = dataset.with_options(option_no_order)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=4)
        dataset = dataset.map(self._read_tfr_record, num_parallel_calls=4)
        for i, d in enumerate(dataset):
            pass
        self.tile_nums=i
    

    def augment(self,methods,params):
        aug = Augment(params['hue'], 
                      params['saturation'], 
                      params['contrast'], 
                      params['brightness'], 
                      params['rotate_prob'], 
                      params['flip_prob'], 
                      params['color_prob'])
        #print('\n'*2+'Applying following Augmentations to'+self.name+' dataset \n')
        print('augmentation...')
        for i, a in enumerate(methods):
            print('{}: {}'.format(i, a))
        columns = [c for c in list(params.keys())]
        values = [v for v in list(params.values())]
        #table = PrettyTable(columns)
        #table.add_row(values)
        #print(table)
        #print('\n')
        for f in methods:
            self.dataset=self.dataset.map(getattr(aug, 'get'+f), num_parallel_calls=4)
            #dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y),  num_parallel_calls=4)
        

    def normalize(self,methods,params):
        channel_means=params['channel_mean']
        channel_std=params['channel_std']
        norm = Normalize(channel_means,channel_std)
        #print('\n'*2+'Applying following normalization methods to '+ self.name+' dataset \n')
        print('normalize...')
        for i, n in enumerate(methods):
            print('{}','{}'.format(i,n))
            self.dataset = self.dataset.map(getattr(norm, 'get'+ n), num_parallel_calls=4)
        if 'StandardizeDataset' in methods:
            columns=['means', 'std']
            values=[channel_means, channel_std]
            #table = PrettyTable(columns)
            #table.add_row(values)
            #print(table)
            #print('\n')
    

    def load(self,batch_size): 
        '''
        generate tf.record.dataset containing  image+ mask tensors with 
        transfomations/augmentations.
        tastType: string multi or binary
        :returns dataset: tfrecord.data.dataset
        '''
        self.batch_size=batch_size
        AUTO = tf.data.experimental.AUTOTUNE
        ignoreDataOrder = tf.data.Options()
        ignoreDataOrder.experimental_deterministic = False
        dataset = tf.data.Dataset.list_files(self.tfrecords)
        dataset = dataset.with_options(ignoreDataOrder)
        dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=16, num_parallel_calls=AUTO)
        dataset = dataset.map(self._read_tfr_record, num_parallel_calls=AUTO)
        #f1=tf.cast(tf.reshape(x,(self.tile_dims,self.tile_dims,3)),tf.float16)
        dataset= dataset.map(lambda x, y: (tf.cast(tf.reshape(x,(self.tile_dims,self.tile_dims,3)),tf.float16),tf.cast(tf.reshape(y,(self.tile_dims,self.tile_dims,3)),tf.float16)))
        #dataset = dataset.map(f2)
        dataset = dataset.map(lambda x, y: (x, y[:,:,0:1]), num_parallel_calls=4)
        if self.task_type=='multi':
            dataset = dataset.map(lambda x, y: (x, tf.one_hot(tf.cast(y[:,:,0], tf.int32), depth=3, dtype=tf.float32)), num_parallel_calls=4)
        #batch train and validation datasets (do not use dataset.repeat())
        #since we build our own custom training loop as opposed to model.fit
        #if model.fit used order of shuffle,cache and batch important
        if self.name!='test':
            dataset = dataset.cache()
            #dataset = dataset.repeat()
            dataset = dataset.shuffle(self.tile_nums, reshuffle_each_iteration=True)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(AUTO)
        else:
            dataset = dataset.batch(1)
        self.dataset=dataset


if  __name__ == '__main__':

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
