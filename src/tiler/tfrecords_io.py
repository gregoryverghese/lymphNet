import os
import sys
import json
import glob
import argparse
import math

import cv2
import numpy as np
import tensorflow as tf


class TFRecordWrite():
    def __init__(
        self,
        db_path,
        parser,
        shard_size=0.01,
        unit=10**9):

        self.db_path = db_path
        self.parser = parser
        self.shard_size = 0.01 
        self.unit = 10**9

    
    def _print_progress(self,i):
        complete = float(i)/self.img_num_per_shard
        print(f'\r- Progress: {complete:.1%}', end='\r')

    
    @staticmethod
    def _wrap_int64(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    @staticmethod
    def _wrap_bytes(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    
    @property
    def mem_size(self):
        mem = sum(sys.getsizeof(p.tobytes()) 
                        for p,_ in self.parser.extract_tiles())
        return mem/self.unit


    @property
    def shard_number(self):
        return int(np.ceil(self.mem_size/self.shard_size))


    @property
    def img_num_per_shard(self):
        return int(np.floor(len(self.parser._tiles)/self.shard_number))


    def convert(self): 

        for i in range(self.shard_number):
            path=os.path.join(
                self.db_path,self.parser.name'_'+str(i)+'.tfrecords')
            writer=tf.io.TFRecordWriter(path)
            for j in range(self.img_num_per_shard):
                t, tile = next(self.parser.extract_tiles())
                m, mask = next(self.parser.extract_masks())
                self._print_progress(j)
                tile = tf.image.encode_png(tile) 
                mask = tf.image.encode_png(mask)
                name = self.name+'_'+str(t[0])+'_'+str(t[1])

                data = {'tile': self._wrap_bytes(tile),
                        'mask': self._wrap_bytes(mask),
                        'name': self._wrap_bytes(name.encode('utf8')),
                        'dims': self._wrap_int64(self.tile.size[0])}
               
                features = tf.train.Features(feature=data)
                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                writer.write(serialized)

