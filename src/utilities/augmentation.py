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
        #if tf.random.uniform(()) > self.colorProb:
            #x = tf.image.random_saturation(x, self.saturationLimits[0],
             #                              self.saturationLimits[1])
        if tf.random.uniform(()) > self.colorProb:
            x = tf.image.random_brightness(x, self.brightnessLimits)
        #if tf.random.uniform(()) > self.colorProb:
            #x = tf.image.random_contrast(x, self.contrastLimits[0],
                                         #self.contrastLimits[1])

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



    def stainNormalize(self,x,y):
        pass



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
        

