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
import imgaug.augmenters as iaa
import staintools

__author__ = 'Gregory Verghese'
__email__ ='gregory.verghese@kcl.ac.uk'


# Ensure eager execution is enabled
#tf.config.run_functions_eagerly(True)

def is_image(I):
    """
    Is I an image.
    """
    if not isinstance(I, np.ndarray):
        print("HOLLY: not a np.ndarray")
        return False
    if not I.ndim == 3:
        print("HOLLY: not dim 3")
        return False
    return True


def is_uint8_image(I):
    """
    Is I a uint8 image.
    """
    if not is_image(I):
        print("HOLLY: is_image returned false")
        return False
    if I.dtype != np.uint8:
        print("HOLLY: dtype is not np.uint8",I.dtype)
        return False
    return True

# Function to standardize brightness and apply stain augmentation
#@tf.py_function(Tout=tf.float32)
def stain_augment_image(image):
    stain_method='macenko'
    sig1 = np.random.uniform(0.55,0.85)
    sig2 = np.random.uniform(0.55,0.85)

    image_np = image.numpy()  # Convert TensorFlow tensor to NumPy array
    #print("in stain_augment_image...image_np max:",np.max(image_np),image_np.shape) 

    # Convert from float32 to uint8 for staintools processing
    image_np = image_np.astype(np.uint8)
    #print(image_np.shape,image_np.dtype)

    # Standardize brightness
    image_standardized = staintools.LuminosityStandardizer.standardize(image_np)
    
    # Stain augment
    augmentor = staintools.StainAugmentor(method=stain_method, sigma1=sig1, sigma2=sig2, augment_background=False)
    augmentor.fit(image_standardized)
    augmented_image = augmentor.pop()

    # Convert the augmented image back to float32
    augmented_image_float16 = augmented_image.astype(np.float32)
    
    
    return tf.convert_to_tensor(augmented_image, dtype=tf.float32)  # Convert back to TensorFlow tensor



# Define a function to apply augmentation using imgaug
def hsv_augment_image(image):

    # Define an HSV color augmentation sequence
    augmentation = iaa.Sequential([
        iaa.AddToHue(),  # Adds a random value to the hue channel
        iaa.MultiplyHueAndSaturation((0.75, 1.25), per_channel=True),  # Multiplies hue and saturation
        iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.25), add=(-5, 10))  # Adjusts brightness
    ])
    
    # Convert the image to a NumPy array
    image_np = image.numpy()
    
    # Apply the augmentation
    augmented_image_np = augmentation(image=image_np)
    
    # Convert the augmented image back to a tensor
    augmented_image = tf.convert_to_tensor(augmented_image_np)
    
    return augmented_image

class Augment():
    '''
    class for applying different augmentations to tf.data.dataset
    on the fly before training
    '''

    def __init__(self, hueLimits, saturationLimits, contrastLimits, brightnessLimits,
                 rotateProb=0.5, flipProb=0.5, colorProb=0.5,stain_method='macenko'): 

        self.hueLimits = hueLimits
        self.saturationLimits = saturationLimits
        self.contrastLimits = contrastLimits
        self.brightnessLimits = brightnessLimits
        self.rotateProb = rotateProb
        self.flipProb = flipProb
        self.colorProb = colorProb
        self.stain_method = stain_method
    

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

        #print("getRotate-img type:",x.dtype)
        tf.debugging.assert_type(x, tf.uint8)
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

    def getStainAugment(self, x, y):
        #print("x type:",x.dtype,x.shape)
        
        #augmented_image = tf.py_function(stain_augment_image, [x], tf.float32)
        #augmented_image = stain_augment_image(x)

        augmented_images = tf.map_fn(
        lambda img: tf.py_function(stain_augment_image, [img], tf.float32),
            x,
            dtype=tf.float32
        )
        augmented_images.set_shape(x.shape)  # Ensure the shape information is retained
        return augmented_images, y



    def getHSVAugment(self, x, y):
        #the original approach using imgaug.augmenters as iaa
        #x = tf.py_function(hsv_augment_image, [x], tf.uint8)

        #revised version for EagerExecution

        ## SET UP SOME RANDOM FACTORS TO AUGMENT BY
        ## use the same ranges as previously used in the preprocessing augmentation
        delta = np.random.uniform(-0.1, 0.1)  # Random value for hue adjustment
        hue_factor = np.random.uniform(0.75, 1.25)
        saturation_factor = np.random.uniform(0.75, 1.25)
        brightness_factor = np.random.uniform(0.75, 1.25)
        add_value = np.random.uniform(-5/255.0, 10/255.0)  # Convert to [0, 1] range

        # Ensure image is in float32 format for transformations
        x = tf.image.convert_image_dtype(x, tf.float32)

        # Apply hue adjustment
        x = tf.image.adjust_hue(x, delta)

        # Apply hue and saturation multiplication
        # Convert the image to HSV
        hsv_image = tf.image.rgb_to_hsv(x)
        # Split the channels
        hue, saturation, value = tf.split(hsv_image, 3, axis=-1)
        # Multiply hue and saturation
        hue = tf.clip_by_value(hue * hue_factor, 0.0, 1.0)
        saturation = tf.clip_by_value(saturation * saturation_factor, 0.0, 1.0)
        # Merge the channels back
        hsv_image = tf.concat([hue, saturation, value], axis=-1)
        # Convert back to RGB
        x = tf.image.hsv_to_rgb(hsv_image)
    
        # Apply brightness multiplication and addition
        x = tf.image.adjust_brightness(x, delta=add_value)
        x =  x * brightness_factor

        # Clip the image to [0, 1] range and convert back to original dtype
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = tf.image.convert_image_dtype(x, tf.uint8)


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
        #print("in getStandardizeImage")
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
        #print("in getStandardizeDataset")
        xnew = (x - self.channelMeans)/self.channelStd
        xnew = tf.clip_by_value(xnew,-1.0, 1.0)
        xnew = (xnew+1.0)/2.0
        return xnew, y


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
        #print("in getScale")
        return x/255.0, y
        

