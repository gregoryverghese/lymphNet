import os
import glob
import cv2
import time
import tensorflow as tf
from skimage import img_as_bool
from skimage.transform import resize
import numpy as np


class ImageMaskProcessor():
    def __init__(self, imagepath, maskpath, targetsize, weights=[1,1,1,1]):

        self.imagepath = imagepath
        self.maskpath = maskpath
        self.targetSize = targetsize
        self.weights = weights


    def getImages(self):
        i=0
        for f in glob.glob(os.path.join(self.imagepath,'*')):
            i=i+1
            img = cv2.imread(f)
            print('image',i, img.shape)
            yield img


    def getMasks(self):
        i=0
        for f in glob.glob(os.path.join(self.maskpath, '*')):
            i=i+1
            mask = np.load(f)
            mask = np.multiply(mask,self.weights)
            mask = tf.cast(mask, tf.float32)
         
            yield mask


    def load(self, n):

        images = self.getImages()
        masks = self.getMasks()

        images = [next(images) for i in range(n)]
        masks = [next(masks) for i in range(n)]

        #print('here now')

        images = np.array(images)
        start_time = time.time()
        #print('im after the images')
        masks = np.array(masks)
        elapsed_time = time.time() - start_time
        print(elapsed_time)    
        #print('im after the masks')
        return images, masks
