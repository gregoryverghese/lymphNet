import os
import glob
import cv2
import time
import tensorflow as tf
from skimage import img_as_bool
from skimage.transform import resize
import numpy as np


class ImageMaskProcessor():
    def __init__(self, n, imagepath, maskpath, targetsize, weights=[1,1,1,1]):

        self.n = n
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
            print(np.unique(mask))
            yield mask



    def __loadimages__(self):

        images = np.empty((self.n, 224, 224, 3))
        imageFiles = glob.glob(os.path.join(self.imagepath, '*'))[:self.n]

        for i, img in enumerate(imageFiles):
            path = os.path.join(self.imagepath,img)
            image = cv2.imread(path)
            images[i,] = image
            print(i)

        images = images/255.0
        print(images.shape)
        return images


    def __loadmasks__(self):

        masks = np.empty((self.n, 224, 224, 4))
        maskFiles = glob.glob(os.path.join(self.maskpath, '*'))[:self.n]

        for i, m in enumerate(maskFiles):
            mask = np.load(os.path.join(self.maskpath,m))
            masks[i,] = mask
            print(i)
        print('masks1', masks.shape)
        masks = tf.multiply(masks, self.weights)
        print('masks2', masks.shape)
        return masks


    def load(self):

        masks = self.__loadmasks__()
        images = self.__loadimages__()    
        
        return images, masks
