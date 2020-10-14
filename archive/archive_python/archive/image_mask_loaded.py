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
        self.weights = np.array(weights)


    def getImages(self):
        i=0
        for f in glob.glob(os.path.join(self.imagepath,'*')):
            i=i+1
            img = cv2.imread(f)
            img = cv2.resize(img, (224, 224))
            print(i)
            yield img


    def weightMasks(self, mask):

       
        print(self.weights)
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                mask[i][j][mask[i,j,:]==1]=self.weights[mask[i,j,:]==1]
        return mask


    def getMasks(self):

        for f in glob.glob(os.path.join(self.maskpath, '*')):
            mask = cv2.imread(f)
            mask = img_as_bool(resize(mask, (224, 224)))
            mask = np.dstack((mask, np.zeros((224,224))))
            mask = mask.astype('uint8')
            mask[:,:,3][mask[:,:,0]==0]=1
            #mask = self.weightMasks(mask)
            mask = np.multiply(mask, self.weights)
            mask = tf.cast(mask, tf.float32)
            print(np.unique(mask, return_counts=True))

            yield mask


    def load(self, n):

        images = self.getImages()
        masks = self.getMasks()

        images = [next(images) for i in range(n)]
        masks = [next(masks) for i in range(n)]

        images = np.array(images)
        start_time = time.time()
        masks = np.array(masks)
        elapsed_time = time.time() - start_time
        print(elapsed_time)
        return images, masks

