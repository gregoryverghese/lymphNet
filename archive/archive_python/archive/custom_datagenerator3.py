import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage import img_as_bool
from skimage.transform import resize

class DataGenerator(keras.utils.Sequence):
    def __init__(self, imgIds, maskIds, imagePath, maskPath, weights=[1,1,1,1],
                        batchSize=16, shuffle=False):
        self.imgIds = imgIds
        self.maskIds = maskIds
        self.imagePath = imagePath
        self.maskPath = maskPath
        self.weights = np.array(weights)
        self.batchSize = batchSize
        self.shuffle = shuffle

    '''
    for each image id load the patch and corresponding mask
    '''
    def __loadmasks__(self,  batchmasks):

        masks = np.empty((self.batchSize, 224, 224, 4))

        for i, m in enumerate(batchmasks):
            mask = np.load(os.path.join(self.maskPath,m))
            masks[i,] = mask
            
        
        masks = tf.multiply(masks, self.weights)
        masks = tf.cast(masks, tf.float32)
        return masks


    '''
    for each image id load the patch and corresponding mask
    '''
    def __loadimages__(self, batchimages):

        images = np.empty((self.batchSize, 224, 224, 3))

        for i, img in enumerate(batchimages):
            
            path = os.path.join(self.imagePath,img)
            image = cv2.imread(path)
            images[i,] = image
          

        images = images/255.0

        return images


    '''
    get the files for each batch (override __getitem__ method)
    '''
    def __getitem__(self, index):

        if(index+1)*self.batchSize > len(self.imgIds):
            self.batchSize = len(self.imgIds) - index*self.batchSize

        batchImgs = self.imgIds[self.batchSize*index:self.batchSize*(index+1)]
        batchMasks = self.maskIds[self.batchSize*index:self.batchSize*(index+1)]

        images =self.__loadimages__(batchImgs)
        masks = self.__loadmasks__(batchMasks)
        
        return images, masks


    '''
    Return number of steps per batch that are needed (override __len__ method)
    '''
    def __len__(self):
        return int(np.ceil(len(self.imgIds)/self.batchSize))
