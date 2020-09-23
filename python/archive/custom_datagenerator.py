import cv2
import os
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
    def __load__(self, imgName, maskName):

        img = cv2.imread(os.path.join(self.imagePath,imgName))
        img = img/255.0

        mask = np.load(os.path.join(self.maskPath,maskName))
        mask = np.multiply(mask, self.weights)
        mask = tf.cast(mask, tf.float32)

        return img, mask


    '''
    get the files for each batch (override __getitem__ method)
    '''
    def __getitem__(self, index):
        images=[]
        masks=[]
        if(index+1)*self.batchSize > len(self.imgIds):
            self.batchSize = len(self.imgIds) - index*self.batchSize

        batchImgs = self.imgIds[self.batchSize*index:self.batchSize*(index+1)]
        batchMasks = self.maskIds[self.batchSize*index:self.batchSize*(index+1)]
        #batchfiles = [self.__load__(imgFile, maskFile) for imgFile, maskFile in zip(batchImgs, batchMasks)]
        for imgFile, maskFile in zip(batchImgs, batchMasks):
            image, mask = self.__load__(imgFile, maskFile)
            images.append(image)
            masks.append(mask)

        #images, masks = zip(*batchfiles)

        return np.array(images), np.array(masks)


    '''
    Return number of steps per batch that are needed (override __len__ method)
    '''
    def __len__(self):
        return int(np.ceil(len(self.imgIds)/self.batchSize))
