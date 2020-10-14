import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import img_as_bool
from skimage.transform import resize


class DataGenerator(keras.utils.Sequence):
    def __init__(self, imgIds, maskIds, imagePath, maskPath, weights=[1,1,1,1],
                        batchSize=16, imageSize = (224, 224, 3), nClasses=4, shuffle=False):
        self.imgIds = imgIds
        self.maskIds = maskIds
        self.imagePath = imagePath
        self.maskPath = maskPath
        self.weights = np.array(weights)
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.nClasses = nClasses
        self.shuffle = shuffle


    def weightMasks(self, mask):

        for i in range(len(mask)):
            for j in range(len(mask[0])):
                mask[i][j][mask[i,j,:]==1]=self.weights[mask[i,j,:]==1]
        return mask


    '''
    for each image id load the patch and corresponding mask
    '''
    def __load__(self, imgName, maskName):

        img = cv2.imread(os.path.join(self.imagePath,imgName))
        img = cv2.resize(img, (self.imageSize[0], self.imageSize[1]))
        img = img/255.0

        mask = cv2.imread(os.path.join(self.maskPath,maskName))
        mask = img_as_bool(resize(mask, (self.imageSize[0], self.imageSize[1])))
        mask = np.dstack((mask, np.zeros((224, 224))))
        mask = mask.astype('uint8')
        mask[:,:,3][mask[:,:,0]==0]=1
        mask = self.weightMasks(mask)
     
        mask = tf.cast(mask, tf.float32)

        return (img, mask)


    '''
    get the files for each batch (override __getitem__ method)
    '''
    def __getitem__(self, index):

        if(index+1)*self.batchSize > len(self.imgIds):
            self.batchSize = len(self.imgIds) - index*self.batchSize

        batchImgs = self.imgIds[self.batchSize*index:self.batchSize*(index+1)]
        batchMasks = self.maskIds[self.batchSize*index:self.batchSize*(index+1)]
        batchfiles = [self.__load__(imgFile, maskFile) for imgFile, maskFile in zip(batchImgs, batchMasks)]
        images, masks = zip(*batchfiles)

        return np.array(list(images)), np.array(list(masks))


    '''
    Return number of steps per batch that are needed (override __len__ method)
    '''
    def __len__(self):
        return int(np.ceil(len(self.imgIds)/self.batchSize))
