import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage import img_as_bool
from skimage.transform import resize


class SegProcessing ():
    def __init__(self, imagepath, maskpath, outpath, weights=[1,1,1,1], imageDim=224):

        self.outpath = outpath
        self.imagepath = imagepath
        self.maskpath = maskpath
        self.weights = np.array(weights)
        self.imageDim = imageDim

    def weightMasks(self, mask):

        for i in range(len(mask)):
            for j in range(len(mask[0])):
                mask[i][j][mask[i,j,:]==1]=self.weights[mask[i,j,:]==1]
        return mask

    def maskProcessing(self, folder='masks_n'):
        print('iamhere')
        maskfiles = glob.glob(os.path.join(self.maskpath, '*'))

        for i, m in enumerate(maskfiles):
            mask = cv2.imread(m)
            mask = img_as_bool(resize(mask, (self.imageDim, self.imageDim)))
            mask = mask.astype('uint16')
            maskNew = np.empty((224, 224))
            maskNew[mask[:,:,0]==0]=0
            maskNew[mask[:,:,0]==1]=1
            maskNew[mask[:,:,1]==1]=2
            
            maskNew = tf.cast(maskNew, tf.float32)
            print(i, flush=True)
            path = os.path.join(self.outpath, folder, os.path.basename(m)[:-4])
            np.save(path, maskNew)


    def imageProcessing(self, directory='images_n'):

        imageFiles = glob.glob(os.path.join(self.imagepath, '*'))

        for i in imageFiles:
             
            image = cv2.imread(i)
            print(i, image.shape)
            image = cv2.resize(image, (224, 224))

            path = os.path.join(self.outpath, directory, os.path.basename(i))
            cv2.imwrite(path, image)


maskPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/two/masks'
imagePath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/two/images'
outPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/two'
weights = [8, 11, 6, 1] 
segPP = SegProcessing(imagePath, maskPath, outPath)

#segPP.maskProcessing()
segPP.imageProcessing()
