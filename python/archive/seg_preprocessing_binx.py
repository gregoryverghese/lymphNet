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

    def maskProcessing(self, folder='masks_gn'):

        maskfiles = glob.glob(os.path.join(self.maskpath, '*'))

        for m in maskfiles:
            print(m)
            mask = cv2.imread(m)
            mask = img_as_bool(resize(mask, (self.imageDim, self.imageDim)))
            mask = mask.astype('float32')
             
            path = os.path.join(self.outpath, folder, os.path.basename(m))
            print(path)
            cv2.imwrite(path, mask)


    def imageProcessing(self, directory='images_gn'):

        imageFiles = glob.glob(os.path.join(self.imagepath, '*'))
        print('now I am here') 
        for i in imageFiles:
            print(i)
            image = cv2.imread(i)
            image = cv2.resize(image, (224, 224))

            path = os.path.join(self.outpath, directory, os.path.basename(i))
            cv2.imwrite(path, image)

print('letes go')
maskPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/masks_g/GERMINAL'
imagePath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/images_g/GERMINAL'
outPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation'
 
segPP = SegProcessing(imagePath, maskPath, outPath)

#segPP.maskProcessing()
segPP.imageProcessing()
