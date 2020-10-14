#!/usr/bin/env python3

import os
import glob
import argparse

import cv2
import numpy as np


def calculateStdMean(path):

    images = glob.glob(os.path.join(path,'*'))
    imageShape = cv2.imread(images[0]).shape
    channelNum = imageShape[-1]
    
    channelValues = np.zeros((channelNum))
    channelValuesSq = np.zeros((channelNum))

    pixelNum = len(images)*imageShape[0]*imageShape[1]
    print('total number pixels: {}'.format(pixelNum))

    for imgPath in images:
        image = cv2.imread(imgPath)
        image = (image/255.0).astype('float64')
        channelValues += np.sum(image, axis=(0,1), dtype='float64')

    mean=channelValues/pixelNum

    for imgPath in images:
        image = cv2.imread(imgPath)
        image = (image/255.0).astype('float64')
        channelValuesSq += np.sum(np.square(image-mean), axis=(0,1), dtype='float64')

    std=np.sqrt(channelValuesSq/pixelNum, dtype='float64')
    
    print('mean: {}, std: {}'.format(mean, std))

    return mean, std


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True, help='path to image set')

    args = vars(ap.parse_args())

    mean, std = calculateStdMean(args['path'])

    










