import numpy as np
import pandas as pd
from keras import backend as K
import tensorflow as tf

'''
def diceCoef(yTrue, yPred, smooth=1):
    print(yTrue.shape, yPred.shape)
    intersection = K.sum(yTrue * yPred, axis=-1)
    union = K.sum(yTrue, axis=-1) + K.sum(yPred, axis=-1)
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

    return dice
'''


def diceCoef(yTrue, yPred, smooth=1):
    print(K.int_shape(yTrue), K.int_shape(yPred))
    intersection = K.sum(yTrue * yPred, axis=[1, 2])
    union = K.sum(yTrue, axis=[1, 2]) + K.sum(yPred, axis=[1, 2])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice





def iouScore(yTrue, yPred, smooth=1):

    intersection = K.sum(yTrue*yPred, axis=[1, 2])
    union = K.sum(yTrue, axis=[1, 2]) + K.sum(yPred, axis=[1, 2]) - intersection
    iou = K.mean((intersection + smooth)/(union + smooth), axis=0)

    return iou
