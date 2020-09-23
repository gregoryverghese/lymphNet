import numpy as np
import pandas as pd
from keras import backend as K
import tensorflow as tf



def diceCoef(yTrue, yPred, axIdx=[1,2,3], smooth=1):
    
    yPred = tf.cast(yPred, tf.float32)
    yTrue = tf.cast(yTrue, tf.float32) 
    intersection = K.sum(yTrue * yPred, axis=axIdx)
    union = K.sum(yTrue, axis=axIdx) + K.sum(yPred, axis=axIdx)
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

    return dice



def iouScore(yTrue, yPred, axIdx=[1,2,3], smooth=1):

    intersection = K.sum(yTrue*yPred, axis=axIdx)
    union = K.sum(yTrue, axis=axIdx) + K.sum(yPred, axis=axIdx) - intersection
    iou = K.mean((intersection + smooth)/(union + smooth), axis=0)

    return iou
