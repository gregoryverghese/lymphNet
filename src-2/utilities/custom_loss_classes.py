#!/usr/bin/env python3

'''
custom_classes.py: script contains classes inheriting from tf.keras.losses.loss
'''

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=K.epsilon(), reduction=tf.keras.losses.Reduction.NONE):
        super(DiceLoss, self).__init__(reduction=reduction)
        self.smooth = smooth

    def call(self, yTrue, yPred):
        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)
        intersection = K.sum(yTrue * yPred, axis=[1, 2, 3])
        union = K.sum(yTrue, axis=[1, 2, 3]) + K.sum(yPred, axis=[1, 2, 3])
        dice = K.mean((2. * intersection + K.epsilon())/(union + K.epsilon()), axis=0)
        return 1 - dice


class BinaryXEntropy(tf.keras.losses.Loss):
    def __init__(self, posWeight, reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
        self.posWeight = posWeight

    def call(self, yTrue, yPred):
        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)
        tf.math.log(yPred/(1-yPred))
        yPred = tf.clip_by_value(yPred, K.epsilon(), 1-K.epsilon())
        logits = tf.math.log(yPred/(1-yPred))
        return tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=yTrue, pos_weight=self.posWeight)


class CategoricalXEntropy(tf.keras.losses.Loss):
    def __init__(self, weights, reduction=tf.keras.losses.Reduction.NONE):
        super().__init__(reduction=reduction)
        self.weights = weights

    def call(self, yTrue, yPred):
        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)
        pixelXEntropies = yTrue * (tf.math.log(yPred))
        weightXEntropies = self.weights * pixelXEntropies
        return -tf.reduce_sum(weightXEntropies, axis=-1)



def get_criterion()
    if x=='cross_entropy':
        criterion=BinaryXEntropy
    elif x=='dice_loss':
        criterion=BinaryXEntropy
    return criterion


































'''
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, pos_weight, weight=1, from_logits=False,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = 1
        self.from_logits = from_logits

    def call(self, y_true, y_pred):

        print('shapessss', K.int_shape(y_pred))

        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)[:,None]
        ce = self.weight * (ce*(1-y_true) + self.pos_weight*ce*(y_true))
        return ce
    '''
