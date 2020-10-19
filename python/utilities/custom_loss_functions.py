#!usr/bin/env python3

'''
custom_loss_functions.py: contains loss functions
'''

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy

from evaluation import diceCoef


def calculateWeightedBinaryCrossEntropy(yTrue, yPred, posWeight):
    yPred = tf.clip_by_value(yPred, K.epsilon(), (1-K.epsilon()))
    logits = lossWithLogits(yPred)
    return tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=yTrue, pos_weight=posWeight)

	
def calculateWeightedCategoricalCrossEntropy(yTrue, yPred):
    yTrueMask = K.clip(yTrue, 0.0, 1.0)
    cce = categorical_crossentropy(yPred, yTrueMask)
    yTrueWeightsMaxed = K.max(yTrue, axis=-1)
    wcce = cce * yTrueWeightsMaxed
    return K.sum(wcce)


def calculateFocalLoss(yTrue, yPred, alpha, gamma):
    yPred = tf.clip_by_value(yPred, K.epsilon(), 1 - K.epsilon())
    logits = lossWithLogits(yPred)
    weightA = alpha * tf.math.pow((1 - yPred), gamma) * yTrue
    weightB = tf.math.pow((1 - alpha) * yPred, gamma) * (1 - yTrue)
    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits))*(weightA + weightB) + logits * weightB


def weightedCategoricalCrossEntropy(yTrue, yPred):
    def loss(yTrue, yPred):
        return calculateWeightedCategoricalCrossEntropy(yTrue, yPred) 
    return loss 


def diceLoss(yTrue, yPred):
    return 1.0 - diceCoef(yTrue, yPred)


def lossWithLogits(yPred):
    return tf.math.log(yPred/(1-yPred))


def focalLoss(alpha=0.25, gamma=2):
    def loss(yTrue, yPred):
        return calculateFocalLoss(yTrue, yPred, alpha, gamma)	    
    return loss


def weightedBinaryCrossEntropy(posWeight=1):
    def loss(yTrue, yPred):
        return calculateWeightedBinaryCrossEntropy(yTrue, yPred, posWeight)
    return loss


def diceBCELoss(yTrue, yPred, posWeight):
    def loss(yTrue, yPred):
        wbce = calculateWeightedBinaryCrossEntropy(yTrue, yPred, posWeight)
        normWBCE=((1/float(wbce))*wbce)
        dice = diceLoss(yTrue, yPred)
        return normWBCE + dice
    return loss


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, pos_weight, weight, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)[:,None]
        ce = self.weight * (ce*(1-y_true) + self.pos_weight*ce*(y_true))
        return ce








