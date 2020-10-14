import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from custom_loss_functions import weightedBinaryCrossEntropy
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import UpSampling2D, concatenate
#tf.config.experimental_run_functions_eagerly(True)


'''
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=K.epsilon(), **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth

    def call(self, yPred, yTrue):


        yPred = K.flatten(yPred)
        yTrue = K.flatten(yTrue)

        intersection = K.sum(yPred * yTrue)
        union = K.sum(yPred) + K.sum(yTrue)

        dice = (2*intersection + self.smooth) / (union + self.smooth)

        return 1 - dice
'''


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=K.epsilon(), **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth

    def call(self, yPred, yTrue):

        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)
        
        intersection = K.sum(yTrue * yPred, axis=[1, 2])
        union = K.sum(yTrue, axis=[1, 2]) + K.sum(yPred, axis=[1, 2])
        dice = K.mean((2. * intersection + K.epsilon())/(union + K.epsilon()), axis=0)

        return 1 - dice


'''
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, pos_weight, weight=1, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
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

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, posWeight, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction)
        self.posWeight = posWeight

    def call(self, yTrue, yPred):
                
        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)

        tf.math.log(yPred/(1-yPred))
        yPred = tf.clip_by_value(yPred, K.epsilon(), 1-K.epsilon())
        logits = tf.math.log(yPred/(1-yPred))
        return tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=yTrue, pos_weight=self.posWeight)

'''
#@tf.function
def diceCoef(yPred, yTrue):

    yPred = tf.cast(yPred, tf.float32)
    yTrue = tf.cast(yTrue, tf.float32)

    yPred = K.flatten(yPred)
    yTrue = K.flatten(yTrue)

    intersection = K.sum(yPred * yTrue)
    union = K.sum(yPred) + K.sum(yTrue)

    dice = (2*intersection + K.epsilon()) / (union + K.epsilon())

    return dice
'''



@tf.function
def diceCoef(yTrue, yPred):
    
    yPred = tf.cast(yPred, tf.float32)
    yTrue = tf.cast(yTrue, tf.float32)

    intersection = K.sum(yTrue * yPred, axis=[1, 2])
    union = K.sum(yTrue, axis=[1, 2]) + K.sum(yPred, axis=[1, 2])
    dice = K.mean((2. * intersection + K.epsilon())/(union + K.epsilon()), axis=0)
    return dice




@tf.function
def trainStep(x, y, model, optimizer, lossFunc):

      
    with tf.GradientTape() as tape:
        
        logits = model(x, training=True)
        loss = lossFunc(y, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, logits


@tf.function
def validStep(x, y, model, lossFunc):
    valLogits = model(x, training=False)
    loss = lossFunc(y, valLogits)

    return valLogits, loss


def forward(model, optimizer, lossFunc, epochs, trainDataset, validDataset, trainSteps):
    

    for e in range(epochs):

        diceLst = []
        lossLst = []
        valDiceLst = []
        valLossLst = []
       
        history = {'trainloss': [], 'trainmetric':[], 'valmetric': []}

        tf.print('Epoch: {}/{}...'.format(e+1, epochs), end="")
        prog = Progbar(trainSteps-1)

        for i, data in enumerate(trainDataset):
            x = data[0]
            y = data[1]

            loss, logits  = trainStep(x, y, model, optimizer, lossFunc)


            print('lossa is gere', loss)


            pred = (logits.numpy() > 0.5).astype('int16').astype(np.float16)
            lossLst.append(loss)
            dice = diceCoef(y, pred)
            diceLst.append(dice)

            prog.update(i)

        avgDice = np.mean(np.array(diceLst))
        avgLoss = np.mean(np.array(lossLst))
        history['trainmetric'].append(avgDice)
        history['trainloss'].append(avgLoss)

        tf.print(' Epoch: {}/{},  loss - {:.2f}, dice - {:.2f}'.format(e+1, epochs, avgLoss, avgDice), end="")

        for data in validDataset:

            x = data[0]
            y = data[1]

            valLogits, valLoss  = validStep(x, y, model, lossFunc)

            pred = (valLogits.numpy() > 0.5).astype('int16').astype(np.float16)

            valDice = diceCoef(y, pred)
            valDiceLst.append(valDice)
            valLossLst.append(valLoss)

        avgDice = np.mean(np.array(valDiceLst))
        avgLoss = np.mean(np.array(valLossLst))
        history['valmetric'].append(avgDice)

        tf.print('  val_loss - {:.3f}, val_dice - {:.3f}'.format(avgLoss, avgDice))

        if avgDice > 0.75:
            return model, history

    return model, history
