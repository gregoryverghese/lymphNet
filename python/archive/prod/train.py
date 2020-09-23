import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import UpSampling2D, concatenate



class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=K.epsilon(), **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth

    def call(self, yPred, yTrue):

        yPred = tf.cast(yPred, tf.float32)
        yTrue = tf.cast(yTrue, tf.float32)

        yPred = K.flatten(yPred)
        yTrue = K.flatten(yTrue)

        intersection = K.sum(yPred * yTrue)
        union = K.sum(yPred) + K.sum(yTrue)

        dice = (2*intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


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


@tf.function
def trainStep(x, y, model, optimizer):

    lossFunc = DiceLoss()
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = lossFunc(y, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, logits


@tf.function
def validStep(x, y, model):
    valLogits = model(x, training=False)
    return valLogits


def forward(model, optimizer, epochs, trainDataset, validDataset, trainSteps):

    for e in range(epochs):

        diceLst = []
        lossLst = []
        valDiceLst = []
        history = {'trainloss': [], 'trainmetric':[], 'valmetric': []}

        tf.print('Epoch: {}/{}...'.format(e+1, epochs), end="")
        prog = Progbar(trainSteps)

        for i, data in enumerate(trainDataset):
            x = data[0]
            y = data[1]

            loss, logits = trainStep(x, y, model, optimizer)
            pred = (logits.numpy() > 0.5).astype('int16').astype(np.float16)

            lossLst.append(loss)
            dice = diceCoef(y, pred)
            diceLst.append(dice)

            prog.update(i)

        avgDice = np.mean(np.array(diceLst))
        avgLoss = np.mean(np.array(lossLst))
        history['trainmetric'].append(avgDice)
        history['trainloss'].append(avgLoss)

        tf.print(' avg loss - {:.2f}, avg dice - {:.2f}'.format(avgLoss, avgDice), end="")

        for data in validDataset:

            x = data[0]
            y = data[1]

            valLogits = validStep(x, y, model)

            pred = (valLogits.numpy() > 0.5).astype('int16').astype(np.float16)

            valDice = diceCoef(y, pred)
            valDiceLst.append(valDice)

        avgDice = np.mean(np.array(valDiceLst))
        history['valmetric'].append(avgDice)

        tf.print('  val_dice - {:.2f}'.format(avgDice))

    return model
