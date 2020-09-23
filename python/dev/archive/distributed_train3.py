import tensorflow as tf
import unetsc
import numpy as np
from custom_loss_classes import WeightedBinaryCrossEntropy
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Progbar

def diceCoef(yTrue, yPred, smooth=1):

    yPred = tf.cast(yPred, tf.float32)
    yTrue = tf.cast(yTrue, tf.float32)
    intersection = K.sum(yTrue * yPred, axis=[1, 2, 3])
    union = K.sum(yTrue, axis=[1, 2, 3]) + K.sum(yPred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    
    return dice


class Train(object):

    def __init__(self, epochs, model, batchSize, strategy, trainSteps, testSteps, imgDims):
        self.epochs = epochs
        self.batchSize = batchSize
        self.strategy = strategy
        self.loss_object = WeightedBinaryCrossEntropy(pos_weight=7, reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam()
        self.metric = diceCoef
        self.model = model
        self.trainSteps = trainSteps
        self.testSteps = testSteps
        self.imgDims = imgDims
        self.history = {'trainloss': [], 'trainmetric':[], 'valmetric': [], 'valLoss':[]}


    def computeLoss(self, label, predictions):

        loss = self.loss_object(label, predictions)
        loss = tf.reduce_sum(loss) * (1. / (self.imgDims*self.imgDims*self.batchSize))

        return loss * (1/self.strategy.num_replicas_in_sync)


    def computeDice(self, yTrue, yPred):

        dice = self.metric(yTrue, yPred)
        dice = dice * (1 / self.strategy.num_replicas_in_sync)

        return dice


    def trainStep(self, inputs):

        x, y = inputs

        with tf.GradientTape() as tape:

            logits = self.model(x, training=True)
            loss = self.computeLoss(y, logits)

            yPred = tf.cast((logits > 0.5), tf.float32)
            dice = self.computeDice(y, yPred)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, dice


    def testStep(self, inputs):

        x, y = inputs
        predictions = self.model(x, training=False)
        loss = self.loss_object(y, predictions)

        yPred = tf.cast((predictions > 0.5), tf.float32)
        dice = self.computeDice(y, yPred)
        loss = tf.reduce_sum(loss) * (1. / (self.imgDims*self.imgDims*self.batchSize))

        return loss, dice


    def distributedTrainEpoch(self, trainData):

      totalLoss = 0.0
      totalDice = 0.0
      prog = Progbar(self.trainSteps-1)

      for i, batch in enumerate(trainData):
          replicaLoss, replicaDice = self.strategy.run(self.trainStep, args=(batch,))
          totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, replicaLoss, axis=None)
          totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, replicaDice, axis=None)
          prog.update(i)

      return totalLoss, totalDice


    def distributedTestEpoch(self, validData):

        totalLoss = 0.0
        totalDice = 0.0

        for d in validData:
            loss, dice = self.strategy.run(self.testStep, args=(d,))
            totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, dice, axis=None)

        return totalLoss, totalDice


    def forward(self, trainDistDataset, testDistDataset, strategy):

        #distributedTrainEpoch = tf.function(self.distributedTrainEpoch)
        #distributedTestEpoch = tf.function(self.distributedTestEpoch)

        for epoch in range(self.epochs):

            trainLoss, trainDice = self.distributedTrainEpoch(trainDistDataset)
            epochTrainLoss, epochTrainDice = float(trainLoss/self.trainSteps), float(trainDice/self.trainSteps)

            tf.print(' Epoch: {}/{},  loss - {:.2f}, dice - {:.2f}'.format(epoch+1, self.epochs, epochTrainLoss, epochTrainDice), end="")

            testLoss, testDice  =  self.distributedTestEpoch(testDistDataset)
            epochTestLoss, epochTestDice = float(testLoss/self.testSteps), float(testDice/self.testSteps)

            tf.print('  val_loss - {:.3f}, val_dice - {:.3f}'.format(epochTestLoss, epochTestDice))

            self.history['trainmetric'].append(epochTrainDice)
            self.history['trainloss'].append(epochTrainLoss)
            self.history['valmetric'].append(epochTestDice)
            self.history['valmetric'].append(epochTestLoss)

        return self.model, self.history



def distributeTrain(epochs, batchSize, imgDims, filters, trainDataset, validDataset, trainSteps, validSteps, numGpu):

  devices = ['/device:GPU:{}'.format(i) for i in range(numGpu)]
  strategy = tf.distribute.MirroredStrategy(devices)

  with strategy.scope():

      print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
      model = unetsc.UnetSC(filters=filters)
      model.build((1, imgDims, imgDims, 3))
      trainer = Train(epochs, model, batchSize, strategy, trainSteps, validSteps, imgDims)

      trainDistDataset = strategy.experimental_distribute_dataset(trainDataset)
      validDistDataset = strategy.experimental_distribute_dataset(validDataset)

      model, history = trainer.forward(trainDistDataset, validDistDataset, strategy)
