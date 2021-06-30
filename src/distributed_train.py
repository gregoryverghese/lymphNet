#!/usr/bin/env python3

'''
distributed_train.py: trains neural network model using custom
training module from tensorflow. Performs each forward pass of 
the network and calculates the new gradients using the loss 
function and performs backward pass with chosen optimizer
'''

import os
import datetime

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Progbar

from utilities.custom_loss_classes import WeightedBinaryCrossEntropy
from utilities.evaluation import diceCoef

__author__ = 'Gregory Verghese'
__email__ = 'gregory.verghese@kcl.ac.uk'

#import memory_saving_gradients
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed


class DistributeTrain():
    '''
    class for training neural network over a number of gpus.
    Performs each forward pass of the network and calculates the 
    new gradients using the loss function and performs backward 
    pass with chosen optimizer. Loss and dice are calculated per
    gpu (known as replica) and combined at the end.
    '''

    def __init__(self, epochs, model, optimizer, lossObject,
                 batchSize, strategy, trainSteps, testNum, 
                 imgDims, threshold, modelName, currentTime, 
                 currentDate, tasktype):

        self.epochs = epochs
        self.batchSize = batchSize
        self.strategy = strategy
        self.loss_object = lossObject
        self.optimizer = optimizer
        self.metric = diceCoef
        self.model = model
        self.trainSteps = trainSteps
        self.testNum = testNum
        self.imgDims = imgDims
        self.history = {'trainloss': [], 'trainmetric':[], 'valmetric': [],'valloss':[]}
        self.threshold = threshold
        self.modelName = modelName
        self.currentTime = currentTime
        self.currentDate = currentDate
        self.tasktype = tasktype

    def computeLoss(self, label, predictions):
        '''
        computes loss for each replica
        Args:
            label: mask tf 4d-tensor (BxHxW)
            predictions: network predictoon tf tensor (BxWxH)
        Returns:
            loss: loss per replica
        '''

        loss = self.loss_object(label, predictions)
        loss = tf.reduce_sum(loss) * (1. / (self.imgDims*self.imgDims*self.batchSize))

        return loss * (1/self.strategy.num_replicas_in_sync)


    def computeDice(self, yTrue, yPred):
        '''
        computes dice score using prediction and mask
        Args:
            yTrue: mask 4d-tensor mask (BxWxH)
            yPred: prediction 4d-tensor prediction (BxWxH)
        Returns:
            dice: dice score per replica
        '''

        #axIdx=[1,2,3] if self.tasktype=='binary' else [1,2]
        dice = tf.reduce_mean([self.metric(yTrue[:,:,:,i], yPred[:,:,:,i]) for i in range(yTrue.shape[-1])])
        #dice = self.metric(yTrue, yPred)
        dice = dice * (1 / self.strategy.num_replicas_in_sync)
        return dice


    def trainStep(self, inputs):
        '''
        perfoms one gradient update using tf.GradientTape.
        calculates loss using logits and then updates all
        trainable parameters
        Args:
            inputs: tuple of image and mask tensors
        Returns:
            loss: loss value
            dice: returns dice coefficient
        '''

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
        '''
        performs prediction using trained model
        Args:
            inputs: tuple of x, y image and mask tensors
        Returns:
            loss: loss value
            dice: returns dice coefficient
        '''

        x, y = inputs
        predictions = self.model(x, training=False)
        loss = self.loss_object(y, predictions)

        yPred = tf.cast((predictions > 0.5), tf.float32)

	#print('y',np.unique(y.numpy()))
	#print('yPred', np.unique(yPred.numpy()))

        dice = self.computeDice(y, yPred)
        loss = tf.reduce_sum(loss) * (1. / (self.imgDims*self.imgDims*self.batchSize))

        return loss, dice


    @tf.function
    def distributedTrainEpoch(self, batch):
        '''
        calculates loss and dice for each replica
        Args:
            batch: containing image and mask tensor data
        Returns:
            replicaLoss:
            replicaDice:
        '''

      #totalLoss = 0.0
      #totalDice = 0.0
      #i = 0
      #prog = Progbar(self.trainSteps-1)
      #for batch in trainData:
          #i+=1
        replicaLoss, replicaDice = self.strategy.run(self.trainStep, args=(batch,))
         # totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, replicaLoss, axis=None)
         # totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, replicaDice, axis=None)
          #prog.update(i)
      #return totalLoss, totalDice
        return replicaLoss, replicaDice

    #ToDo: shitty hack to include progbar in distributed train function. need a
    #way of converting tensor i to integer
    def getDistTrainEpoch(self, trainData):
        '''
        iterates over each batch in the data and calulates total
        loss and dice over all replicas using tf strategy.
        Args:
            trainData: contains train image and mask tensors
        Returns:
            totalLoss: total loss across all replicas
            totalDice: total dice across all dice
        '''

        #use Progbar to provide visual update of training
        totalLoss = 0.0
        totalDice = 0.0
        i = 0
        prog = Progbar(self.trainSteps-1)
        for batch in trainData:
            replicaLoss, replicaDice = self.distributedTrainEpoch(batch)
            totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, replicaLoss, axis=None)
            totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, replicaDice, axis=None)
            prog.update(i) 
            i+=1

        return totalLoss, totalDice

	
       
    @tf.function
    def distributedTestEpoch(self, validData):
        '''
        calculates loss and dice across all replicas
        for the test data
        Args:
            validData: contains validation image and mask tensors
        Returns:
            totalloss: loss summed over replicas
            totalDice dice summed over replicas
        '''

        totalLoss = 0.0
        totalDice = 0.0

        for d in validData:
            loss, dice = self.strategy.run(self.testStep, args=(d,))
            totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, dice, axis=None)

        return totalLoss, totalDice


    #we wantt o stop on a moving average value, min threshold dice and min epoch iterations 
    def earlyStop(self, valDice, epoch):
        '''
        implements and earlystopping criteria based on two dice thresholds
        and the number of epochs that has passed
        Args:
            valDice: current validation dice value
            epoch: epoch number
        Returns:
            stop: boolean
        '''

        if epoch > self.threshold['first']['epochs'] and valDice > self.threshold['first']['metric']:
            stop = True
        elif epoch > self.threshold['second']['epochs'] and valDice > self.threshold['second']['metric']:
            stop = True
        else:
            stop = False

        return stop


    def forward(self, trainDistDataset, testDistDataset):
        '''
        performs the forward pass of the network. calls each training epoch
        and prediction on validation data. Records results in history
        dictionary. Record logs for tensorboard using create_file_writer
        Args:
            trainDistDataset: contains train image and mask tensor data
            testDistDataset: contains valid image and mask tensor data
        Returns:
            self.model: trained tensorflow/keras subclassed model
            self.history: dictonary containing train and validation scores
        '''

        currentTime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        trainLogDir = os.path.join('tensorboard_logs', 'train', self.currentDate, self.modelName + '_' + self.currentTime)
        testLogDir = os.path.join('tensorboard_logs', 'test', self.currentDate, self.modelName + '_' + self.currentTime)

        trainWriter = tf.summary.create_file_writer(trainLogDir)
        testWriter = tf.summary.create_file_writer(testLogDir)

        for epoch in range(self.epochs):

            #trainLoss, trainDice = self.distributedTrainEpoch(trainDistDataset)
            trainLoss, trainDice = self.getDistTrainEpoch(trainDistDataset)

            print('trainLoss', trainLoss, 'trainDice', trainDice)
            epochTrainLoss, epochTrainDice = float(trainLoss/self.trainSteps), float(trainDice/self.trainSteps)

            with trainWriter.as_default():
                tf.summary.scalar('loss', epochTrainLoss, step=epoch)
                tf.summary.scalar('dice', epochTrainDice, step=epoch)

            tf.print(' Epoch: {}/{},  loss - {:.2f}, dice - {:.2f}, lr - {:.5f}'.format(epoch+1, self.epochs, epochTrainLoss,
                     epochTrainDice, 1), end="")

            testLoss, testDice  =  self.distributedTestEpoch(testDistDataset)
            epochTestLoss, epochTestDice = float(testLoss/self.testNum), float(testDice/self.testNum)

            with testWriter.as_default():
                tf.summary.scalar('loss', epochTestLoss, step=epoch)
                tf.summary.scalar('Dice', epochTestDice, step=epoch)

            tf.print('  val_loss - {:.3f}, val_dice - {:.3f}'.format(epochTestLoss, epochTestDice))

            self.history['trainmetric'].append(epochTrainDice)
            self.history['trainloss'].append(epochTrainLoss)
            self.history['valmetric'].append(epochTestDice)
            self.history['valloss'].append(epochTestLoss)

            if self.earlyStop(epochTestDice, epoch):
                print('Stopping early on epoch: {}'.format(epoch))
                break

        return self.model, self.history
