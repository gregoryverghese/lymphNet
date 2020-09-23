import os
import tensorflow as tf
import numpy as np
import datetime
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Progbar
from utilities.custom_loss_classes import WeightedBinaryCrossEntropy
from utilities.evaluation import diceCoef

#import memory_saving_gradients
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed


class DistributeTrain():

    def __init__(self, epochs, model, optimizer, lossObject, batchSize,
                 strategy, trainSteps, testNum, imgDims, threshold, modelName, currentTime, currentDate, tasktype):
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
		
        loss = self.loss_object(label, predictions)
        loss = tf.reduce_sum(loss) * (1. / (self.imgDims*self.imgDims*self.batchSize))

        return loss * (1/self.strategy.num_replicas_in_sync)


    def computeDice(self, yTrue, yPred):
        
        axIdx=[1,2,3] if self.tasktype=='binary' else [1,2]
        dice = self.metric(yTrue, yPred, axIdx)
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

	#print('y',np.unique(y.numpy()))
	#print('yPred', np.unique(yPred.numpy()))

        dice = self.computeDice(y, yPred)
        loss = tf.reduce_sum(loss) * (1. / (self.imgDims*self.imgDims*self.batchSize))

        return loss, dice


    @tf.function
    def distributedTrainEpoch(self, batch):

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

        totalLoss = 0.0
        totalDice = 0.0

        for d in validData:
            loss, dice = self.strategy.run(self.testStep, args=(d,))
            totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, dice, axis=None)

        return totalLoss, totalDice


    #we wantt o stop on a moving average value, min threshold dice and min epoch iterations 
    def earlyStop(self, valDice, epoch):
        
        if epoch > self.threshold['first']['epochs'] and valDice > self.threshold['first']['metric']:
            stop = True
        elif epoch > self.threshold['second']['epochs'] and valDice > self.threshold['second']['metric']:
            stop = True
        else:
            stop = False

        return stop


    def forward(self, trainDistDataset, testDistDataset):

        currentTime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        trainLogDir = os.path.join('tensorboard_logs', 'train', self.currentDate, self.modelName + '_' + self.currentTime)
        testLogDir = os.path.join('tensorboard_logs', 'test', self.currentDate, self.modelName + '_' + self.currentTime)

        trainWriter = tf.summary.create_file_writer(trainLogDir)
        testWriter = tf.summary.create_file_writer(testLogDir)

        for epoch in range(self.epochs):

            #trainLoss, trainDice = self.distributedTrainEpoch(trainDistDataset)
            trainLoss, trainDice = self.getDistTrainEpoch(trainDistDataset)
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
