import os
import os
import os
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import tensorflow.keras.backend as K
from tensorflow.keras import Model
import unetsc
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import UpSampling2D, concatenate
from evaluation import diceCoef
from custom_loss_classes import WeightedBinaryCrossEntropy, DiceLoss

tf.config.experimental_run_functions_eagerly(True)


class Train():
    def __init__(self, model, lossFunc, optimizer, strategy, epochs, batchSize):
        self.epochs = epochs
        self.batchSize = batchSize
        self.strategy = strategy
        #self.lossFunc = lossFunc
        self.lossFunc = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = optimizer
        self.model = model
        self.history = {'trainloss': [], 'trainmetric':[], 'valmetric': []}


    def computeLoss(self, yPred, yTrue):

        #loss = tf.reduce_sum(self.lossFunc(yPred, yTrue)) * (1./self.batchSize)
        loss = self.lossFunc(yPred, yTrue)
        loss = loss * (1. / self.strategy.num_replicas_in_sync)
        #print(loss)

        return loss


    #@tf.function
    def trainStep(self, x, y, i):
        #x = batch[0]
        #y = batch[1]
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32) 
        #print(self.model.trainable_variables)
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            logits = tf.cast(logits, tf.float32) 
            loss = self.computeLoss(logits, y)
            #loss = self.lossFunc(logits, y)
            print('loss', loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        print(len(gradients))
        print(len(self.model.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, logits


    #@tf.function
    def validStep(self, x, y):
        logits = self.model(x, training=False)
        loss = self.lossFunc(y, logits)

        return loss, logits,



    @tf.function
    def distributedTrainEpoch(self, dataset, trainSteps):

        totalDice = 0
        totalLoss = 0
        #prog = Progbar(trainSteps-1)

        for i, batch in enumerate(dataset):
            x = batch[0]
            y = batch[1]
            batchLoss, logits = self.strategy.run(self.trainStep, args=(x,y,i))
            pred = (logits.numpy() > 0.5).astype('int16').astype(np.float16)
            batchDice = self.strategy.run(diceCoef, args=(pred, y))
            totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, batchLoss, axis=None)
            totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, batchDice, axis=None)
            #prog.update(i)

        return totalLoss, totalDice


    @tf.function
    def distributedValidEpoch(self, dataset):
        totalLoss = 0
        totalDice = 0
        for d in dataset:
            x = d[0]
            y = tf.expand_dims(d[1], axis=-1)
            loss, logits = self.strategy.run(self.validStep, args=(x, y))
            pred = (logits.numpy() > 0.5).astype('int16').astype(np.float16)
            dice = self.strategy.run(diceCoef, args=(pred, y))
            totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, dice, axis=None)

        return totalLoss, totalDice



    def forward(self, trainDataset, validDataset, trainSteps, validSteps):
        
        @tf.function
        def distributedTrainEpoch(self, dataset, trainSteps):

            totalDice = 0
            totalLoss = 0
            #prog = Progbar(trainSteps-1)

            for i, batch in enumerate(dataset):
                x = batch[0]
                y = batch[1]
                batchLoss, logits = self.strategy.run(self.trainStep, args=(x,y,i))
                pred = (logits.numpy() > 0.5).astype('int16').astype(np.float16)
                batchDice = self.strategy.run(diceCoef, args=(pred, y))
                totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, batchLoss, axis=None)
                totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, batchDice, axis=None)
                #prog.update(i)

                return totalLoss, totalDice


        @tf.function
        def distributedValidEpoch(self, dataset):
            totalLoss = 0
            totalDice = 0
            for d in dataset:
                x = d[0]
                y = tf.expand_dims(d[1], axis=-1)
                loss, logits = self.strategy.run(self.validStep, args=(x, y))
                pred = (logits.numpy() > 0.5).astype('int16').astype(np.float16)
                dice = self.strategy.run(diceCoef, args=(pred, y))
                totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
                totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, dice, axis=None)

            return totalLoss, totalDice

        for e in range(self.epochs):

            tf.print('Epoch: {}/{}...'.format(e+1, self.epochs), end="")

            trainLoss, trainDice = self.distributedTrainEpoch(trainDataset, trainSteps)
            avgTrainDice = trainDice.numpy()[0] / trainSteps
            avgTrainLoss = trainLoss.numpy() / trainSteps
            print('train', avgTrainDice)
            print('loss', avgTrainLoss)

            tf.print(' Epoch: {}/{},  loss - {:.2f}, dice - {:.2f}'.format(e+1,
                   self.epochs, avgTrainLoss, avgTrainDice), end="")

            valLoss, valDice = self.distributedValidEpoch(validDataset)

            avgValidDice = valDice.numpy()[0] / validSteps
            avgValidLoss = valLoss.numpy() / validSteps
            
            
            self.history['trainmetric'].append(avgTrainDice)
            self.history['trainloss'].append(avgTrainLoss)
            self.history['valmetric'].append(avgValidDice)
            self.history['valmetric'].append(avgValidLoss)

            tf.print('  val_loss - {:.3f}, val_dice - {:.3f}'.format(avgValidLoss, avgValidDice))

        return self.model, history


def test(multiDict, loss, optimizer, model,trainDataset, validDataset, epoch, batchSize, optKwargs, api, filters, imgDims, trainSteps, validSteps):

    #devices = ['/device:GPU:{}'.format(i) for i in range(multiDict['num'])]
    strategy = tf.distribute.MirroredStrategy()
    
    '''
    with strategy.scope():
        
        if loss=='weightedBinaryCrossEntropy':
            print('Loss: Weighted binaryCrossEntropy')
            posWeight = float(weights[0])
            loss = WeightedBinaryCrossEntropy(posWeight)
        elif loss=='diceloss':
            print('Loss: diceloss')
            loss=DiceLoss()
        else:
            raise ValueError('No loss requested, please update config file')

        if optimizer=='adam':
            print('Optimizer: {}'.format(optimizer))
            optimizer = tf.keras.optimizers.Adam(**optKwargs)
        elif optimizer=='Nadam':
            print('Optimizer: {}'.format(optimizer))
            optimizer = tf.keras.optimizers.NAdam(**optKwargs)
        elif optimizer=='SGD':
            print('Optimizer: {}'.format(optimizer))
            optimizer=tf.keras.optimizers.SGD(**optKwargs)
        else:
            raise ValueError('No optimizer selected, please update config file')

        if model == 'fcn8':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api == 'functional':
                    fcn = FCN()
                    model = fcn.getFCN8()
                elif api=='subclass':
                    model = FCN()

        elif model == 'unet':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api=='functional':
                    unetModel = unet2.UnetFunc()
                    model = unetModel.unet()
                elif api=='subclass':
                    model = unetsc.UnetSC(filters=filters)
                    model.build((1, imgDims, imgDims, 3))

        elif model == 'unetmini':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api == 'functional':
                    unetminiModel = UnetMini(filters=filters)
                    model = unetminiModel.unetmini()
                elif api=='subclass':
                    model = UnetMini(filters)

        elif model == 'resunet':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api=='functional':
                    resunetModel =  ResUnet(filters)
                    model = resunetModel.ResUnetFunc()
                elif api=='subclass':
                    model = ResunetSc(filters)

        elif model == 'resunet-a':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api=='functional':
                    resunetModel =  ResUnetA(filters)
                    model = resunetModel.ResUnetAFunc()
                elif api=='subclass':
                    model = ResunetASc(filters)

        elif model == 'attention':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api == 'functional':
                    attenModel = AttenUnetFunc(filters)
                    model = attenModel.attenUnet()
                elif api=='subclass':
                    model = AttenUnetSC(filters)

        else:
            raise ValueError('No model requested, please update config file')
        '''

    with strategy.scope():
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        optimizer = tf.keras.optimizers.Adam(**optKwargs)
        loss = DiceLoss()
        model = unetsc.UnetSC(filters=filters)
        model.build((1, imgDims, imgDims, 3))       

        trainer = Train(model, loss, optimizer, strategy, epoch, batchSize)

        trainDistDataset = strategy.experimental_distribute_dataset(trainDataset)
        validDistDataset = strategy.experimental_distribute_dataset(validDataset)

        model, history = trainer.forward(trainDistDataset, validDistDataset, trainSteps, validSteps)
        
        return model, history
