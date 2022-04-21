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

from utilities.custom_loss_classes import BinaryXEntropy
from utilities.evaluation import diceCoef

__author__ = 'Gregory Verghese'
__email__ = 'gregory.verghese@kcl.ac.uk'

#import memory_saving_gradients
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed


class DistributeTrain():
    '''
    Multi-gpu training: performs each forward pass of the network 
    and performs backward pass with chosen optimizer. Loss and dice 
    are calculated per gpu (known as replica) and combined at the end.
    '''
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion, 
                 epoch, 
                 batch_size, 
                 strategy, 
                 train_steps, 
                 test_num, 
                 img_dims, 
                 threshold, 
                 model_name, 
                 name, 
                 task_type):

        self.epochs = epoch
        self.batch_size = batch_size
        self.strategy = strategy
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = diceCoef
        self.model = model
        self.train_steps = train_steps
        self.test_num = test_num
        self.img_dims = img_dims
        self.history = {'trainloss': [], 'trainmetric':[], 'valmetric': [],'valloss':[]}
        self.stop_criteria = stop_criteria
        self.threshold = threshold
        self.model_name = model_name
        self.name=name
        self.task_type = task_type


    def compute_loss(self, label, predictions):
        '''
        computes loss for each replica
        :param label: mask tf 4d-tensor (BxHxW)
        :param predictions: network predictoon tf tensor (BxWxH)
        :returns loss: loss per replica
        '''
        loss = self.loss_object(label, predictions)
        loss = tf.reduce_sum(loss) * (1. / (self.img_dims*self.img_dims*self.batch_size))
        #loss = tf.reduce_sum(loss) * (1. / (self.batchSize))
        return loss * (1/self.strategy.num_replicas_in_sync)


    def compute_dice(self, y_true, y_pred):
        '''
        computes dice score using prediction and mask
        :params y_true: mask 4d-tensor mask (BxWxH)
        :params y_pred: prediction 4d-tensor prediction (BxWxH)
        "returns dice: dice score per replica
        '''
        #axIdx=[1,2,3] if self.tasktype=='binary' else [1,2]
        dice = tf.reduce_mean([self.metric(y_true[:,:,:,i], y_pred[:,:,:,i])
                               for i in range(y_true.shape[-1])])
        #dice = self.metric(yTrue, yPred)
        dice = dice * (1 / self.strategy.num_replicas_in_sync)
        return dice


    def train_step(self, inputs):
        '''
        perfoms one gradient update using tf.GradientTape.
        calculates loss and updates all trainable parameters
        :params inputs: tuple of image and mask tensors
        :returns loss: loss value
        :returns dice: returns dice coefficient
        '''
        x, y = inputs
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.compute_loss(y, logits)
            y_pred = tf.cast((logits > self.threshold), tf.float32)
            dice = self.compute_dice(y, y_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, dice


    def test_step(self, inputs):
        '''
        performs prediction using trained model
        :params  inputs: tuple of x, y image and mask tensors
        :returns loss: loss value
        :returns dice: returns dice coefficient
        '''
        x, y = inputs
        logits = self.model(x, training=False)
        loss = self.loss_object(y, logits)
        y_pred = tf.cast((logits > self.threshold), tf.float32)
        dice = self.compute_dice(y, y_pred)
        loss = tf.reduce_sum(loss) * (1. / (self.img_dims*self.img_dims*self.batch_size))
        return loss, dice


    @tf.function
    def distributed_train_epoch(self, batch):
        '''
        calculates loss and dice for each replica
        :params batch: containing image and mask tensor data
        :returns replica_loss:
        :returns replica_dice:
        '''
      #totalLoss = 0.0
      #totalDice = 0.0
      #i = 0
      #prog = Progbar(self.trainSteps-1)
      #for batch in trainData:
          #i+=1
        replica_loss, replica_dice = self.strategy.run(self.train_step, args=(batch,))
         # totalLoss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, replicaLoss, axis=None)
         # totalDice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, replicaDice, axis=None)
        #prog.update(i)
      #return totalLoss, totalDice
        return replica_loss, replica_dice

    #ToDo: shitty hack to include progbar in distributed train function. need a
    #way of converting tensor i to integer
    def get_distributed_train_epoch(self, train_data):
        '''
        iterates over each batch in the data and calulates total
        loss and dice over all replicas using tf strategy.
        :returns train_data: contains train image and mask tensors
        :returns total_loss: total loss across all replicas
        :returns total_dice: total dice across all replicas
        '''
        total_loss = 0.0
        total_dice = 0.0
        prog = Progbar(self.trainSteps-1)
        for i, batch in enumerate(train_data):
            replica_loss, replica_dice = self.distributed_train_epoch(batch)
            total_loss += self.strategy.reduce(tf.distribute.ReduceOp.SUM,replica_loss, axis=None)
            total_dice += self.strategy.reduce(tf.distribute.ReduceOp.SUM,replica_dice, axis=None)
            prog.update(i) 
            i+=1
        return total_loss, total_dice

	
    @tf.function
    def distributed_test_epoch(self, valid_data):
        '''
        calculates loss and dice across all replicas
        for the test data
        :params valid_data: contains validation image and mask tensors
        :returns total_loss: loss summed over replicas
        :returns total_dice: dice summed over replicas
        '''
        total_loss = 0.0
        total_dice = 0.0
        for d in valid_data:
            loss, dice = self.strategy.run(self.test_step, args=(d,))
            total_loss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            total_dice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, dice, axis=None)
        return total_loss, total_dice   


    #we want to stop on a moving average value, min threshold dice and min epoch iterations 
    def early_stop(self, val_dice, epoch):
        '''
        implements and earlystopping criteria based on two dice thresholds
        and the number of epochs that has passed
        :params val_dice: current validation dice value 
        :params epoch: epoch number
        :returns stop: boolean
        '''
        if epoch > self.stop_criteria['first']['epochs']:
            if val_dice > self.stop_criteria['first']['metric']:
                stop = True
        elif epoch > self.stop_criteria['second']['epochs']:
            if val_dice > self.stop_criteria['second']['metric']:
                stop = True
        else:
            stop = False
        return stop


    def forward(self, train_dataset, test_dataset):
        '''
        performs the forward pass of the network. calls each training epoch
        and prediction on validation data. Records results in history
        dictionary. Record logs for tensorboard using create_file_writer
        :params train_dataset: contains train image and mask tensor data
        :params test_dataset: contains valid image and mask tensor data
        :returns self.model: trained tensorflow/keras subclassed model
        :returns self.history: dictonary containing train and validation scores
        '''
        train_log_dir = os.path.join('tensorboard_logs', 'train',self.model_name)
        test_log_dir = os.path.join('tensorboard_logs', 'test', self.model_name)
        train_write = tf.summary.create_file_writer(train_log_dir)
        test_write = tf.summary.create_file_writer(test_log_dir)
        for epoch in range(self.epochs):
            #trainLoss, trainDice = self.distributedTrainEpoch(trainDistDataset)
            train_loss, train_dice = self.get_dist_train_epoch(train_dataset)
            train_loss, train_dice = float(train_loss/self.train_steps),float(train_dice/self.train_steps)
            with trainWriter.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('dice', train_dice, step=epoch)
            tf.print(' Epoch: {}/{},  loss - {:.2f}, dice - {:.2f}, lr - {:.5f}'.format(epoch+1, self.epochs, train_loss, train_dice, 1), end="")

            test_loss, test_dice  =  self.distributed_test_epoch(test_dataset)
            test_loss, test_dice = float(test_loss/self.test_num),float(test_dice/self.test_num)
            with testWriter.as_default():
                tf.summary.scalar('loss', test_loss, step=epoch)
                tf.summary.scalar('Dice', test_dice, step=epoch)
            tf.print('  val_loss - {:.3f}, val_dice - {:.3f}'.format(test_loss, test_dice))
            
            self.history['trainmetric'].append(train_dice)
            self.history['trainloss'].append(train_loss)
            self.history['valmetric'].append(test_dice)
            self.history['valloss'].append(test_loss)

            if self.early_stop(test_dice, epoch):
                print('Stopping early on epoch: {}'.format(epoch))
                break

        return self.model, self.history
