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


class DistributedTraining():
    '''
    Multi-gpu training: performs each forward pass of the network 
    and performs backward pass with chosen optimizer. Loss and dice 
    are calculated per gpu (known as replica) and combined at the end.
    '''
    def __init__(self, 
                 model, 
                 train_loader,
                 valid_loader,
                 optimizer, 
                 criterion,
                 strategy,
                 batch_size,
                 epoch,
                 img_dims,
                 stop_criteria,
                 threshold, 
                 model_name, 
                 task_type):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.strategy = strategy
        self.epochs = epoch
        self.batch_size = batch_size
        self.strategy = strategy
        self.metric = diceCoef
        self.img_dims = img_dims
        self.history = {'train_loss': [], 'train_metric':[], 'val_metric':[],'val_loss':[]}
        self.stop_criteria = stop_criteria
        self.threshold = threshold
        self.model_name = model_name
        self.task_type = task_type


    def compute_loss(self, label, predictions):
        '''
        computes loss for each replica
        :param label: mask tf 4d-tensor (BxHxW)
        :param predictions: network predictoon tf tensor (BxWxH)
        :returns loss: loss per replica
        '''
        loss = self.criterion(label, predictions)
        loss = tf.reduce_sum(loss) * (1. / (self.img_dims*self.img_dims*self.batch_size))
        #loss = tf.reduce_sum(loss) * (1. / (self.batchSize))
        loss = loss * (1/self.strategy.num_replicas_in_sync)
        return loss


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
        dice = dice * (1 / self.strategy.num_replicas_in_sync)
        return dice


    def _train_step(self, inputs):
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


    def _test_step(self, inputs):
        '''
        performs prediction using trained model
        :params  inputs: tuple of x, y image and mask tensors
        :returns loss: loss value
        :returns dice: returns dice coefficient
        '''
        x, y = inputs
        logits = self.model(x, training=False)
        loss = self.criterion(y, logits)
        y_pred = tf.cast((logits > self.threshold), tf.float32)
        dice = self.compute_dice(y, y_pred)
        loss = tf.reduce_sum(loss) * (1. / (self.img_dims*self.img_dims*self.batch_size))
        return loss, dice


    #ToDo: shitty hack to include progbar in distributed train function. need a
    #way of converting tensor i to integer
    @tf.function
    def _run(self,batch):
        replica_loss, replica_dice = self.strategy.run(self._train_step,args=(batch,))
        return replica_loss, replica_dice


    def _train(self):
        '''
        iterates over each batch in the data and calulates total
        loss and dice over all replicas using tf strategy.
        :returns train_data: contains train image and mask tensors
        :returns total_loss: total loss across all replicas
        :returns total_dice: total dice across all replicas
        '''
        total_loss = 0.0
        total_dice = 0.0
        prog = Progbar(self.train_loader.steps-1)
        for i, batch in enumerate(self.train_loader.dataset):
            replica_loss, replica_dice = self._run(batch)
            total_loss += self.strategy.reduce(tf.distribute.ReduceOp.SUM,replica_loss, axis=None)
            total_dice += self.strategy.reduce(tf.distribute.ReduceOp.SUM,replica_dice, axis=None)
            prog.update(i) 
        return total_loss, total_dice

	
    @tf.function
    def _test(self):
        '''
        calculates loss and dice across all replicas
        for the test data
        :params valid_data: contains validation image and mask tensors
        :returns total_loss: loss summed over replicas
        :returns total_dice: dice summed over replicas
        '''
        total_loss = 0.0
        total_dice = 0.0
        for batch in self.valid_loader.dataset:
            loss, dice = self.strategy.run(self._test_step, args=(batch,))
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
        first_epoch=self.stop_criteria['first']['epochs']
        first_metric=self.stop_criteria['first']['metric']
        second_epoch=self.stop_criteria['second']['epochs']
        second_metric=self.stop_criteria['second']['metric']

        if epoch > first_epoch and val_dice > first_metric:
            stop = True
        elif epoch > second_epoch and val_dice > second_metric:
            stop = True
        else:
            stop = False
        return stop


    def forward(self):
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
        train_writer = tf.summary.create_file_writer(train_log_dir)
        test_writer = tf.summary.create_file_writer(test_log_dir)
        for epoch in range(self.epochs):
            #trainLoss, trainDice = self.distributedTrainEpoch(trainDistDataset)
            train_loss, train_dice = self._train()
            print(train_loss,self.train_loader.steps)
            train_loss = float(train_loss/self.train_loader.steps)
            train_dice = float(train_dice/self.train_loader.steps)
            print(train_loss)
            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('dice', train_dice, step=epoch)
            tf.print(' Epoch: {}/{},  loss - {:.2f}, dice - {:.2f}, lr - {:.5f}'.format(epoch+1, self.epochs, train_loss, train_dice, 1), end="")

            test_loss, test_dice  =  self._test()
            test_loss = float(test_loss/self.valid_loader.steps)
            test_dice = float(test_dice/self.valid_loader.steps)
            with test_writer.as_default():
                tf.summary.scalar('loss', test_loss, step=epoch)
                tf.summary.scalar('dice', test_dice, step=epoch)
            tf.print('  val_loss - {:.3f}, val_dice - {:.3f}'.format(test_loss, test_dice))
            
            self.history['train_metric'].append(train_dice)
            self.history['train_loss'].append(train_loss)
            self.history['val_metric'].append(test_dice)
            self.history['val_loss'].append(test_loss)

            if self.early_stop(test_dice, epoch):
                print('Stopping early on epoch: {}'.format(epoch))
                break

        return self.model, self.history
