#!/usr/bin/env python3

'''
distributed_train.py: trains neural network model using custom
training module from tensorflow. Performs each forward pass of 
the network and calculates the new gradients using the loss 
function and performs backward pass with chosen optimizer
'''

import os
import datetime
import subprocess
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Progbar

from utilities.custom_loss_classes import BinaryXEntropy
from utilities.evaluation import diceCoef
from utilities.utils import save_experiment

__author__ = 'Gregory Verghese'
__email__ = 'gregory.verghese@kcl.ac.uk'

#import memory_saving_gradients
#tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed

DEBUG = True

def get_gpu_memory_used():
    command = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
    output = subprocess.check_output(command.split())
    memory_used = [int(x) for x in output.decode().strip().split('\n')]
    return memory_used

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
                 global_batch_size,
                 epoch,
                 img_dims,
                 stop_criteria,
                 threshold, 
                 task_type,
                 train_writer,
                 test_writer,
                 save_path,
                 name,
                 config
                 ):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.strategy = strategy
        self.epochs = epoch
        self.global_batch_size = global_batch_size
        self.metric = diceCoef
        self.img_dims = img_dims

        self.history = {
            'train_loss': [], 
            'train_metric':[], 
            'val_metric':[],
            'val_loss':[] ,
            'weighted_sum':[]
        }

        self.stop_criteria = stop_criteria
        self.threshold = threshold
        self.task_type = task_type
        self.train_writer = train_writer
        self.test_writer = test_writer
        self.save_path = save_path
        self.config = config
        self.name = name

    def compute_loss(self, label, predictions):
        '''
        computes loss for each replica
        :param label: mask tf 4d-tensor (BxHxW)
        :param predictions: network predictoon tf tensor (BxWxH)
        :returns loss: loss per replica
        '''
        loss = self.criterion(label, predictions)
        loss = tf.reduce_sum(loss) * (1. /(self.img_dims*self.img_dims*self.global_batch_size))
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
        #print(inputs)
        #print(self.threshold)
        logits = self.model(x, training=False)
        
        loss = self.criterion(y, logits)
        #print(loss)
        y_pred = tf.cast((logits > self.threshold), tf.float32)
        #print(np.sum(y_pred))
        
        dice = self.compute_dice(y, y_pred)
        #print(dice)
        #need to change batch size variable (testing uses 1)
        print(tf.reduce_sum(loss))
        print(self.img_dims)
        #loss = tf.reduce_sum(loss) * (1. / (self.img_dims*self.img_dims*1))
        #should we just call self.compute_loss like we do in _train_step
        #it contains an additional line to div by the num of replicas
        loss = tf.reduce_sum(loss) * (1. /(self.img_dims*self.img_dims*self.global_batch_size))
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
            #print(get_gpu_memory_used())
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
            print("enumerate valid loader: ")
            loss, dice = self.strategy.run(self._test_step, args=(batch,))
            print(loss,dice)
            total_loss += self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            total_dice += self.strategy.reduce(tf.distribute.ReduceOp.SUM, dice, axis=None)
            print(total_loss,total_dice)
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
        first_epoch=self.stop_criteria['epochs'][0]
        first_metric=self.stop_criteria['metric'][0]
        second_epoch=self.stop_criteria['epochs'][1]
        second_metric=self.stop_criteria['metric'][1]

        if epoch > first_epoch and val_dice < first_metric:
            stop = True
        elif epoch > second_epoch and val_dice < second_metric:
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
        #HR - 15/06 - must run for at least XX epochs before we save the model
        min_num_epochs = 10
        weight_loss = 0.7
        weight_dice = 0.3
        model_save_path = os.path.join(self.save_path,'models')

        for epoch in range(self.epochs):
            print(epoch)
            #trainLoss, trainDice = self.distributedTrainEpoch(trainDistDataset)
            train_loss, train_dice = self._train()
            train_loss = float(train_loss/self.train_loader.steps)
            train_dice = float(train_dice/self.train_loader.steps)
            with self.train_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('dice', train_dice, step=epoch)
            epoch_str=' Epoch: {}/{},  loss - {:.2f}, dice - {:.2f}, lr - {:.5f}'
            tf.print(epoch_str.format(epoch+1, self.epochs, train_loss, train_dice, 1), end="")

            test_loss, test_dice  =  self._test()
            print(test_loss,test_dice)
            test_loss = float(test_loss/self.valid_loader.steps)
            test_dice = float(test_dice/self.valid_loader.steps)
            with self.test_writer.as_default():
                tf.summary.scalar('loss', test_loss, step=epoch)
                tf.summary.scalar('dice', test_dice, step=epoch)
            #epoch_str = '  val_loss - {:.3f}, val_dice - {:.3f}'
            #tf.print(epoch_str.format(test_loss, test_dice))
            
            self.history['train_metric'].append(train_dice)
            self.history['train_loss'].append(train_loss)
            self.history['val_metric'].append(test_dice)
            self.history['val_loss'].append(test_loss)
            print("finished epoch HOLLY")
            weighted_sum = (test_loss * weight_loss)+((1-test_dice)*weight_dice)
            self.history['weighted_sum'].append(weighted_sum)
           
            epoch_str = '  val_loss - {:.3f}, val_dice - {:.3f}, w_sum - {:.3f}'
            tf.print(epoch_str.format(test_loss, test_dice,weighted_sum))

            
            #HR - 15/06 - must run for at least XX epochs before we save
            if epoch >= min_num_epochs:
                #HR - 18/06 - we only add weighted sum to the history when we are passed the min epochs
                #otherwise we risk comparing to a min weighted sum that has not been saved
                #print("lowest weighted sum: ",min(self.history['weighted_sum'])

                #HR try out a weighted sum instead of just comparing to val loss
                if weighted_sum <= min(self.history['weighted_sum']):
                    #if test_loss <= min(self.history['val_loss']):
                    print("saving best model...at epoch ",epoch+1)
                    print(model_save_path)
                    save_experiment(self.model, 
                                    self.config,
                                    self.history, 
                                    self.name,
                                    model_save_path)


            #if self.early_stop(test_loss, epoch):
            #    print('Stopping early on epoch: {}'.format(epoch))
            #    break
        #save final model
        os.makedirs(os.path.join(model_save_path,'final'),exist_ok=True)
        save_experiment(self.model, 
                        self.config,
                        self.history, 
                        self.name,
                        os.path.join(model_save_path,'final'))


        return self.model, self.history
