#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
predict.py: segmentation prediction on wsi images
'''

import os
import glob
import json

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

from utilities.evaluation import diceCoef, iouScore
from utilities.utilities import oneHotToMask


__author__ = 'Gregory Verghese'
__email__ = 'gregory.verghese@kcl.ac.uk'


def readTF(serialized):
    '''
    read tfrecord files (.tfrecords) and converts
    serialized image and mask data into 4d-tensors
    Args:
        serialized: tfrecord file
    Returns:
        image: tensorflow 4d-tensor 
        mask : tensorfow 4d-tensor groundtruth
        label: string name
    '''

    dataMap = {'image': tf.io.FixedLenFeature((), tf.string),
               'mask': tf.io.FixedLenFeature((), tf.string),
               'xDim': tf.io.FixedLenFeature((), tf.int64),
               'yDim': tf.io.FixedLenFeature((), tf.int64), 
               'label': tf.io.FixedLenFeature((), tf.string)}

    example = tf.io.parse_single_example(serialized, dataMap)
    image = tf.image.decode_png(example['image'])
    mask = tf.image.decode_png(example['mask'])
    xDim = example['xDim']
    yDim = example['yDim']
    label = example['label']

    print('xDim: {}, yDim:{}'.format(xDim, yDim))

    image = tf.reshape(image, (xDim, yDim, 3))
    mask = tf.reshape(mask, (xDim, yDim, 3))
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    return image, mask, label


class PatchPredictions(object):
    '''
    class that applies trained model to predict
    the class that each pixel belongs to. Calculates
    dice/iou score for each patch and saves down 
    average across test dataset
    '''

    def __init__(self, model, modelName, batchSize, currentTime, 
                 currentDate, threshold):

        self.model = model
        self.modelName = modelName
        self.batchSize = batchSize
        self.currentTime = currentTime
        self.currentDate = currentDate
        self.threshold = threshold


    @tf.function
    def testStep(self, x):
        logits = self.model(x, training=False)
        return logits


    def __call__(self, testdataset, outPath):
 
        path=os.path.join(outPath, 'patches', self.currentDate)

        try:
            os.mkdir(path)
        except Exception as e:
            print(e)
        
        path=os.path.join(path, self.modelName + '_' +
                          self.currentTime)
        try:
            os.mkdir(path)    
        except Exception as e:
            print(e)

        diceLst=[]
        iouLst=[]

        for i, data in enumerate(testdataset):
            image = tf.cast(data[0], tf.float32)
            mask = tf.cast(data[1], tf.float32)
            
            #predicts on 4d image and calculates class based on 
            #activation threshold
            logits = self.testStep(image)
            prediction = tf.cast(logits > self.threshold, tf.float32)

            dice = diceCoef(mask, prediction)
            diceLst.append(dice.numpy())

            iou = iouScore(mask, prediction)
            iouLst.append(iou.numpy())
            
            image = tf.cast(image, tf.uint8)
            fig, axs = plt.subplots(1, 3, figsize=(5, 5))
            axs[0].imshow(image[0,:,:,:], cmap='gray')
            axs[1].imshow(mask[0,:,:,0]*255, cmap='gray')
            axs[2].imshow(prediction[0,:,:,0]*255, cmap='gray')
            fig.savefig(os.path.join(path, str(i) + '.png'))
            plt.close()

        avgDice = np.mean(np.array(diceLst))
        avgIOU = np.mean(np.array(iouLst))
        print('Avg dice: {} \n Avg iou {}'.format(avgDice, avgIOU))
        
        #saves down dice/iou for each patch and avg across testadataset
        imgscores = pd.DataFrame({'dice':diceLst, 'iou':iouLst})
        imgscores.to_csv(os.path.join(path, self.modelName+'_imgscores.csv'))
        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
        summary.to_csv(os.path.join(path, self.modelName+'_summary.csv'))



class WSIPredictions(object):
    '''
    class that applies model to predict on large region/entire whople slide
    image
    '''
    def __init__(self, model, modelName, feature, magnification, imgDims, 
                 step, threshold, currentTime, currentDate, tasktype, 
                 channelMeans, channelStd, 
                 resolutionDict={'2.5x':16,'5x':8,'10x':4}, figureSize=500):

        self.model = model 
        self.modelName = modelName
        self.feature = feature
        self.resolutionDict = resolutionDict
        self.imgDims = imgDims
        self.figureSize = figureSize
        self.magnification = magnification
        self.magFactor = self.resolutionDict[self.magnification] 
        self.currentDate = currentDate
        self.currentTime = currentTime
        self.threshold = threshold,
        self.step = step
        self.tasktype = tasktype
        self.channelMeans = channelMeans
        self.channelStd = channelStd

    
    def predict(self, image, mask, label, outPath):
        '''
        takes a image and predicts a class for each pixel using 
        trained self.model. If the image is large than certain 
        threshold self.step we split the image into smaller regions 
        and predict on each one and then stitch back together.
        Args:
            image: 4d-tensor image
            mask: 4d-tensor groundtruth mask
            label: string name
            outPath: path to save down predictions
        Returns:
            dice: float dice score
            iou: float iou score
        '''

        _,x,y,_ = K.int_shape(image)
        if x>self.step and y>self.step:

            #split into patches based with self.step
            #predict on each patch and then stitch together
            patches=[]
            for i in range(0, x, self.step):
                for j in range(0, y, self.step):
                    patch = image[:,i:i+self.step,j:j+self.step,:]
                    patches.append(patch) 

            #predict on cpu
            probs=[]
            with tf.device('/cpu:0'):
                for i in range(len(patches)):
                    for img in patches[i]:
                        probs.append(self.model.predict(img))
            
            for p, imgs in zip(probs, patches):
                f = lambda x: tf.reshape(i,(self.step, self.step,3)
                patches = list(map(f, imgs))
                f = lambda x: tf.reshape(i,(self.step, self.step,1)
                probs = list(map(f, p)
            
            image=np.vstack([np.hstack(i) for i in patches])
            probs=np.vstack([np.hstack(p) for p in probs])

            prediction=tf.cast((probs>self.threshold), tf.float32)
            prediction = tf.expand_dims(prediction, axis=0)

        else:

            with tf.device('/cpu:0'):
                probs=self.model.predict(image)
                prediction=tf.cast((probs>self.threshold), tf.float32)

                #
                dice = [diceCoef(mask[:,:,:,i],prediction[:,:,:,i]) for i in range(mask.shape[-1])]
                iou = [iouScore(mask[:,:,:,i], prediction[:,:,:,i]) for i in range(mask.shape[-1])]
            print('dice scores', dice)            
 
        prediction = prediction.numpy().astype(np.uint8)[0,:,:,:]
        mask  = mask.numpy().astype(np.uint8)[0,:,:,:]
        
        if self.tasktype=='multi':
            prediction = oneHotToMask(prediction)

        cv2.imwrite(os.path.join(outPath, label +'_pred.png'),prediction*int(255))
        
        return np.mean(dice)

                    
    def __call__(self, path, tfrecordDir='tfrecords_wsi', outPath='/home/verghese/breastcancer_ln_deeplearning/output/predictions/wsi'):
        '''
        set up paths and iterate over dataset calling predict.
        save down results for each image in csv
        Args:
            path: string path to
            tfrecordDir: path to testing tfrecord files
            outPath: path to save prediction images down
        Returns:
            avgDice: average of test dice scores across test images
            avgIOU: average of test iou across test images
        '''

        outPath = os.path.join(outPath, self.currentDate)
        try:
            os.mkdir(outPath)
        except Exception as e:
            print(e)
        
        outPath = os.path.join(outPath, self.modelName + '_' + self.currentTime)
        try:
            os.mkdir(outPath)
        except Exception as e:
            print(e)
 
        record = glob.glob(os.path.join(path, self.feature, '*'))
        print('recordpath: {}'.format(os.path.join(path,self.feature)))
        
        diceLst = []
        iouLst = []
        names = []

        dataset = tf.data.TFRecordDataset(record[0])
        dataset = dataset.map(readTF, num_parallel_calls=1)
        dataset = dataset.batch(1)

        for data in dataset:

            image = tf.cast(data[0], tf.float32)
            mask = tf.cast(data[1], tf.float32)
            label = (data[2].numpy()[0]).decode('utf-8')


            print(np.unique(mask[:,:,0]),np.unique([:,:,1]),np.unique([:,:,2]))

            if self.tasktype=='multi':
                mask = tf.one_hot(tf.cast(mask[:,:,:,0], tf.int32), depth=3, dtype=tf.float32)
             
            #ToDO: Hack need to remove duplicate test image with different name
            if '100188_01_R' in label:
                continue

            dice = self.predict(image, mask, label, outPath)

            print('shape of the image', K.int_shape(image))
            print('Image: {}'.format(label))
            print('dice: {}'.format(dice))

            diceLst.append(dice)
            iouLst.append(dice)
            names.append(label)

        imgscores = pd.DataFrame({'image': names, 'dice':diceLst, 'iou':iouLst})
        imgscores.to_csv(outPath + '_imgscores.csv')

        avgDice = np.mean(diceLst)
        avgIOU = np.mean(iouLst)

        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
        summary.to_csv(outPath + '_summary.csv')
            
        return avgDice, avgIOU

