#!/usr/bin/env python3
#-*- coding: utf-8 -*-

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

    print(K.int_shape(image))

    return image, mask, label


class PatchPredictions(object):
    '''
    class that applies trained model to predict
    the class that each pixel belongs to. Calculates
    dice/iou score for each patch and saves down 
    average across test dataset
    '''

    def __init__(self, model, modelName, batchSize, currentTime, currentDate, threshold):
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
    def __init__(self, model, modelName, feature, magnification, imgDims, step,
                 threshold, currentTime, currentDate, tasktype, channelMeans, channelStd, resolutionDict={'2.5x':16,'5x':8,'10x':4}, figureSize=500):
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
        takes a image and predicts a class for each pixel using trained 
        self.model. If the image is large than certain threshold
        self.step we split the image into smaller regions and predict 
        on each one and then stitch back together.
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
            #split the image into several patches based on self.step
            #predict on each patch and then stitch back together
            imagepatches = [[image[:,i:i+self.step,j:j+self.step,:] 
                             for j in range(0, y, self.step)] for i in range(0, x, self.step)]

            #force prediction onto cpu for memory reasons
            with tf.device('/cpu:0'):            
                probabilities = [[self.model.predict(img) for img in imagepatches[i]] 
                            for i in range(len(imagepatches))]

            imagepatches = [list(map(lambda i: tf.reshape(i,(self.step,self.step,3)),imgs)) for imgs in imagepatches]
            probabilities = [list(map(lambda i:tf.reshape(i,(self.step,self.step,1)),probs)) for probs in probabilities]

            image=np.vstack([np.hstack(i) for i in imagepatches])
            probabilities=np.vstack([np.hstack(i) for i in probabilities])

            prediction=tf.cast((probabilities>self.threshold), tf.float32)
            prediction = tf.expand_dims(prediction, axis=0)

        else:

            with tf.device('/cpu:0'):
                probs=self.model.predict(image)
                prediction=tf.cast((probs>self.threshold), tf.float32)
            
        with tf.device('/cpu:0'):
            axIdx=[1,2,3] if self.tasktype=='binary' else [1,2]
            dice = diceCoef(mask, prediction[:,:,:,0:1], axIdx)
            iou = iouScore(mask, prediction[:,:,:,0:1], axIdx)

        prediction = prediction.numpy().astype(np.uint8)[0,:,:,:]
        mask  = mask.numpy().astype(np.uint8)[0,:,:,:]
        
        if self.tasktype=='multi':
            prediction = onehotToMask(prediction)
        elif self.tasktype=='binary':
            prediction = prediction*int(255)

        cv2.imwrite(os.path.join(outPath, label +'_pred.png'),prediction)
        #cv2.imwrite(os.path.join(outPath, label +'_mask.png'),mask)
        plt.close()

        return dice, iou

                    
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
            image = image/255.0
            image = (image-self.channelMeans)/self.channelStd
            mask = tf.cast(data[1], tf.float32)
            label = (data[2].numpy()[0]).decode('utf-8')
            
            if '100188_01_R' in label:
                continue

            print('shape of the image', K.int_shape(image))
            print('Image: {}'.format(label))

            dice, iou = self.predict(image, mask, label, outPath)
            
            diceLst.append(dice.numpy())
            iouLst.append(iou.numpy())
            names.append(label)

        imgscores = pd.DataFrame({'image': names, 'dice':diceLst, 'iou':iouLst})
        imgscores.to_csv(outPath + '_imgscores.csv')

        avgDice = np.mean(diceLst)
        avgIOU = np.mean(iouLst)

        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
        summary.to_csv(outPath + '_summary.csv')
            
        return avgDice, avgIOU

