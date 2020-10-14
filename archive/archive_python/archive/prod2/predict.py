import os
import glob
import cv2
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from evaluation import diceCoef, iouScore
import datetime
import matplotlib.pyplot as plt



def readTF(serialized):

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
    def __init__(self, model, modelName, batchSize, currentTime, currentDate):
        self.model = model
        self.modelName = modelName
        self.batchSize = batchSize
        self.currentTime = currentTime
        self.currentDate = currentDate


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
            
            logits = self.testStep(image)
            prediction = tf.cast(logits > 0.5, tf.float32)

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

        imgscores = pd.DataFrame({'dice':diceLst, 'iou':iouLst})
        imgscores.to_csv(os.path.join(path, self.modelName+'_imgscores.csv'))
        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
        summary.to_csv(os.path.join(path, self.modelName+'_summary.csv'))



class WSIPredictions(object):
    def __init__(self, model, modelName, feature, magnification, imgDims,
                 currentTime, currentDate, resolutionDict={'2.5x':16,'5x':8,'10x':4}, figureSize=500):
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

    def predict(self, image, mask, label, outPath):

        with tf.device('/cpu:0'):

            try:
                probabilities = self.model.predict(image)

            prediction = tf.cast((probabilities > 0.5), tf.float32)

            mask = mask.numpy().astype(np.uint8)
            mask[mask!=0]=1
            mask = tf.cast(tf.convert_to_tensor(mask), tf.float32)
        
            dice = diceCoef(mask[:,:,:,:], prediction[:,:,:,0:1])
            iou = iouScore(mask[:,:,:,:], prediction[:,:,:,0:1])
 
            prediction = prediction.numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(outPath, label +'_pred.png'),prediction[0,:,:,0]*int(255))
            plt.close()

            retun dice, iou


    def __call__(self, path, tfrecordDir='tfrecords_wsi', outPath='output/predictions/wsi'):
                
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
	
            print('Image: {}'.format(label))
            if 'U_100188_15_B_NA_15_L1' in label:
                continue

            dice, iou = self.predict(image, mask, label, outPath)
            
            diceLst.append(dice.numpy())
            iouLst.append(iou.numpy())
            names.append(label)

        imgscores = pd.DataFrame({'image': names, 'dice':diceLst, 'iou':iouLst})
        imgscores.to_csv(os.path.join(outPath, '_imgscores.csv'))

        avgDice = np.mean(diceLst)
        avgIOU = np.mean(iouLst)

        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
        summary.to_csv(os.path.join(outPath, '_summary.csv'))
            
        return avgDice, avgIOU

