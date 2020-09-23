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
               'label': tf.io.FixedLenFeature((), tf.string)}

    example = tf.io.parse_single_example(serialized, dataMap)
    image = tf.image.decode_png(example['image'])
    mask = tf.image.decode_png(example['mask'])
    label = example['label']

    image = tf.reshape(image, (2048, 2048, 3))
    mask = tf.reshape(mask, (2048, 2048, 3))
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


    def mergePatches(self, name, boundaries, outPath):

        m = ((boundaries[0][1] -
              boundaries[0][0])//(self.imgDims*self.magFactor)*self.imgDims*self.magFactor)

        for h in range(boundaries[1][0], boundaries[1][1], self.imgDims*self.magFactor):
            for w in range(boundaries[0][0], boundaries[0][1], self.imgDims*self.magFactor):

                predPath = os.path.join(outPath, 'patches', name +'_'+str(w)+'_'+str(h)+'_pred.png')
                patch = cv2.imread(predPath)
                patchNew = cv2.resize(patch, (self.figureSize, self.figureSize))

                if w == boundaries[0][0]:
                    image = patchNew
                else:
                    image = np.hstack((image, patchNew))

            if (w, h) == (boundaries[0][0]+m, boundaries[1][0]):
                final = image
            else:
                final = np.vstack((final, image))

        cv2.imwrite(os.path.join(outPath, name +'_pred.png'), final)
        cv2.imwrite(os.path.join('home/verghese/output/whole', self.modelName + '_' + self.currentTime + name +'_pred.png'), final)

        del final
        del image


    def predict(self, dataset, outPath, imgName):

        diceLst=[]
        iouLst=[]

        for data in dataset:

            image = tf.cast(data[0], tf.float32)
            mask = tf.cast(data[1], tf.float32)
            label = (data[2].numpy()[0]).decode('utf-8')
            figPath = os.path.join(outPath, 'figures')
            outPatchPath = os.path.join(outPath, 'patches')

            with tf.device('/cpu:0'):
                probabilities = self.model.predict(image)

            prediction = tf.cast((probabilities > 0.5), tf.float32)
            
            mask = mask.numpy().astype(np.uint8)
            mask[mask!=0]=1
            mask = tf.cast(tf.convert_to_tensor(mask), tf.float32)
            dice = diceCoef(mask, prediction)
            diceLst.append(dice.numpy())
            iou = iouScore(mask, prediction)
            iouLst.append(iou.numpy())
             
            prediction = prediction.numpy().astype(np.uint8)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].set_title(label)
            axs[0].imshow(image[0,:,:,0])
            axs[0].axis('off')
            axs[1].imshow(mask[0,:,:,0]*int(255), cmap='gray')
            axs[1].axis('off')
            axs[2].imshow(prediction[0,:,:,0]*int(255), cmap='gray')
            axs[2].axis('off')

            fig.savefig(os.path.join(figPath, label + '_pred.png'))
            cv2.imwrite(os.path.join(outPatchPath, label +'_pred.png'),prediction[0,:,:,0]*int(255))
            plt.close()

        imgscores = pd.DataFrame({'dice':diceLst, 'iou':iouLst})
        imgscores.to_csv(os.path.join(outPatchPath, '_imgscores.csv'))

        avgDice = np.mean(diceLst)
        avgIOU = np.mean(iouLst)
        summary = pd.DataFrame({'dice':[avgDice], 'iou': [avgIOU]})
        summary.to_csv(os.path.join(outPatchPath, '_summary.csv'))

        return avgDice, avgIOU



    def __call__(self, path, tfrecordDir='tfrecords_wsi', outPath='output/predictions/wsi'):
                
        outPath = os.path.join(outPath, self.currentDate)
        try:
            os.mkdir(outPath)
        except Exception as e:
            print(e)
        
        outpath = os.path.join(outPath, self.modelName + '_' + self.currentTime)
        try:
            os.mkdir(outpath)
        except Exception as e:
            print(e)
 
        records = glob.glob(os.path.join(path, self.feature, '*'))
        print('recordpath: {}'.format(os.path.join(path,self.feature)))
        with open('config/config_boundaries.json') as jsonFile:
            boundaries = dict(json.load(jsonFile))
        boundaries = boundaries[self.magnification]
        for r in records:

            imgName = os.path.basename(r)[:-10]
	
            print('Image: {}'.format(imgName))
            imgOutPath = os.path.join(outpath, imgName)

            try:
                os.mkdir(os.path.join(imgOutPath))
            except Exception as e:
                print(e)
                
            try:
                os.mkdir(os.path.join(imgOutPath, 'patches'))
                os.mkdir(os.path.join(imgOutPath,'figures'))
            except Exception as e:
                print(e)

            dataset = tf.data.TFRecordDataset(r)
            dataset = dataset.map(readTF, num_parallel_calls=1)
            dataset = dataset.batch(1)
            
            avgDice, avgIOU = self.predict(dataset, imgOutPath, imgName)
            self.mergePatches(imgName, boundaries[imgName], imgOutPath)

        return avgDice, avgIOU




