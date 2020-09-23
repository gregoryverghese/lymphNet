import os
import sys
import csv
import cv2
import glob
import numpy as np
import pickle
import random
import argparse
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow import keras
from skimage.transform import resize
from skimage import img_as_bool
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import multi_gpu_model

import tfrecord_read
import decay_schedules
import train
from unet_sc import Unet
from resunet_multi import ResUnet
#from unet import Unet
from fcn8 import FCN
from unet_mini import UnetMini
from utilities import saveModel, saveHistory, getTrainMetric
from evaluation import diceCoef
from custom_loss_functions import weightedCategoricalCrossEntropy, weightedBinaryCrossEntropy, focalLoss, diceLoss, diceBCELoss, WeightedBinaryCrossEntropy
from prediction import getPrediction
from calculate_classweights import calculateWeights
from predict_wsi import getWSIPredictions


def trainSegmentationModel(args):

    recordsPath = args['recordpath']
    outPath = args['outpath']
    checkpointPath = args['checkpointpath']
    configFiles = args['configfile']

    with open(configFiles) as jsonFile:
        params = json.load(jsonFile)
  
    model = params['model']
    optimizer = params['optimizer']['method']
    loss =  params['loss']
    dropout = float(params['dropout'])
    batchSize = int(params['batchSize'])
    epoch = int(params['epoch'])
    ratio = float(params['ratio'])
    augment = params['augmentation']
    weights = params['weights']
    loss =  params['loss']
    metric = params['metric']
    modelName = params['modelname']
    optKwargs = params['optimizer']['parameters']
    decaySchedule = params['optimizer']['decaymethod']
    normalize = bool(params['normalize'])
    padding = params['padding']
    multiDict = params['multi']
    imgDims = params['imageDims']
    nClasses = int(params['nClasses'])
    final = params['final']
    filters = params['filters']

    trainFiles = glob.glob(os.path.join(recordsPath,'train','*.tfrecords'))
    validFiles = glob.glob(os.path.join(recordsPath,'validation','*.tfrecords'))
    testFiles = glob.glob(os.path.join(recordsPath,'test','*.tfrecords'))
    
    trainNum = tfrecord_read.getRecordNumber(trainFiles)
    validNum = tfrecord_read.getRecordNumber(validFiles)
    testNum = tfrecord_read.getRecordNumber(testFiles)
    
    trainSteps = np.floor(trainNum/batchSize)

    if validNum<batchSize:
        validSteps=np.ceil(validNum/batchSize)
    else:
        validSteps=np.floor(validNum/batchSize)

    if testNum<batchSize:
        testSteps=np.ceil(testNum/batchSize)
    else:
        testSteps = np.floor(testNum/batchSize)
   
    print('trainNum: {}'.format(trainNum))
    print('validNum: {}'.format(validNum))
    print('testNum: {}'.format(testNum))
    print('trainSteps: {}'.format(trainSteps))
    print('testSteps: {}'.format(testSteps))
    print('validSteps: {}'.format(validSteps)) 
    print('validNum', validNum)
    print('testNum', testNum)

    #tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='tensorboard/logs')
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(checkpointPath,modelName), monitor=metric, period=10)
    #callbacks = [checkpoint]    
    '''
    print('Tensorflow API: {}'.format(api))    
    
    if api = "functional":
        models = functionalModels
        print('Model: {}'.format(model))       
        with tf.device('/cpu:0'):
            model = models[model] 

    elif api = "subclass":
        print('Model: {}'.format(model))
        models = subclassModels
        with tf.device('/cpu:0'):
            model = models[model]
          
    subclassParams = {"filters":filters,'dtype':dtype,"nOutput":nClasses, "finalActivation":final}
    functionalParams = {"filters":filters, "finalActivation":final, "nOutput":nClasses}
    '''

    if model == 'resunet':
        print('Model: {}'.format(model))
        with tf.device
        resunetModel =  ResUnet(int(params['imageDims']), nOutput = int(params['nClasses']), finalActivation=params['final'], dropout=dropout)
        with tf.device('/cpu:0'):
            model = resunetModel.ResUnet()
    elif model == 'unet':
        print('Model: {}'.format(model))
        with tf.device('/cpu:0'):
            model = Unet(filters, 'float32', nOutput = nClasses, finalActivation=final)
            #model = unetModel.unet()    
    elif model == 'fcn8':
        print('Model: {}'.format(model))
        with tf.device('/cpu:0'):
            fcn = FCN(int(params['imageDims']), nClasses = int(params['nClasses']), finalActivation=params['final'])
        model = fcn.getFCN8()
    elif model == 'unetmini':
        print('Model: {}'.format(model))
        with tf.device('/cpu:0'):
            model = UnetMini(nOutput=nClasses, finalActivation=final)
    elif model == 'attention':
        print('Model: {}'.format(model))
        with tf.device('/cpu:0'):
            model = AttenUnet(filters, nOutput=nClasses, finalActivation=final)           
    else:
        print('No model requested, please update config file')
        sys.exit()
    
    multi = multiDict['flag']
    if multi!=False:
        gpuNum = multiDict['num']
        print('Number of gpus: {}'.format(gpuNum))
        model = multi_gpu_model(model, gpus=gpuNum)

    if weights is None:
        weights = calculateWeights('test', 'test', 'test', params['nClasses'])
    
    if loss=='binaryCrossEntropy':
        print('Loss: Binary Crossentropy')
        loss = binary_crossentropy

    elif loss=='weightedBinaryCrossEntropy':
        print('Loss: Weighted binaryCrossEntropy')
        posWeight = float(weights[0])
        loss = weightedBinaryCrossEntropy(posWeight)
    
    elif loss=='weightedCategoricalCrossEntropy':
        print('Loss: Weighted categoricalCrossEntropy')
        loss==weightedCategoricalCrossEntropy()

    elif loss=='diceloss':
        print('Loss: diceloss')
        loss=diceLoss

    elif loss=='focalloss':
        print('Loss: focalloss')
        loss=focalLoss()
    elif loss=='diceBCELoss':
        print('Loss: dice and weighted binaryCrossEntropy combined')
        loss = diceBCELoss(weights[0])
    else:
        print('No loss requested, please update config file')
        sys.exit()

    if optimizer=='adam':
        print('Optimizer: {}'.format(optimizer))
        optimizer = keras.optimizers.Adam(**optKwargs)
    elif optimizer=='Nadam':
        print('Optimizer: {}'.format(optimizer))
        optimizer = keras.optimizers.NAdam(**optKwargs)
    elif optimizer=='SGD':
        print('Optimizer: {}'.format(optimizer))
        optimizer=keras.optimizers.SGD(**optKwargs)
    else:
        print('No optimizer selected, please update config file')
        sys.exit()        
    print(decaySchedule)
    if decaySchedule != "None":
       print('Decaying learning rate using {}'.format(decaySchedule))
       decay=0
       scheduleKwargs = params['optimizer']['decayparams']
       scheduler = getattr(decay_schedules, decaySchedule)(**scheduleKwargs)
       callbacks += [LearningRateScheduler(scheduler)]
       scheduler.plot(epoch, modelName + ': ' + decaySchedule + ' learning rate')
       plt.savefig(os.path.join(outPath,modelName+'.png'))  
    
    print('valid steps', validSteps)    
    #loss = WeightedBinaryCrossEntropy(pos_weight=float(weights[0]), weight = 1.0, from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=[diceCoef])

    trainDataset = tfrecord_read.getShards(trainFiles, imgDims=imgDims, batchSize=batchSize, dataSize=trainNum,  augmentations=augment)
    validDataset = tfrecord_read.getShards(validFiles, imgDims=imgDims, batchSize=batchSize, dataSize=validNum)
    model, history = train.forward(model, optimizer, epoch, trainDataset, validDataset, trainSteps)
    
    model.save(os.path.join('/home/verghese/models', modelName, save_format='tf')
    saveHistory(history, modelName+'_hist')
    
    testdataset = tfrecord_read.getShards(testFiles, batchSize=batchSize,imgDims=imgDims,  dataSize=testNum, test=True)
    result = getPrediction(model, testdataset, testSteps, modelName, batchSize, outPath)
    getWSIPredictions(model, modelName)
    
    #getTrainMetric(history, 'loss', 'val_loss', outPath, modelName)
    #getTrainMetric(history, metric, 'val_'+ metric, outPath, modelName)

    return result    

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--recordpath', required=True, help='path to tfrecords')
    ap.add_argument('-op', '--outpath', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpointpath', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--configfile', help='file containing parameters')

    args = vars(ap.parse_args())

    trainSegmentationModel(args)
