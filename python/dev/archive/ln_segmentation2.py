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

import fcn8
import unet2
import unetsc
import resunet
import unet_mini
import atten_unet
import train as train
import tfrecord_read
import decay_schedules
from evaluation import diceCoef
from predict import getPrediction
from predict_wsi import getWSIPredictions
from utilities import saveModel, saveHistory, getTrainMetric
from calculate_classweights import calculateWeights
from custom_loss_classes import WeightedBinaryCrossEntropy, DiceLoss


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
    multi = multiDict['flag']
    imgDims = params['imageDims']
    nClasses = int(params['nClasses'])
    final = params['final']
    filters = params['filters']
    api = params['api']

    trainFiles = glob.glob(os.path.join(recordsPath,'train','*.tfrecords'))
    validFiles = glob.glob(os.path.join(recordsPath,'validation','*.tfrecords'))
    testFiles = glob.glob(os.path.join(recordsPath,'test','*.tfrecords'))

    trainNum = tfrecord_read.getRecordNumber(trainFiles)
    validNum = tfrecord_read.getRecordNumber(validFiles)
    testNum = tfrecord_read.getRecordNumber(testFiles)

    trainSteps = np.ceil(trainNum/batchSize) if trainNum<batchSize else np.floor(trainNum/batchSize)
    validSteps = np.ceil(validNum/batchSize) if validNum<batchSize else np.floor(validNum/batchSize)
    testSteps = np.ceil(testNum/batchSize) if testNum<batchSize else np.floor(testNum/batchSize)

    if weights is None:
        weights = calculateWeights('test', 'test', 'test', nClasses)

    print('TrainNum: {}'.format(trainNum))
    print('ValidNum: {}'.format(validNum))
    print('TestNum: {}'.format(testNum))
    print('TrainSteps: {}'.format(trainSteps))
    print('TestSteps: {}'.format(testSteps))
    print('ValidSteps: {}'.format(validSteps))
    print('ValidNum: {}'.format(validNum))
    print('TestNum: {}'.format(testNum))
    print('Weights: {}'.format(weights))

    trainDataset = tfrecord_read.getShards(trainFiles, imgDims=imgDims, batchSize=batchSize, dataSize=trainNum,  augmentations=augment)
    validDataset = tfrecord_read.getShards(validFiles, imgDims=imgDims, batchSize=batchSize, dataSize=validNum)
    testdataset = tfrecord_read.getShards(testFiles, batchSize=batchSize,imgDims=imgDims,  dataSize=testNum, test=True)

    '''
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
        raise ValueError('No optimizer selected, please update config file')
    '''

    if decaySchedule != None:
       print('Decaying learning rate using {}'.format(decaySchedule))
       decay = 0
       scheduleKwargs = params['optimizer']['decayparams']
       scheduler = getattr(decay_schedules, decaySchedule)(**scheduleKwargs)
       callbacks += [LearningRateScheduler(scheduler)]
       scheduler.plot(epoch, modelName + ': ' + decaySchedule + ' learning rate')
       plt.savefig(os.path.join(outPath,modelName+'.png'))

    '''
    if loss=='weightedBinaryCrossEntropy':
        print('Loss: Weighted binaryCrossEntropy')
        posWeight = float(weights[0])
        loss = WeightedBinaryCrossEntropy(posWeight)
    elif loss=='diceloss':
        print('Loss: diceloss')
        loss=DiceLoss()
    else:
        raise ValueError('No loss requested, please update config file')
    '''
    #tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='tensorboard/logs')
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(checkpointPath,modelName), monitor=metric, period=10)
    #callbacks = [checkpoint]

    #devices = ['/device:GPU:{}'.format(i) for i in range(multiDict['num'])]
    #strategy = tf.distribute.MirroredStrategy(devices)

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
        optimizer = keras.optimizers.Adam(**optKwargs)
    elif optimizer=='Nadam':
        print('Optimizer: {}'.format(optimizer))
        optimizer = keras.optimizers.NAdam(**optKwargs)
    elif optimizer=='SGD':
        print('Optimizer: {}'.format(optimizer))
        optimizer=keras.optimizers.SGD(**optKwargs)
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


    #trainer = train.forward(model, loss, optimizer, strategy, epoch, batchSize)
    model, history = train.forward(model, optimizer, loss, epoch, trainDataset, validDataset, trainSteps)

    #trainDistDataset = strategy.experimental_distribute_dataset(trainDataset)
    #validDistDataset = strategy.experimental_distribute_dataset(validDataset)

    #train.main(epoch, batchSize, imgDims, filters, trainDataset, validDataset, 2)

    #model, history = train.test(multiDict, loss, optimizer, model,trainDataset, validDataset, epoch, batchSize, optKwargs, api, filters, imgDims, trainSteps, validSteps)
    #model.save(os.path.join('/home/verghese/models', modelName), save_format='tf')
    #   saveHistory(history, modelName+'_hist')

    result = getPrediction(model, trainDataset, testSteps, modelName, batchSize, outPath)
    getWSIPredictions(model, modelName)

    #getTrainMetric(history, 'loss', 'valloss', outPath, modelName)
    #getTrainMetric(history, metric, 'valmetric', outPath, modelName)

    return result




if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--recordpath', required=True, help='path to tfrecords')
    ap.add_argument('-op', '--outpath', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpointpath', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--configfile', help='file containing parameters')

    args = vars(ap.parse_args())

    trainSegmentationModel(args)
