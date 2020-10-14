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
import datetime
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
import unet
import resunet
import resunet_a
import unet_mini
import atten_unet
import distributed_train
import tfrecord_read
import decay_schedules
from evaluation import diceCoef
from predict2 import WSIPredictions, PatchPredictions
from utilities import getTrainMetric
from calculate_classweights import calculateWeights
from custom_loss_classes import WeightedBinaryCrossEntropy, DiceLoss, WeightedCategoricalCrossEntropy


def trainSegmentationModel(args):

    recordsPath = args['recordpath']
    recordsDir = args['recordDir']
    outPath = args['outpath']
    checkpointPath = args['checkpointpath']
    configFiles = args['configfile']

    with open(configFiles) as jsonFile:
        params = json.load(jsonFile)

    feature = params['feature']
    tasktype = params['tasktype']
    print('paramssssssss', params['model'])
    model = params['model']['network']
    filters = params['model']['parameters']['filters']
    finalActivation = params['model']['parameters']['finalActivation']
    optimizer = params['optimizer']['method']
    loss =  params['loss']
    #dropout = params['dropout']
    batchSize = params['batchSize']
    epoch = params['epoch']
    ratio = params['ratio']
    augment = params['augmentation']["methods"]
    augparams = params['augmentation']['parameters']
    weights = params['weights']
    loss =  params['loss']
    metric = params['metric']
    modelName = params['modelname']
    optKwargs = params['optimizer']['parameters']
    decaySchedule = params['optimizer']['decaymethod']
    normalize = params['normalize']
    padding = params['padding']
    multiDict = params['multi']
    multi = multiDict['flag']
    imgDims = params['imageDims']
    nClasses = params['nClasses']
    api = params['modelapi']
    magnification = params['magnification']
    stopthresholds = params['stopthresholds']
	
    currentDate = str(datetime.date.today())
    currentTime = datetime.datetime.now().strftime('%H:%M')

    #ToDo: set up all folders for files here
    #try:
        #os.mkdir(outPredPath)
    #except Exception as e:
        #print(e)

    outModelPath = os.path.join(outPath,'models',currentDate)
    try:
        os.mkdir(outModelPath)
    except Exception as e:
        print(e)

    outCurvePath = os.path.join(outPath,'curves',currentDate)
    try:
        os.mkdir(outCurvePath)
    except Exception as e:
        print(e)
        
    trainFiles = glob.glob(os.path.join(recordsPath,recordsDir,'train','*.tfrecords'))
    validFiles = glob.glob(os.path.join(recordsPath,recordsDir, 'validation','*.tfrecords'))
    testFiles = glob.glob(os.path.join(recordsPath,recordsDir, 'test','*.tfrecords'))

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

    trainDataset = tfrecord_read.getShards(trainFiles, imgDims=imgDims, batchSize=batchSize, dataSize=trainNum, augParams=augparams, augmentations=augment, taskType=tasktype)
    validDataset = tfrecord_read.getShards(validFiles, imgDims=imgDims,
                                           batchSize=batchSize,
                                           dataSize=validNum, taskType=tasktype)
    testdataset = tfrecord_read.getShards(testFiles,
                                          batchSize=batchSize,imgDims=imgDims,
                                          dataSize=testNum, test=True,
                                          taskType=tasktype)

    if decaySchedule != None:
       print('Decaying learning rate using {}'.format(decaySchedule))
       decay = 0
       scheduleKwargs = params['optimizer']['decayparams']
       scheduler = getattr(decay_schedules, decaySchedule)(**scheduleKwargs)
       callbacks += [LearningRateScheduler(scheduler)]
       scheduler.plot(epoch, modelName + ': ' + decaySchedule + ' learning rate')
       plt.savefig(os.path.join(outPath,modelName+'.png'))


    #if multiDict['num']==1:
    #devices = tf.config.experimental.list_physical_devices('GPU')
    #devices = [x.name.replace('/physical_device:', '') for x in devices]

    #else:
    devices = ['/device:GPU:{}'.format(i) for i in range(multiDict['num'])]
    strategy = tf.distribute.MirroredStrategy(devices)

    with strategy.scope():

        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        if loss=='weightedBinaryCrossEntropy':
            print('Loss: Weighted binaryCrossEntropy')
            posWeight = weights[0]
            lossObject = WeightedBinaryCrossEntropy(posWeight)
        elif loss=='weightedCategoricalCrossEntropy':
            lossObject = WeightedCategoricalCrossEntropy(weights)
        elif loss=='diceloss':
            print('Loss: diceloss')
            lossObject = DiceLoss()
        else:
            raise ValueError('No loss requested, please update config file')
        
        boundaries = [trainSteps*20,trainSteps*40,trainSteps*60,trainSteps*80]
        values = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        if optimizer=='adam':
            print('Optimizer: {}'.format(optimizer))
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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
                    unetModel = unet.UnetFunc()
                    model = unetModel.unet()
                elif api=='subclass':
                    model = unet.UnetSC(filters=filters, finalActivation=finalActivation)
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
                    resunetModel =  resunet.ResUnet(filters)
                    model = resunetModel.ResUnetFunc()
                elif api=='subclass':
                    model = resunet.ResUnetSC(filters)

        elif model == 'resunet-a':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api=='functional':
                    resunetModel =  resunet_a.ResUnetA(filters)
                    model = resunetModel.ResUnetAFunc()
                elif api=='subclass':
                    model = resunet_a.ResUnetASC(filters)

        elif model == 'attention':
            print('Model: {}'.format(model))
            with tf.device('/cpu:0'):
                if api == 'functional':
                    attenModel = atten_unet.AttenUnetFunc(filters)
                    model = attenModel.attenunet()
                elif api=='subclass':
                    model = atten_unet.AttenUnetSC(filters)
        else:
            raise ValueError('No model requested, please update config file')

        print('Stopping threshold: {}'.format(stopthresholds))

        trainDistDataset = strategy.experimental_distribute_dataset(trainDataset)
        validDistDataset = strategy.experimental_distribute_dataset(validDataset)

        train = distributed_train.DistributeTrain(epoch, model, optimizer,
                                                  lossObject, batchSize,
                                                  strategy, trainSteps,
                                                  validSteps, imgDims,
                                                  stopthresholds, modelName, currentDate,
                                                  currentTime)

        model, history = train.forward(trainDistDataset, validDistDataset)
    if api == 'subclass':
        model.save(os.path.join(outModelPath, modelName + '_' + currentTime), save_format='tf')
    #tf.keras.models.save_model(model, os.path.join(outModelPath, modelName + '_' + currentTime), save_format="h5")
    if api == 'functional': 
        model.save(os.path.join(outModelPath, modelName + '_' + currentTime + '.h5'))

    with open(os.path.join(outModelPath,  modelName+'_'+ currentTime + '_history'), 'wb') as f:
        pickle.dump(history, f)

    #patchpredict = PatchPredictions(model, modelName, batchSize, currentTime, currentDate)
    #patchpredict(testdataset, os.path.join(outPath, 'predictions'))
    if magnification in ['2.5x','5x', '10x']:
        wsipredict = WSIPredictions(model, modelName, feature, magnification, 2048, currentTime, currentDate)
        result = wsipredict(os.path.join(recordsPath, 'tfrecords_wsi'))

    #ToDo: replace generic trainmetric with metric name from config file
    getTrainMetric(history, 'trainloss', 'valloss', outCurvePath,
                   modelName+'_'+currentTime)
    getTrainMetric(history, 'trainmetric', 'valmetric', outCurvePath,
                   modelName+'_'+currentTime)
    
    result=0
    return result



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--recordpath', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--recordDir', required=True, help='directory for the tfrecords dataset')    
    ap.add_argument('-op', '--outpath', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpointpath', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--configfile', help='file containing parameters')

    args = vars(ap.parse_args())

    trainSegmentationModel(args)
