#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
ln_segmentation.py: main module that sets up analysis, models
parameters and calls training and prediction scripts
'''

import os
import sys
import csv
import glob
import pickle
import random
import argparse
import json
import datetime

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from prettytable import PrettyTable
from tensorflow import keras
from skimage.transform import resize
from skimage import img_as_bool
#SM_FRAMEWORK=tf.keras
#import segmentation_models as sm
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
#from tensorflow.keras.utils import multi_gpu_model

import distributed_train
from models import fcn8
from models import unet
from models import mobile
from models import resunet
from models import resunet_a
from models import unet_mini
from models import atten_unet
from models import multiscale
from models import multi_atten
from data import tfrecord_read
from utilities import decay_schedules
from utilities.evaluation import diceCoef
from predict import WSIPredictions, PatchPredictions
#from utilities.utilities import getTrainMetric
from preprocessing.calculate_classweights import calculateWeights
from utilities.custom_loss_classes import WeightedBinaryCrossEntropy, DiceLoss, WeightedCategoricalCrossEntropy

__author__ = 'Gregory Verghese'
__email__='gregory.verghese@kcl.ac.uk'


FUNCMODELS={
            'unet':unet.UnetFunc,
            'unetmini':unet_mini.UnetMiniFunc,
            'attention':atten_unet.AttenUnetFunc,
            'multiscale':multiscale.MultiScaleUnetFunc,
            'multiatten':multi_atten.MultiAttenFunc,
            'resunet':resunet.ResUnetFunc,
            'fcn8':unet.UnetFunc,
            'mobile':mobile.MobileUnetFunc
              }


SUBCLASSMODELS={
              'unet':unet.UnetSC,
              'unetmini':unet_mini.UnetMiniSC,
              'attention':atten_unet.AttenUnetSC,
              'multiscale':multiscale.MultiScaleUnetSC,
              'resunet':resunet.ResUnetFunc,
              'fcn8':unet.UnetFunc
               }            


def prepareDatasets(recordsPath, recordsDir,batchSize,imgDims,augParams,
                   augment,tasktype,normalize,normalizeParams):

    #load data as data.Dataset for each of train, validation and test. Counts 
    #the number of images in each record and calculate the number of steps 
    #in one training iteration for each batch using batchsize and image number.
    trainFiles = glob.glob(os.path.join(recordsPath,recordsDir,'train','*.tfrecords'))
    validFiles = glob.glob(os.path.join(recordsPath,recordsDir, 'validation','*.tfrecords'))
    testFiles = glob.glob(os.path.join(recordsPath,recordsDir, 'test','*.tfrecords'))

    trainNum = tfrecord_read.getRecordNumber(trainFiles)
    validNum = tfrecord_read.getRecordNumber(validFiles)
    testNum = tfrecord_read.getRecordNumber(testFiles)

    trainSteps = np.ceil(trainNum/batchSize) if trainNum<batchSize else np.floor(trainNum/batchSize)
    validSteps = np.ceil(validNum/batchSize) if validNum<batchSize else np.floor(validNum/batchSize)
    testSteps = np.ceil(testNum/batchSize) if testNum<batchSize else np.floor(testNum/batchSize)

    table = PrettyTable(['\nTrainNum', 'ValidNum', 'TestNum', 
                         'TrainSteps', 'ValidSteps', 'TestSteps'])
    table.add_row([trainNum, validNum, testNum, trainSteps, validSteps, testSteps])
    print(table)

    trainDataset = tfrecord_read.getShards(trainFiles, imgDims=imgDims, batchSize=batchSize, 
                                           dataSize=trainNum,augParams=augParams,augmentations=augment,
                                           taskType=tasktype,normalize=normalize,normalizeParams=normalizeParams,datasetName='Train')

    validDataset = tfrecord_read.getShards(validFiles, imgDims=imgDims, batchSize=batchSize,
                                           dataSize=validNum, taskType=tasktype, normalize=normalize,
                                           normalizeParams=normalizeParams,datasetName='Valid')

    testDataset = tfrecord_read.getShards(testFiles, batchSize=batchSize,imgDims=imgDims,
                                          dataSize=testNum, taskType=tasktype,
                                          normalize=normalize,normalizeParams=normalizeParams,datasetName='Test')

    return trainDataset,validDataset,testDataset,trainSteps,validSteps,testSteps
  

def main(args, modelname):
    '''
    sets up analysis based on config file. Choose following design choices:

    1. model
    2. optimizer and optimizer decay
    3. loss function

    Calls scripts to load data stored as tfrecords, training and prediction
    scripts. The trained model is passed to  predict on either individual 
    patches or/and whole regions of wsi. Parameters and the config
    file specific to this analysis are saved down along with the model.
    
    Args:
        args: contains command line arguments
        modelname: string containing modelname
    Returns:
        result: returns avg dice and iou score
    '''
    
    recordsPath = args['recordpath']
    recordsDir = args['recordDir']
    outPath = args['outpath']
    checkpointPath = args['checkpointpath']
    configFiles = args['configfile']

    with open(configFiles) as jsonFile:
        params = json.load(jsonFile)

    feature = params['feature']
    tasktype = params['tasktype']
    #model = params['model']['network']
    filters = params['model']['parameters']['filters']
    finalActivation = params['model']['parameters']['finalActivation']
    optimizername = params['optimizer']['method']
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
    normalize = params['normalize']['methods']
    normalizeParams= params['normalize']['parameters']
    padding = params['padding']
    multiDict = params['multi']
    multi = multiDict['flag']
    imgDims = params['imageDims']
    nClasses = params['nClasses']
    api = params['modelapi']
    magnification = params['magnification']
    stopthresholds = params['stopthresholds']
    activationthreshold =  params['activationthreshold']
    step = params['step']
    upTypeName = params['upTypeName']
    trainingapi=params['trainingapi']

    currentDate = str(datetime.date.today())
    currentTime = datetime.datetime.now().strftime('%H:%M')
    
    ############################setup paths##########################
    outModelPath = os.path.join(outPath,'models')
    os.makedirs(outModelPath,exist_ok=True)
    outModelPath = os.path.join(outPath,'models',currentDate)
    os.makedirs(outModelPath,exist_ok=True)
    outCurvePath = os.path.join(outPath,'curves')
    os.makedirs(outCurvePath,exist_ok=True)
    outCurvePath = os.path.join(outPath,'curves',currentDate)
    os.makedirs(outCurvePath,exist_ok=True)
    outPredictPath=os.path.join(outPath,'predictions')
    os.makedirs(outPredictPath,exist_ok=True)

    if weights is None:
        weights = calculateWeights('test', 'test', 'test', nClasses)

    data=prepareDatasets(recordsPath,recordsDir,batchSize,imgDims,augparams,
                         augment,tasktype,normalize,normalizeParams)
   
    trainDataset,validDataset,testDataset,trainSteps,validSteps,testSteps=data

    print('\n'*4+'-'*25 + 'Breast Cancer Lymph Node Deep learning segmentation project' +'-'*25+'\n'*2 + \
          'network:{} \
          \nfeatures:{} \
          \nmagnification:{} \
          \nloss:{} \
          \weights:{} \
          \naugmentation:{} \n'.format(modelname, feature, magnification, 
                                       loss, augment, weights))

    #get the number of gpus available and initiate a distribute mirror strategy
    devices = tf.config.experimental.list_physical_devices('GPU')
    devices = [x.name.replace('/physical_device:', '') for x in devices] 
    #devices = ['/device:GPU:{}'.format(i) for i in range(multiDict['num'])]

    nnParams={
               'filters':filters,
               'finalActivation':finalActivation,
               'nOutput':nClasses,
               'upTypeName':upTypeName
                }


    strategy = tf.distribute.MirroredStrategy(devices)
    with strategy.scope():
    #if trainingapi=='keras':
        #lossObject=getLoss(loss, weights)
        #optimizer=getOptimizer(optimizername,decaySchedule, trainSteps)

    
        #get optimizer and decay schedule if applicable
        if decaySchedule['keras'] is not None:
            boundaries = [b*trainSteps for b in decaySchedule['decayparams']['boundaries']]
            values = decaySchedule['decayparams']['values']
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries,values)
        elif decaySchedule['custom']['method'] is not None:
            scheduleKwargs = decaySchedule['custom']['decayparams']
            scheduler = getattr(decay_schedules, decaySchedule['custom']['method'])(**scheduleKwargs)
            callbacks += [LearningRateScheduler(scheduler)]
            scheduler.plot(epoch, modelName + ': ' + decaySchedule + ' learning rate')

        if optimizername=='adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        elif optimizername=='Nadam':
            optimizer = keras.optimizers.NAdam(**optKwargs)
        elif optimizername=='SGD':
            optimizer=keras.optimizers.SGD(**optKwargs)
        else:
            raise ValueError('No optimizer selected, please update config file')


        if loss=='weightedBinaryCrossEntropy':
            lossObject = WeightedBinaryCrossEntropy(weights[0])
        elif loss=='weightedCategoricalCrossEntropy':
            lossObject = WeightedCategoricalCrossEntropy(weights)
        elif loss=='diceloss':
            lossObject = DiceLoss()
        else:
            raise ValueError('No loss requested, please update config file')

        with tf.device('/cpu:0'):
            if api=='functional':
                model=FUNCMODELS[modelname](**nnParams)
            elif api=='subclass':
                model=SUBCLASSMODELS[modelname](**nnParams)

            model=model.build()
            #model = sm.Unet('vgg19', input_shape=(None, None, 3))
            if model is None:
                raise ValueError('No model requested, please update config file')
        print(model)
        table = PrettyTable(['Model', 'Loss', 'Optimizer', 'Devices','DecaySchedule'])
        table.add_row([modelname, loss, optimizername, len(devices),decaySchedule['keras']])
        print(table)
        
        if trainingapi=='custom':
            #call distributed training script to allow for training on multiple gpus
            trainDistDataset = strategy.experimental_distribute_dataset(trainDataset)
            validDistDataset = strategy.experimental_distribute_dataset(validDataset)
                                                  
            train = distributed_train.DistributeTrain(epoch, model, optimizer, 
                                                      lossObject, batchSize, strategy, 
                                                      trainSteps, validSteps, imgDims, 
                                                      stopthresholds, modelName, currentDate, 
                                                      currentTime, tasktype)

            model, history = train.forward(trainDistDataset, validDistDataset)
            
    if trainingapi=='keras':

        #model = multi_gpu_model(model, gpus=multiDict['num'])
        model.compile(optimizer=optimizer,loss=lossObject,metrics=[diceCoef])
        model.fit(trainDataset, epochs=epoch, steps_per_epoch=trainSteps, validation_data=validDataset)


    #save models down to model folder along with training history
    if api == 'subclass':
        model.save(os.path.join(outModelPath, modelName + '_' + currentTime), save_format='tf')

    elif api == 'functional': 
        model.save(os.path.join(outModelPath, modelName + '_' + currentTime + '.h5'))

    with open(os.path.join(outModelPath,  modelName+'_'+ currentTime + '_history'), 'wb') as f:
        pickle.dump(history, f)

    #call predict on individual patches and entire annotated wsi region
    #patchpredict = PatchPredictions(model, modelName, batchSize, currentTime, currentDate, activationthreshold)
    #patchpredict(testdataset, os.path.join(outPath, 'predictions'))

    outpath='/home/verghese/breastcancer_ln_deeplearning/output/predictions/wsi'
    wsipredict = WSIPredictions(model, modelName, feature,
                                magnification,step, step,
                                activationthreshold, currentTime,
                                currentDate, tasktype, normalize,
                                normalizeParams)

    result = wsipredict(os.path.join(recordsPath, 'tfrecords_wsi'), outpath)
    #ToDo: replace generic trainmetric with metric name from config file
    getTrainCurves(history,'trainloss','valloss',outCurvePath,modelName+'_'+currentTime)
    getTrainCurves(history,'trainmetric', 'valmetric', outCurvePath,modelName+'_'+currentTime)

    #finally save the config for this file with the model and predictions
    configPath=os.path.join(configPath,modelName+'_'+currentTime+'_config.json')
    with open(configPath, 'w') as jsonFile:
        json.dump(params, jsonFile)

    return result


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--recordpath', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--recordDir', required=True, help='directory for the tfrecords dataset')    
    ap.add_argument('-op', '--outpath', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpointpath', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--configfile', help='file containing parameters')
    ap.add_argument('-mn', '--modelname', help='name of neural network model')

    args = vars(ap.parse_args())
     
    recordsPath = args['recordpath']
    recordsDir = args['recordDir']
    outPath = args['outpath']
    checkpointPath = args['checkpointpath']
    configFiles = args['configfile']

    with open(configFiles) as jsonFile:
        params = json.load(jsonFile)

    modelname=args['modelname']
    main(args,modelname)
