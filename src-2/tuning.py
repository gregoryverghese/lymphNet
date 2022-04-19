#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Tuning.py: hyperparameter tuning script

opens config template file and iterates over sets of parameters
Executes hyperparameter tuning by calling models  with different sets of parameters
'''

import os
import json
import datetime
import argparse

import tensorflow as tf
import pandas as pd

from main import main

__author__ = 'Gregory Verghese'
__email__ = 'gregory.e.verghese@kcl.ac.uk'

N=20

def tuning(args):

    '''
    generates series of config files to tune different parameters
    :param args: command line arguments
    '''
    
    date = str(datetime.date.today())
    currentTime = datetime.datetime.now().strftime('%H:%M')
    resultsPath = os.path.join(args['outpath'], 'summaries')

    indexes = []
    results = []

    modelname = args['modelname']
    configTemplate = args['configfile']

    #open config template file and get parameters
    with open(configTemplate) as jsonFile:
        jsonDict = json.load(jsonFile)

    losses = jsonDict['loss']
    #dropouts = jsonDict['dropouts']
    augmentation = jsonDict['augmentation']['methods'] 
    augmentation=augmentation*N
    feature = jsonDict['feature']
    mag = jsonDict['magnification']
    #Loop over parameters (augmentation and loss functions)
    for a in augmentation:
       for l in losses:
          with open(configTemplate) as jsonFile:
              jsonDict = json.load(jsonFile)

          jsonDict['loss'] = l
          jsonDict['augmentation']['methods'] = a
          name = jsonDict['modelname']
          name = name.replace('$model', modelname)
          name = name.replace('$loss', l)
          #name = name.replace('$drop', str(d))
          augInitials = [i[0] for i in a]
          name = name.replace('$augment', ''.join(augInitials))
          name=name.replace('$optimizer',jsonDict['optimizer']['method'])
          name=name.replace('$epoch',str(jsonDict['epoch']))
          name=name.replace('$dim',str(jsonDict['imageDims']))
          
          name=name.replace('$uptypename',str(jsonDict['upTypeName']))
          name=name.replace('$threshold',str(jsonDict['activationthreshold']))
          name=name.replace('$feature',str(jsonDict['feature']))
          name=name.replace('$magnification',str(jsonDict['magnification']))
          jsonDict['modelname'] = name
          
          #save down analysis specific config file
          configFile = os.path.join(os.path.split(configTemplate)[0], name+'.json')
          args['configfile'] = configFile
          with open(configFile, 'w') as jsonFile:
              json.dump(jsonDict, jsonFile)

          result = main(args)
          indexes.append(name)
          results.append(result)

    #results (avg dice and IOU for each analysis are saved down in a csv file
    df = pd.DataFrame({'dice': results}, index=indexes)
    df.to_csv(os.path.join(resultsPath,modelname+'_'+mag+'_'+feature+'_'+date+'_'+currentTime+'.csv'))



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--recordpath', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--recordDir', required=True, help='directory for the tfrecords dataset')
    ap.add_argument('-op', '--outpath', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpointpath', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--configfile', help='file containing parameters')
    ap.add_argument('-mn', '--modelname', help='name of neural network model')
    
    args = vars(ap.parse_args())
    
    devices  = tf.config.experimental.list_physical_devices('GPU')

    tuning(args)
