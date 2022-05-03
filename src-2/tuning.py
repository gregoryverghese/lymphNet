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

N=10

def tuning(args):

    '''
    generates series of config files to tune different parameters
    :param args: command line arguments
    '''
    
    date = str(datetime.date.today())
    current_time = datetime.datetime.now().strftime('%H:%M')
    results_path = os.path.join(args.save_path, 'summaries')
    experiment_name=date+'_'+current_time

    indexes = []
    results = []

    model_name = args.model_name
    config_template = args.config_file

    #open config template file and get parameters
    with open(config_template) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    losses = config['loss']
    #dropouts = jsonDict['dropouts']
    augmentation = config['augmentation']['methods'] 
    augmentation=augmentation*N
    feature = config['feature']
    mag = config['magnification']
    #Loop over parameters (augmentation and loss functions)
    for a in augmentation:
       for l in losses:
          with open(config_template) as yaml_file:
              config = yaml.load(yaml_file)

          config['loss'] = l
          config['augmentation']['methods'] = a
          name = config['experiment_name']
          name = name.replace('$model', experiment_name)
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
          args.config_file = configFile
          with open(configFile, 'w') as jsonFile:
              json.dump(jsonDict, jsonFile)
          print(args) 
          print(args.config_file)
          result = main(args,experiment_name)
          indexes.append(name)
          results.append(result)

    #results (avg dice and IOU for each analysis are saved down in a csv file
    df = pd.DataFrame({'dice': results}, index=indexes)
    df.to_csv(os.path.join(resultsPath,modelname+'_'+mag+'_'+feature+'_'+date+'_'+currentTime+'.csv'))



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--record_path', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--record_dir', required=True, help='directory for the tfrecords dataset')
    ap.add_argument('-op', '--out_path', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpoint_path', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--config_file', help='file containing parameters')
    ap.add_argument('-mn', '--model_name', help='name of neural network model') 
    args = ap.parse_args()
    
    devices  = tf.config.experimental.list_physical_devices('GPU')
    print(devices)
    tuning(args)
