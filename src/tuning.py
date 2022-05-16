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
import yaml

import tensorflow as tf
import pandas as pd

from main import main

__author__ = 'Gregory Verghese'
__email__ = 'gregory.e.verghese@kcl.ac.uk'

N=10

def tuning(args,config,save_path,curr_date,curr_time):

    '''
    generates series of config files to tune different parameters
    :param args: command line arguments
    '''  

    indexes = []
    results = []

    model_name = args.model_name
    losses = config['loss']
    #dropouts = config['dropouts']
    augmentation = config['augmentation']['methods'] 
    augmentation=augmentation*N
    feature = config['feature']
    mag = config['magnification']
    #Loop over parameters (augmentation and loss functions)
    #generate experiment specific config file
    for a in augmentation:
       for l in losses:
          config['loss'] = l
          config['augmentation']['methods'] = a
          #generate experiment name using 
          name = config['name']
          name = name.replace('$model', model_name)
          name=name.replace('$feature',str(config['feature']))
          name=name.replace('$mag',str(config['magnification']))
          aug_initials = [i[0] for i in a]
          name = name.replace('$augment', ''.join(aug_initials))
          name=name.replace('$dim',str(config['image_dims']))
          config['experiment_name'] = name
          name = name+curr_date+'_'+curr_time 
          #set up folders for experiment
          save_path = os.path.join(args.save_path,name)
          os.makedirs(save_path,exist_ok=True)
          model_save_path = os.path.join(save_path,'models')
          os.makedirs(model_save_path,exist_ok=True)
          curve_save_path = os.path.join(save_path,'curves')
          os.makedirs(curve_save_path,exist_ok=True)
          save_predict_path=os.path.join(save_path,'predictions')
          os.makedirs(save_predict_path,exist_ok=True)

          args.config_file = config
          #save down analysis specific config file
          #config_save_path = os.path.join(os.path.split(config_template)[0], name+'.json')
          #with open(config_save_path, 'w') as config_file:
              #json.dump(config, config_file)
          
          print(f'experiment name: {name}')
          result = main(args,config,name,save_path)
          indexes.append(name)
          results.append(result)

    #results (avg dice and IOU for each analysis are saved down in a csv file
    df = pd.DataFrame({'dice': results}, index=indexes)
    df.to_csv(os.path.join(save_path,'summary.csv'))



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-rp', '--record_path', required=True, help='path to tfrecords')
    ap.add_argument('-rd', '--record_dir', required=True, help='directory for the tfrecords dataset')
    ap.add_argument('-op', '--save_path', required=True, help='output path for predictions')
    ap.add_argument('-cp', '--checkpoint_path', required=True, help='path for checkpoint files')
    ap.add_argument('-cf', '--config_file', help='file containing parameters')
    ap.add_argument('-mn', '--model_name', help='name of neural network model') 
    ap.add_argument('-p', '--predict', help='set this flag to run the trained model on test set automatically')

    args = ap.parse_args()

    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H:%M')

    with open(args.config_file) as yaml_file:
        config=yaml.load(yaml_file, Loader=yaml.FullLoader)

    #set up paths for models, training curves and predictions
    save_path = os.path.join(args.save_path,curr_date)
    os.makedirs(save_path,exist_ok=True)

    tuning(args,config,save_path,curr_date,curr_time)
