#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Calculate_classweights.py: returns weights for each class
by calculating proportion of pixels for each class in the mask.

Author: Gregory Verghese
Email: gregory.verghese@kcl.ac.uk

'''

import os
import glob
import argparse

import cv2
import numpy as np
from sklearn.utils import class_weight

def calculate_weights(
    mask_path: str, 
    out_path: str, 
    file_name: str, 
    num_classes: int
) -> None:
    """Calculate and save class weights based on pixel proportions in mask images.

    Args:
        mask_path (str): Path to the directory containing mask images.
        out_path (str): Path to the directory where the weights will be saved.
        file_name (str): Name of the file to save the weights.
        num_classes (int): Number of classes in the mask images.

    Returns:
        None
    """

    total = {c:0 for c in range(num_classes)}
    #masks = glob.glob(os.path.join(mask_path,'*/masks/*'))
    masks = glob.glob(os.path.join(mask_path,'*.png'))    
    #print(mask_path)
    #print(masks)
    print(len(masks))
    for i, f in enumerate(masks):
        print(f)
        mask = cv2.imread(f)
        labels = mask.reshape(-1)
        classes = np.unique(labels, return_counts=True)
        pixel_dict = dict(list(zip(*classes))) 
        for k, v in pixel_dict.items():
            total[k] = total[k] + v

    print(total)
    if num_classes==2:
        weight = total[0]/total[1]
    else:
        weight = [1/v for v in list(total.values())]
    #np.savetxt(os.path.join('~',file_name+'.csv'), average_weights, delimiter=',')
    print(weight)

if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-mp', '--maskpath', required=True, help='path to mask files')
    ap.add_argument('-op', '--outpath', required=True, help='path to save weights')
    ap.add_argument('-sn', '--savename', required=True, help='filename for saving')
    ap.add_argument('-nc', '--number', required=True, help='number of classes')
    args = vars(ap.parse_args())

    mask_path = args['maskpath']
    out_path = args['outpath']
    file_name = args['savename']
    num_classes = int(args['number'])
        
    calculate_weights(mask_path, out_path, file_name, num_classes)
