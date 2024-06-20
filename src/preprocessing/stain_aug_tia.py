#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
stain_aug_tia.py: generates N new patches with different colour profiles but the same mask - alternate method using TIA Toolbox

choice of models: macenko, vahadane

author: Holly Rafique
email: holly.rafique@kcl.ac.uk
'''

import datetime
import argparse
import time
import yaml
import os
import glob
import cv2
from tiatoolbox.tools.stainaugment import StainAugmentor
import numpy as np
    
def stain_augment(image_path, output_path, method_name, sig1=0.7, sig2=0.3, bgaug=False):
    #output_path = os.path.join(output_path,method_name+"_sigma_"+str(sig1)+"_"+str(sig2)+"_"+str(bgaug))
    #os.makedirs(output_path,exist_ok=True)

    #read all images names in directories
    image_paths=glob.glob(os.path.join(image_path,'*.png'))
    image_paths = sorted(image_paths)

    for i_path in image_paths:
        # Read data
        print(i_path)
        name = os.path.basename(i_path)
        img = cv2.imread(i_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sig1 = np.random.uniform(0.55,0.85)
        sig2 = np.random.uniform(0.55,0.85)


        stain_augmentor = StainAugmentor(method=method_name,sigma1=sig1, sigma2=sig2, augment_background=bgaug)
        img_aug = stain_augmentor.apply(img)
        cv2.imwrite(os.path.join(output_path,name+"_"+method_name[:3]+".png"),img_aug)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to images to be normalised')
    ap.add_argument('-mp', '--mask_path', required=True, help='path to masks for images to be normalised')
    ap.add_argument('-op', '--output_path', required=True, help='path for saving normalised images')
    ap.add_argument('-mn', '--method_name', required=True, help='name of normalisation method: vahadane/macenko/GAN')
    ap.add_argument('-s1', '--sigma1', default=0.7, help='sigma1 value - between 0 and 1')
    ap.add_argument('-s2', '--sigma2', default=0.7, help='sigma2 value - between 0 and 1')
    ap.add_argument('-bga', '--bgaug', default=False, help='augment the background colour True/False')

    args = ap.parse_args()

    curr_datetime = time.strftime("%Y%m%d-%H%M%S")

    #set up paths for models
    output_path = args.output_path

    print(output_path)
    os.makedirs(args.output_path,exist_ok=True)
    os.makedirs(os.path.join(output_path,"images"),exist_ok=True)
    #os.makedirs(os.path.join(output_path,"masks"),exist_ok=True)

    stain_augment(args.input_path, output_path, args.method_name, args.sigma1, args.sigma2, args.bgaug)




