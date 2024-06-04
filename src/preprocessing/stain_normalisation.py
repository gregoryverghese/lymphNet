#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
stain_normalisation.py: normalises H&E stain of digital pathology slides to a common colour space

choice of models: macenko, vahadane
1
2
'''

import staintools
from staintools.miscellaneous.exceptions import TissueMaskException
import datetime
import argparse
import time
import yaml
import os
import glob
import cv2

STAINTOOLS_METHODS = ['vahadane', 'macenko']
GAN_METHODS = ['gan', 'staingan']
NET_METHODS = ['net', 'stainnet'] 


def stain_norm(image_path, target_path, target_name, output_path, method_name):


    
    #can use staintools module for vahadane and macenko
    if(method_name.lower() in STAINTOOLS_METHODS):
        staintools_norm(image_path, target_path, target_name, output_path, method_name) 
  
    #elif(method_name.lower() in GAN_METHODS):
    #elif(method_name.lower() in NET_METHODS):

def staintools_norm(image_path, target_path, target_name, output_path, method_name):

    saved_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/100cohort/stainnormed/images/M2'
    NO_OVERWRITE=True

    if NO_OVERWRITE:
        existing = os.listdir(saved_path)
    else:
        existing = []
    #read all images names in directories
    #target_paths=glob.glob(os.path.join(target_path,'*.png'))
    target_path=os.path.join(target_path,target_name)
    image_paths=glob.glob(os.path.join(image_path,'*'))
    print(image_paths)
    #define ideal target patch and train normaliser
    #if there is more than one image in the target folder we will only read the first
    target = staintools.read_image(target_path)
    #target = staintools.LuminosityStandardizer.standardize(target)
    print("target defined")
    normalizer = staintools.StainNormalizer(method=method_name)

    print("Stain Matrix:",normalizer.extractor.get_stain_matrix(target))
    
    normalizer.fit(target)
    print("fitted")
    #during testing we will just try a few images at once
    #image_paths = image_paths[:100]
    untransformed = []

    for i_path in image_paths:
        i_name = os.path.basename(i_path)
        if i_name in existing:
            print("already processed: ",i_name)
            continue

        print("*\n")
        try:        
            print(i_path)
            # Read data from names 
            #staintools reads into RGB format
            img = staintools.read_image(i_path)
            #standardize brightness
            img = staintools.LuminosityStandardizer.standardize(img)
            #stain normalise image towards target
            img = normalizer.transform(img)
        
            pass
        except TissueMaskException as e:
            print("cannot transform so using the original")
            print(e)  
            untransformed.append(i_path)          
            pass
        except Exception as e:
            print("something unexpected happened")
            print(e)
            untransformed.append(i_path)
            pass

        #write image to output folder
        #img is RGB so need to make sure we write it correctly as CV2 expects BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, i_name),img)


    print("The following images remain untransformed: ")
    print(untransformed)
    #USING LIST COMPREHENSION - too memory intensive for 4k images at once?
    #images = []
    #images = [staintools.read_image(i_path)  for i_path in image_paths]
    #Standardise Luminosity
    #images = [staintools.LuminosityStandardizer.standardize(img) for img in images]
    #Normalise Images
    #images = [normalizer.transform(img) for img in images]


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to images to be normalised')
    ap.add_argument('-tp', '--target_path', help='path to target image - should only be one image in the directory')
    ap.add_argument('-tn', '--target_name', default='staintarget.png', help='name of target image')
    ap.add_argument('-op', '--output_path', required=True, help='path for saving normalised images')
    ap.add_argument('-mn', '--method_name', required=True, help='name of normalisation method: vahadane/macenko/GAN') 

    args = ap.parse_args()

    curr_datetime = time.strftime("%Y%m%d-%H%M%S")

    if(args.target_path):
        target_path = args.target_path
    else:
        target_path = args.input_path
        
    #set up paths for models
    output_path = os.path.join(args.output_path,str(args.method_name+"-"+curr_datetime))
    print(output_path)
    os.makedirs(output_path,exist_ok=True)


    stain_norm(args.input_path, target_path, args.target_name, output_path, args.method_name)
