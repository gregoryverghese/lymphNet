#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
stain_normalisation.py: normalises H&E stain of digital pathology slides to a common colour space

choice of models: macenko, vahadane
1
2
'''

import staintools
import datetime
import argparse
import time
import yaml
import os
import glob
import cv2


def stain_norm(image_path, target_path, output_path, method_name):

    #read all images names in directories
    target_paths=glob.glob(os.path.join(target_path,'*'))
    image_paths=glob.glob(os.path.join(image_path,'*'))

    #define ideal target patch and train normaliser
    #if there is more than one image in the target folder we will only read the first
    target = staintools.read_image(target_paths[0])
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method=method_name)
    normalizer.fit(target)

    #during testing we will just try a few images at once
    image_paths = image_paths[:100]
   
    for i_path in image_paths:
        # Read data from names 
        #staintools reads into RGB format
        img = staintools.read_image(i_path)
        #standardize brightness
        img = staintools.LuminosityStandardizer.standardize(img)
        #stain normalise image towards target
        img = normalizer.transform(img)
        #write image to output folder
        #img is RGB so need to make sure we write it correctly as CV2 expects BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, os.path.basename(i_path)),img)


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
    ap.add_argument('-tp', '--target_path', required=True, help='path to target image - should only be one image in the directory')
    ap.add_argument('-op', '--output_path', required=True, help='path for saving normalised images')
    ap.add_argument('-mn', '--method_name', help='name of normalisation method: vahadane/macenko/GAN') 

    args = ap.parse_args()

    curr_datetime = time.strftime("%Y%m%d-%H%M%S")


    #set up paths for models
    output_path = os.path.join(args.output_path,curr_datetime)
    os.makedirs(output_path,exist_ok=True)

    #method = 'macenko'
    method = 'vahadane'
    if(args.method_name):
        method = args.method_name

    stain_norm(args.input_path, args.target_path, output_path, method)
