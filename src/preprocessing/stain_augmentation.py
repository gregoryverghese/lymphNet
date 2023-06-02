#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
stain_augmentation.py: generates N new patches with different colour profiles but the same mask

choice of models: macenko, vahadane

author: Holly Rafique
email: holly.rafique@kcl.ac.uk
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
import imgaug.augmenters as iaa

STAINTOOLS_METHODS = ['vahadane', 'macenko']
GAN_METHODS = ['gan', 'staingan']
NET_METHODS = ['net', 'stainnet'] 


NUM_AUGS = 1

def stain_aug(image_path,mask_path, output_path, method_name, sig1=0.7, sig2=0.7, bgaug=False):
    
    #can use staintools module for vahadane and macenko
    #if(method_name.lower() in STAINTOOLS_METHODS):
    #    staintools_aug(image_path, output_path, method_name, sig1, sig2, bgaug) 

        

    #read all images names in directories
    image_paths=glob.glob(os.path.join(image_path,'*.png'))
    #print(image_paths)
    #during testing we will just try a few images at once
    #image_paths = image_paths[:10]

    for i_path in image_paths:
        # Read data
        print(i_path)
        name =os.path.basename(i_path) 
        staintools_aug(i_path,mask_path,output_path,STAINTOOLS_METHODS[0],sig1,sig2,bgaug)
        staintools_aug(i_path,mask_path,output_path,STAINTOOLS_METHODS[1],sig1,sig2,bgaug)
        hsv_augment(i_path,mask_path,output_path)       




def staintools_aug(image_path,mask_path, output_path, method_name, sig1=0.7, sig2=0.7, bgaug=False):

        img = staintools.read_image(image_path)
        name = os.path.basename(image_path) 
        mask = cv2.imread(os.path.join(mask_path,name))
        # Standardize brightness (This step is optional but can improve the tissue mask calculation)
        img = staintools.LuminosityStandardizer.standardize(img)

        # Stain augment 1
        augmentor = staintools.StainAugmentor(method=method_name, sigma1=sig1, sigma2=sig2,augment_background=bgaug)
        augmentor.fit(img)

        for i in range(NUM_AUGS):
            augmented_image = augmentor.pop()

            #do we need to convert back to RGB?
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_path,"images",name+"_"+method_name[:3]+"_"+str(i)+".png"),augmented_image)
            cv2.imwrite(os.path.join(output_path,"masks",name+"_"+method_name[:3]+"_"+str(i)+".png"),mask)

def hsv_augment(img_path,mask_path,output_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    name = os.path.basename(img_path)
    mask = cv2.imread(os.path.join(mask_path,name))

    # Define an HSV color augmentation sequence
    augmentation = iaa.Sequential([
        iaa.AddToHue(),
        iaa.MultiplyHueAndSaturation((0.75, 1.25), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.25), add=(-5, 10))
    ])

    # Apply the augmentation to the image
    for i in range(NUM_AUGS):
        augmented_image = augmentation(image=img)
        cv2.imwrite(os.path.join(output_path,"images",name+"_hsv_"+str(i)+".png"),augmented_image)
        cv2.imwrite(os.path.join(output_path,"masks",name+"_hsv_"+str(i)+".png"),mask)


        
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
    output_path = os.path.join(args.output_path,str(curr_datetime)+"-sigma_"+str(args.sigma1)+"_"+str(args.sigma2)+"_"+str(args.bgaug))


    print(output_path)
    os.makedirs(output_path,exist_ok=True)
    os.makedirs(os.path.join(output_path,"images"),exist_ok=True)
    os.makedirs(os.path.join(output_path,"masks"),exist_ok=True)

    stain_aug(args.input_path, args.mask_path, output_path, args.method_name, args.sigma1, args.sigma2, args.bgaug)


