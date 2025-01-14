#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
stain_normalisation.py: normalises H&E stain of digital pathology slides to a common colour space

Supported methods: macenko, vahadane

WARNING:
    The code uses the staintools package, which is not compatible with more recent versions of python.
    This code will need to be rewritten - potentially using tiatoolbox.

Author: Holly Rafique
Email: holly.rafique@kcl.ac.uk

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


def stain_norm(image_path: str, target_path: str, target_name: str, output_path: str, method_name: str) -> None:
    '''
    Normalize the stain of images using the specified method.

    Args:
        image_path (str): Path to the directory containing input images.
        target_path (str): Path to the target image directory.
        target_name (str): Name of the target image.
        output_path (str): Path to the output directory for saving normalized images.
        method_name (str): Name of the stain normalization method (e.g., "macenko" or "vahadane").
    '''

    if(method_name.lower() in STAINTOOLS_METHODS):
        staintools_norm(image_path, target_path, target_name, output_path, method_name)
    else:
        print(f"Unknown method: {method_name} - exiting without applying normalisation")  
  

def staintools_norm(image_path: str, target_path: str, target_name: str, output_path: str, method_name: str) -> None:
    '''
    Perform stain normalization using staintools.

    Args:
        image_path (str): Path to the directory containing input images.
        target_path (str): Path to the target image directory.
        target_name (str): Name of the target image.
        output_path (str): Path to the output directory for saving normalized images.
        method_name (str): Name of the stain normalization method (e.g., "macenko" or "vahadane").
    '''
    # added this to make sure we don't waste time re-processing existing images
    # TODO: make saved_path an additional command line argument for __main__
    # saved_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/100cohort/stainnormed/images/M2'

    saved_path = output_path
    NO_OVERWRITE=True
    existing = os.listdir(saved_path) if NO_OVERWRITE else []

    # OLD CODE just took the first image in the target paths directory
    #target_paths=glob.glob(os.path.join(target_path,'*.png'))
    #target_path = target_paths[0]
    # NEW CODE uses the specified target image
    target_path=os.path.join(target_path,target_name)
    image_paths=glob.glob(os.path.join(image_path,'*'))

    # 1. define ideal target patch and train normaliser
    target = staintools.read_image(target_path)

    # 2. Create a normaliser with the method name provided and fit to the target image
    normalizer = staintools.StainNormalizer(method=method_name)
    #print("Stain Matrix:",normalizer.extractor.get_stain_matrix(target)
    normalizer.fit(target)

    # 3. Loop through the images
    untransformed = []
    for i_path in image_paths:
        i_name = os.path.basename(i_path)
        if i_name in existing:
            print(f"already processed: {i_name}")
            continue

        print("*\n")
        try:        
            print(f"processing: {i_name}")
            # 3.1 read the image 
            # staintools reads into RGB format
            img = staintools.read_image(i_path)

            # 3.2 standardize brightness
            img = staintools.LuminosityStandardizer.standardize(img)

            # 3.3 stain normalise image towards target
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

        # 3.4 write image to output folder
        #img is RGB so need to make sure we write it correctly as CV2 expects BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, i_name),img)


    print("The following images remain untransformed: ")
    print(untransformed)


if __name__ == '__main__':
    '''
    Entry point for the script. Parses command-line arguments and initiates stain normalization.
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to images to be normalised')
    ap.add_argument('-tp', '--target_path', help='path to target image - should only be one image in the directory')
    ap.add_argument('-tn', '--target_name', default='staintarget.png', help='name of target image')
    ap.add_argument('-op', '--output_path', required=True, help='path for saving normalised images')
    ap.add_argument('-mn', '--method_name', required=True, help='name of normalisation method: vahadane/macenko/GAN') 

    args = ap.parse_args()

    curr_datetime = time.strftime("%Y%m%d-%H%M%S")
    target_path = args.target_path if args.target_path else args.input_path
        
    # set up paths for models
    # this is important to avoid accidentally overwriting previously generated images
    output_path = os.path.join(args.output_path,str(args.method_name+"-"+curr_datetime))
    print(f"saving new images to: {output_path}")
    os.makedirs(output_path,exist_ok=True)


    stain_norm(args.input_path, target_path, args.target_name, output_path, args.method_name)
