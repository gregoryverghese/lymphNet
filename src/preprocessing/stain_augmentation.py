#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
stain_augmentation.py: generates N new patches with different colour profiles and duplicates the unadjusted mask

WARNING:
The code uses the staintools package, which is not compatible with more recent versions of python.  
This code will need to be rewritten - potentially using tiatoolbox.

choice of stain methods: macenko, vahadane

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
import imgaug.augmenters as iaa

STAINTOOLS_METHODS = ['vahadane', 'macenko']

# Dictionary to map methods to their corresponding functions
AUGMENTATION_FUNCTIONS: Dict[str, Callable] = {
    'vahadane': lambda *args, **kwargs: staintools_aug(*args, method_name='vahadane', **kwargs),
    'macenko': lambda *args, **kwargs: staintools_aug(*args, method_name='macenko', **kwargs),
    'hsv': hsv_augment
}

NUM_AUGS = 1

   
def stain_aug(image_path: str, mask_path: str, output_path: str, method_name: str, sig1: float = 0.7, sig2: float = 0.7, bgaug: bool = False) -> None:
    """
    Performs stain augmentation on images in a given directory using specified methods.
    Saves the new image and the unadjusted mask to output_path/images and output_path/masks.

    Args:
        image_path (str): Path to the directory containing input images.
        mask_path (str): Path to the directory containing corresponding masks.
        output_path (str): Path to the directory for saving augmented images and masks.
        method_name (str): Name of the stain augmentation method to use.
        sig1 (float, optional): Sigma1 value for augmentation. Defaults to 0.7.
        sig2 (float, optional): Sigma2 value for augmentation. Defaults to 0.7.
        bgaug (bool, optional): Whether to augment the background color. Defaults to False.

    Returns:
        None
    """

    #can use staintools module for vahadane and macenko
    #if(method_name.lower() in STAINTOOLS_METHODS):
    #    staintools_aug(image_path, output_path, method_name, sig1, sig2, bgaug) 

        

    #read all images names in directories
    image_paths=glob.glob(os.path.join(image_path,'*.png'))
    image_paths = sorted(image_paths)

    
    for i_path in image_paths:
        # Read data
        name =os.path.basename(i_path) 

        ## OLD CODE: specifically only using Macenko augmentation
        ## TODO: modify code so that you specify all the methods you want to use when calling the method
        #staintools_aug(i_path,mask_path,output_path,STAINTOOLS_METHODS[1],sig1,sig2,bgaug)

        # Call the appropriate function dynamically
        augment_func = AUGMENTATION_FUNCTIONS.get(method_name.lower())
        if augment_func:
            if method_name.lower() in STAINTOOLS_METHODS:
                augment_func(i_path, mask_path, output_path, sig1=sig1, sig2=sig2, bgaug=bgaug)
            else:
                augment_func(i_path, mask_path, output_path)
        else:
            print(f"Unknown method: {method_name} - returning without augmenting")


def staintools_aug(image_path: str, mask_path: str, output_path: str, method_name: str, sig1: float = 0.7, sig2: float = 0.7, bgaug: bool = False) -> None:
    """
    Applies stain augmentation using staintools methods.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the directory containing masks.
        output_path (str): Path to the directory for saving augmented images and masks.
        method_name (str): Name of the staintools method (e.g., vahadane, macenko).
        sig1 (float, optional): Sigma1 value for augmentation. Defaults to 0.7.
        sig2 (float, optional): Sigma2 value for augmentation. Defaults to 0.7.
        bgaug (bool, optional): Whether to augment the background color. Defaults to False.

    Returns:
        None
    """
    img = staintools.read_image(image_path)
    name = os.path.basename(image_path) 
    mask = cv2.imread(os.path.join(mask_path,name))

    # 1. Standardize brightness (This step is optional but can improve the tissue mask calculation)
    img = staintools.LuminosityStandardizer.standardize(img)

    # 2. Fit an augmentor with our params and the image
    augmentor = staintools.StainAugmentor(method=method_name, sigma1=sig1, sigma2=sig2,augment_background=bgaug)
    augmentor.fit(img)

    # if we want multiple augmented versions then we can set NUM_AUGS > 1
    for i in range(NUM_AUGS):
        #3. Get an augmented image 
        augmented_image = augmentor.pop()

        #optionally conveert back to RGB
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path,"images",name+"_"+method_name[:3]+"_"+str(i)+".png"),augmented_image)
        cv2.imwrite(os.path.join(output_path,"masks",name+"_"+method_name[:3]+"_"+str(i)+".png"),mask)


def hsv_augment(img_path: str, mask_path: str, output_path: str) -> None:
    """
    Applies HSV color augmentation to the input image.

    Args:
        img_path (str): Path to the input image.
        mask_path (str): Path to the directory containing masks.
        output_path (str): Path to the directory for saving augmented images and masks.

    Returns:
        None
    """

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    name = os.path.basename(img_path)
    mask = cv2.imread(os.path.join(mask_path,name))

    # 1. Define an HSV color augmentation sequence
    # The values used here produce a good colour spread but could be adjusted as required
    augmentation = iaa.Sequential([
        iaa.AddToHue(),
        iaa.MultiplyHueAndSaturation((0.75, 1.25), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.25), add=(-5, 10))
    ])

    # Apply the augmentation to the image
    # if we want multiple augmented versions then we can set NUM_AUGS > 1
    for i in range(NUM_AUGS):
        # 2. Get an augmented image
        augmented_image = augmentation(image=img)
        cv2.imwrite(os.path.join(output_path,"images",name+"_hsv_"+str(i)+".png"),augmented_image)
        cv2.imwrite(os.path.join(output_path,"masks",name+"_hsv_"+str(i)+".png"),mask)


        
if __name__ == '__main__':
    """
    Main function to parse command-line arguments and initiate stain augmentation.

    Command-line Args:
        -ip/--input_path (str): Path to images to be normalized.
        -mp/--mask_path (str): Path to masks for images to be normalized.
        -op/--output_path (str): Path for saving normalized images.
        -mn/--method_name (str): Name of the normalization method (e.g., vahadane/macenko/GAN).
        -s1/--sigma1 (float, optional): Sigma1 value. Defaults to 0.7.
        -s2/--sigma2 (float, optional): Sigma2 value. Defaults to 0.7.
        -bga/--bgaug (bool, optional): Whether to augment the background color. Defaults to False.

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to images to be normalised')
    ap.add_argument('-mp', '--mask_path', required=True, help='path to masks for images to be normalised')
    ap.add_argument('-op', '--output_path', required=True, help='path for saving normalised images')
    # TODO: Enable a list of method_names to be passed
    ap.add_argument('-mn', '--method_name', required=True, help='name of normalisation method: vahadane/macenko/GAN') 
    ap.add_argument('-s1', '--sigma1', default=0.7, help='sigma1 value - between 0 and 1')
    ap.add_argument('-s2', '--sigma2', default=0.7, help='sigma2 value - between 0 and 1')
    ap.add_argument('-bga', '--bgaug', default=False, help='augment the background colour True/False')

    args = ap.parse_args()

    curr_datetime = time.strftime("%Y%m%d-%H%M%S")
        
    #set up paths for models
    output_path = os.path.join(args.output_path,str(curr_datetime)+"-sigma_"+str(args.sigma1)+"_"+str(args.sigma2)+"_"+str(args.bgaug))


    print(f"Setting output path: {output_path}")
    os.makedirs(output_path,exist_ok=True)
    os.makedirs(os.path.join(output_path,"images"),exist_ok=True)
    os.makedirs(os.path.join(output_path,"masks"),exist_ok=True)

    stain_aug(args.input_path, args.mask_path, output_path, args.method_name, args.sigma1, args.sigma2, args.bgaug)


