#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stain_aug_tia.py: This script generates new image patches with altered color profiles using the TIA Toolbox for stain augmentation.

Supported methods for augmentation include macenko, vahadane. It applies random variations to image color characteristics
while preserving the masks, ensuring the semantic content remains consistent.

This is the preferred script as it uses TiaToolbox rather than staintools, which is not compatible with most packages on Python>3.9

Author: Holly Rafique
Email: holly.rafique@kcl.ac.uk
"""

import datetime
import argparse
import time
import yaml
import os
import glob
import cv2
from tiatoolbox.tools.stainaugment import StainAugmentor
import numpy as np
    
def stain_augment(
    image_path: str, 
    output_path: str, 
    method_name: str, 
    sig1: float = 0.7, 
    sig2: float = 0.3, 
    bgaug: bool = False
) -> None:
    """Perform stain augmentation on a set of images.

    Args:
        image_path (str): Path to the directory containing input images.
        output_path (str): Path to the directory where augmented images will be saved.
        method_name (str): Name of the stain normalization method (e.g., 'macenko', 'vahadane', 'GAN').
        sig1 (float, optional): Sigma1 value for the augmentation, controlling variation. Defaults to 0.7.
        sig2 (float, optional): Sigma2 value for the augmentation, controlling variation. Defaults to 0.3.
        bgaug (bool, optional): Whether to augment background color. Defaults to False.

    Returns:
        None
    """

    #read all images names in directories
    image_paths=glob.glob(os.path.join(image_path,'*.png'))
    image_paths = sorted(image_paths)

    for i_path in image_paths:
        print(f"Processing: {i_path}")
        name = os.path.basename(i_path)
        # 1. Read the image
        img = cv2.imread(i_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Previously fixed sig1 and sig2 at 0.7 but drawing from a uniform distrib gives more randomness
        sig1 = np.random.uniform(0.55,0.85)
        sig2 = np.random.uniform(0.55,0.85)

        # 2. Define a stainaugmentor
        stain_augmentor = StainAugmentor(method=method_name,sigma1=sig1, sigma2=sig2, augment_background=bgaug)
        # 3. Get a new augmented image 
        # (ould apply this multiple times for multiple versions of the image)
        img_aug = stain_augmentor.apply(img)
        cv2.imwrite(os.path.join(output_path,name+"_"+method_name[:3]+".png"),img_aug)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to images to be normalised')
    ap.add_argument('-op', '--output_path', required=True, help='path for saving normalised images')
    ap.add_argument('-mn', '--method_name', required=True, help='name of normalisation method: vahadane/macenko/GAN')
    ap.add_argument('-s1', '--sigma1', default=0.7, help='sigma1 value - between 0 and 1')
    ap.add_argument('-s2', '--sigma2', default=0.7, help='sigma2 value - between 0 and 1')
    ap.add_argument('-bga', '--bgaug', default=False, help='augment the background colour True/False')

    args = ap.parse_args()

    curr_datetime = time.strftime("%Y%m%d-%H%M%S")

    os.makedirs(args.output_path,exist_ok=True)
    os.makedirs(os.path.join(args.output_path,"images"),exist_ok=True)
 
    stain_augment(
        image_path=args.input_path,
        output_path=args.output_path,
        method_name=args.method_name,
        sig1=args.sigma1,
        sig2=args.sigma2,
        bgaug=args.bgaug
    )




