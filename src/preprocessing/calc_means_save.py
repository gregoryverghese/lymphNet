#!/usr/bin/env python3

'''
' author: Holly Rafiqe
' email: holly.rafique@kcl.ac.uk
'
' returns: means and std in RGB format
'
'''

import os
import glob
import argparse

import cv2
import numpy as np
import pandas as pd

def calculate_std_mean(path,outpath):
    print(path)
    images = glob.glob(os.path.join(path,'*'))
    image_shape = cv2.imread(images[0]).shape
    channel_num = image_shape[-1]
    channel_values = np.zeros((channel_num))
    channel_values_sq = np.zeros((channel_num))
    print("# images: ",len(images))
    #HR - 14/06/23 - saving each image mean and std to run stats against
    color_distrib={}
    color_distrib['image-name']=[]
    color_distrib['mean']=[]
    color_distrib['std']=[]

    pixels_perimg = image_shape[0]*image_shape[1]   
    pixel_num = len(images)*pixels_perimg
    print('total number pixels: {}'.format(pixel_num))

    for path in images:
        print(path)
        image = cv2.imread(path)
        #HR - 14/06/23 - need to output as RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = (image/255.0).astype('float64')
        
        #HR - 14/06/23 - saving image mean and std to run stats
        img_mean=np.sum(image, axis=(0,1), dtype='float64')/pixels_perimg
        img_var=np.sum(np.square(image-img_mean), axis=(0,1), dtype='float64')
        img_std = np.sqrt(img_var/pixels_perimg, dtype='float64')
        color_distrib['image-name'].append(os.path.basename(path))
        color_distrib['mean'].append(img_mean)
        color_distrib['std'].append(img_std) 

        channel_values += np.sum(image, axis=(0,1), dtype='float64')

    mean=channel_values/pixel_num
    print("*** half way")
    print("mean:",mean)
    for path in images:
        print(path)
        image = cv2.imread(path)
        #HR - 14/06/23 - need to output as RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image/255.0).astype('float64')
        channel_values_sq += np.sum(np.square(image-mean), axis=(0,1), dtype='float64')

    std=np.sqrt(channel_values_sq/pixel_num, dtype='float64')
    print('mean: {}, std: {}'.format(mean, std))

    color_distrib_df = pd.DataFrame(color_distrib)
    color_distrib_df.to_csv(os.path.join(outpath,'color_distribution.csv'))

    return mean, std


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True, help='path to image set')
    ap.add_argument('-op', '--outpath', help='path to save csv')
    args = ap.parse_args()

    if(args.outpath):
        outpath = args.outpath
    else:
        outpath = args.path
    
    mean, std = calculate_std_mean(args.path, outpath)

    










