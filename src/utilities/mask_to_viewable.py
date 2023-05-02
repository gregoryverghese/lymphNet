#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

'''
import glob
import os
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2



def convert_multiple(input_path, output_path):
    #read all images names in directories
    input_paths=glob.glob(os.path.join(input_path,'*'))
    print("converting: "+str(len(input_paths))+" mask files")
    for ip in input_paths:
        convert(ip, output_path)
    print("done converting")


def convert(input_path, output_path):
    #read in image
    img=cv2.imread(input_path)    
    #do we need to do this?
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #convert
    #mask=np.expand_dims(mask,axis=0)
    #img_out = img[0,:,:,:]
    img_out = img*255
    
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    
    #write converted image
    cv2.imwrite(os.path.join(output_path,os.path.basename(input_path)),img_out)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to masks files to be converted' )
    ap.add_argument('-op', '--output_path', required=True, help='path where imgs should be saved')

    args = ap.parse_args()

    os.makedirs(args.output_path,exist_ok=True)
    convert_multiple(args.input_path, args.output_path)


