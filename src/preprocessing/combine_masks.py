"""
combine_masks.py: 
combine LN segmentation mask and histoqc mask

"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

import glob

HISTOQC_PATH = "D:\\03 Cancer Bioinformatics\\tissue_masks\\100cohort\\histoqc"
LN_SEG_PATH = "D:\\03 Cancer Bioinformatics\\tissue_masks\\100cohort\\using"
OUT_PATH = "D:\\03 Cancer Bioinformatics\\tissue_masks\\100cohort\\combined_with_hqc"
IS_BINARY = True

hqc_paths=glob.glob(os.path.join(HISTOQC_PATH,'*.png'))

for histoqc_path in hqc_paths:
    
    wsi_name = os.path.basename(histoqc_path).replace("_mask_use.png","")
    print(wsi_name)

    histoqc_mask = cv2.imread(histoqc_path, cv2.IMREAD_GRAYSCALE) 
    _, histoqc_binary = cv2.threshold(histoqc_mask, 5, 255, cv2.THRESH_BINARY)

    #read in ln segmentation tissue mask
    ln_mp = os.path.join(LN_SEG_PATH,wsi_name+".png_lnmask.png")
    ln_mask = cv2.imread(ln_mp, cv2.IMREAD_GRAYSCALE) 

    if IS_BINARY:
        ln_binary = ln_mask
    else:
        # Ensure the mask is a binary image (this step is crucial if the mask isn't already binary)
        _, ln_binary = cv2.threshold(ln_mask, 127, 255, cv2.THRESH_BINARY)
    
    #scale up the ln segmentation mask to histoqc size
    ln_resized = cv2.resize(ln_binary, None, fx=4,fy=4, interpolation=cv2.INTER_NEAREST)

    #combine the masks
    combined_mask = cv2.bitwise_and(histoqc_binary, ln_resized)

    #write out the combined mask
    cv2.imwrite(os.path.join(OUT_PATH,wsi_name+'_tissuemask.png'), combined_mask)
    cv2.imwrite(os.path.join(OUT_PATH,'VIZ_'+wsi_name+'.png'), combined_mask*255)

