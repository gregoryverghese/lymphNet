import os
import glob
import argparse

import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from pyslide.slide import Annotations,Slide
from pyslide.patching import Patch
#from utilities import mask2rgb

STEP=1024
MAG_LEVEL=1
SIZE=(2048,2048)

WSI_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi-masks'
SAVE_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/v2/20x-2048-1024'
WSI_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/wsi/train'
ANNOTATIONS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/annotations/v2'
FILTER_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/images'

#patient_lst=['11.90656 C L1.2','90157_01_LR_and_R']

wsi_paths=glob.glob(os.path.join(WSI_PATH,'*'))
annotations_paths=glob.glob(os.path.join(ANNOTATIONS_PATH,'*'))
classes=['GC','sinus']

filtered=glob.glob(os.path.join(FILTER_PATH,'*'))
filtered=[os.path.basename(f)[:-4] for f in filtered]

print(len(filtered))

saved=glob.glob(os.path.join(SAVE_PATH,'*'))
saved=[os.path.basename(s) for s in saved]

total=0

for p in wsi_paths:
    name=os.path.basename(p)[:-5]
    #if name in saved:
        #continue
    print('slide',name)
    if '90405_02_R'==name:
        continue
    print(name)
    #print(annotations_paths)
    ann_path=[a for a in annotations_paths if name in a]
    print('length',len(ann_path))
    if len(ann_path)==0:
        continue
    save_path=os.path.join(SAVE_PATH,name)
    os.makedirs(save_path,exist_ok=True)

    #annotate=Annotations(ann_path,source='qupath',labels=classes)
    #annotations=annotate._annotations
    #wsi=Slide(p,annotations=annotate)
    #mask=wsi.slide_mask
    #cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'_mask_.png'),mask)
    
    #get border from border annotation
    annotate=Annotations(ann_path,source='qupath',labels=['border'])
    annotations=annotate._annotations
    print(len(annotations))
    if len(annotations)==0:
        continue
    border=annotations['border'][0]
    xs,ys=list(zip(*border))
    border=[[min(xs),max(xs)],[min(ys),max(ys)]]
    
    #germinal centres
    annotate_germs=Annotations(ann_path,source='qupath',labels=['GC'])
    annotations=annotate_germs._annotations 
    wsi_germs=Slide(p,annotations=annotate_germs)
    
    patches=Patch(wsi_germs,mag_level=MAG_LEVEL,border=border,size=SIZE)
    num=patches.generate_patches(STEP)
    #print('g','num patches: {}'.format(num))
    #for p_idx in patches._patches:
        #if p_idx['name'] not in filtered:
            #patches._patches.remove(p_idx)
    #test=[p_id for p_id in patches._patches if p_id['name'] in filtered]
    #patches._patches=test
    print('g','num patches: {}'.format(len(patches._patches)))
    patches.save(save_path,mask_flag=True)
    #patches.save(save_path,mask_flag=True)
    #patches.save_mask(save_path, 'mask')
        
    #sinuses
    annotate_sinus=Annotations(ann_path,source='qupath',labels=['sinus'])
    annotations=annotate_sinus._annotations 
    wsi_sinus=Slide(p,annotations=annotate_sinus)
    
    patches=Patch(wsi_sinus,mag_level=MAG_LEVEL,border=border,size=SIZE)
    num=patches.generate_patches(STEP)
    #test=[p_id for p_id in patches._patches if p_id['name'] in filtered]
    #print('g','num patches: {}'.format(num))
    #patches._patches=test
    patches.save(save_path,mask_flag=True)
    patches.save_mask(save_path,'sinus_mask')

    #s=Stitching(os.path.join(save_path,'images'),mag_level=MAG_LEVEL,name=name)
    #canvas=s.stitch((2000,2000))
    #canvas=cv2.resize(canvas,(2000,2000))
    #cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'_stitch.png'),canvas)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-cp',
                    '--config', 
                    required=True,
                    help='config file path')

    args=vars(ap.parse_args())
"""
