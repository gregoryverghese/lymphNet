import os
import glob
import argparse

import pandas as pd
import numpy as np
import cv2
import seaborn as sns

from patching import Annotations,Slide,Patching,Stitching
from utilities import mask2rgb

STEP=512
MAG_LEVEL=2
SIZE=(1500,1500)

SI_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-torch/data/masks'
GERMINAL_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-torch/data/patches/10x/germinal'
SINUS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-torch/data/patches/10x/sinus'

wsi_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode/data/wsi/Guys/all/wsi'
annotations_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode/data/wsi/Guys/sum_ff_toms/annotations'

wsi_paths=glob.glob(os.path.join(wsi_path,'*'))
annotations_paths=glob.glob(os.path.join(annotations_path,'*'))

for p in wsi_paths:
    name=os.path.basename(p)[:-5]
    ann_path=[a for a in annotations_paths if name in a]
    ann_obj=Annotations(ann_path,source=['imagej','qupath'])
    ann_obj.generate_annotations()
    ann_obj.filter_labels(['GC', 'sinus', 'GERMINAL CENTRE', 'SINUS'])
    annotations=ann_obj._annotations
    wsi=Slide(p,annotations=annotations,draw_border=True)
    wsi.get_border()
    mask=wsi._slide_mask
    mask=cv2.resize(mask, (1000,1000))
    rgb_mask=mask2rgb(mask)
    cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'.png'),wsi_mask)
    
    ann_obj.filter_labels(['GC', 'GERMINAL CENTRE'])
    annotations=ann_obj._annotations
    new_annotations=annotations
    if len(ann_path)==2:
        new_annotations={}
        new_annotations['germinals']=annotations['GC']+annotations['GERMINAL CENTRE']
    new_annotations={i: v for i, v in enumerate(new_annotations.values())}
    wsi_germs=Slide(p,annotations=new_annotations,draw_border=True)
    wsi_germs.get_border()
    patches=Patching(wsi_germs,mag_level=MAG_LEVEL,size=SIZE)
    num=patches.generate_patches(STEP, mask_flag=True)
    print('num patches: {}'.format(num))
    patches.save(GERMINAL_PATH,mask_flag=True)

    ann_obj.filter_labels(['sinus', 'SINUS'])
    annotations=ann_obj._annotations
    new_annotations=annotations
    if len(ann_path)==2:
        new_annotations={}
        new_annotations['sinus']=annotations['sinus']+annotations['SINUS']
    new_annotations={i: v for i, v in enumerate(new_annotations.values())}
    wsi_sinus=Slide(p,annotations=new_annotations,draw_border=True)
    wsi_sinus.get_border()
    patches=Patching(wsi_sinus,mag_level=MAG_LEVEL,size=SIZE)
    num=patches.generate_patches(STEP, mask_flag=True)
    print('num patches: {}'.format(num))
    patches.save(SINUS_PATH,mask_flag=True)


    #s=Stitching(os.path.join(PATCH_PATH,'masks'),mag_level=MAG_LEVEL,name=name)
    #canvas=s.stitch()
    #canvas=cv2.resize(canvas,(2000,2000))
    #cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'_stitch.png'),canvas)

"""
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-cp','--configpath', required=True,help='path to config
                    file')
    args=vars(ap.parse_args())
"""
