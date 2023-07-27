import os
import glob
import argparse
import itertools
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from pyslide.slide import Annotations,Slide
from pyslide.patching import Patch

from pyslide.util.utilities import detect_tissue_section
from pyslide.util.utilities import match_annotations_to_tissue_contour
#from utilities import mask2rgb

STEP=512
MAG_LEVEL=2
SIZE=(1024,1024)

#WSI_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi-masks'
#SAVE_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/v2/20x-2048-1024'
SAVE_PATH='/home/verghese/test'
WSI_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/wsi/all'
ANNOTATIONS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/smuLymphNet/annotations'
#FILTER_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/images'

wsi_paths=glob.glob(os.path.join(WSI_PATH,'*'))
annotations_paths=glob.glob(os.path.join(ANNOTATIONS_PATH,'*'))
classes=['GC','sinus']

#filtered=glob.glob(os.path.join(FILTER_PATH,'*'))
#filtered=[os.path.basename(f)[:-4] for f in filtered]
#saved=glob.glob(os.path.join(SAVE_PATH,'*'))
#saved=[os.path.basename(s) for s in saved]
def main_tiler(wsi_path,ann_path,args)
    
    annotate=Annotations(ann_path,source='qupath',labels=classes)
    annotations=annotate._annotations
    wsi=Slide(p,annotations=annotate)
    border=wsi.get_border(space=150)
    mask=wsi.slide_mask
    
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
    #print(patches._patches)
    #patches.save(save_path,mask_flag=True)
    #patches.save(save_path,mask_flag=True)
    #patches.save_mask(save_path, 'mask')
    """    
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
    #patches.save_mask(save_path,'sinus_mask')

    #s=Stitching(os.path.join(save_path,'images'),mag_level=MAG_LEVEL,name=name)
    #canvas=s.stitch((2000,2000))
    #canvas=cv2.resize(canvas,(2000,2000))
    #cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'_stitch.png'),canvas)

STEP=512
MAG_LEVEL=2
SIZE=(1024,1024)

#WSI_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi-masks'
#SAVE_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/patches/v2/20x-2048-1024'
SAVE_PATH='/home/verghese/test'
WSI_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/wsi/all'
ANNOTATIONS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/smuLymphNet/annotations'
#FILTER_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/images'
"""
if __name__=='__main__':

    ap=argparse.ArgumentParser()
    ap.add_argument('-wp','--wsi_path',required=True,type=str)
    ap.add_argument('-ap','--ann_path',required=True,type=str)
    ap.add_argument('-sp','--save_path',required=True,type=str)
    ap.add_argument('-s','--step',required=True,type=int,default=512)
    ap.add_argument('-m','--mag_level',required=True,type=int,default=2)
    ap.add_argument('-d','--dims',required=True,type=int,default=1024)
    args=ap.parse_args()

    os.makedirs(args.save_path,exist_ok=True)

    for wp in args.wsi_path:
        name=os.path.basename(wp)[:-5]
        print(f'tiling {name}')
        ap=[a for a in args.ann_path if name in a]
        
        if len(ap)==0:
            print(f'skipping {name}')
            continue
    
        save_path=os.path.join(SAVE_PATH,name)
        os.makedirs(save_path,exist_ok=True)
        main_tiling(wp,ap,args)




    #cv2.imwrite(os.path.join(SAVE_PATH,name+'_mask_.png'),mask*255)

    #get border from border annotation
    #annotate=Annotations(ann_path,source='qupath',labels=['border'])
    #annotations=annotate._annotations
    #print(len(annotations))
    #if len(annotations)==0:
        #continue
    #border=annotations['border'][0]
    #xs,ys=list(zip(*border))
    #border=[[min(xs),max(xs)],[min(ys),max(ys)]]

    #contour LNs and get border
    #contours=detect_tissue_section(wsi)
    #print(wsi.level_downsamples[6])
    #slide=wsi.get_thumbnail(wsi.level_dimensions[6])
    #slide=np.array(slide.convert('RGB'))
    #slide=cv2.drawContours(slide,contours,-1,(0,0,255),3)

    #annotations=list(itertools.chain(*list(annotate.annotations.values())[0]))
    #c=match_annotations_to_tissue_contour(contours,annotations,wsi.level_downsamples[6])
    #rect = cv2.boundingRect(c)
    #x,y,w,h = rect
    #slide=cv2.rectangle(slide,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.imwrite(os.path.join(SAVE_PATH,name+'_slidethumb.png'),slide)
    #border=[[x*64,(x+w)*64],[y*64,(y+h)*64]]


