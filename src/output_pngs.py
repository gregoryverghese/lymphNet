"""
" output_pngs.py
" uses the pyslide library to output a png for the WSI 
" and create masks for the WSIs
" reads annotations of class 'GC' and 'sinus'
"
"""
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
#from utilities import mask2rgb
from pyslide.util.utilities import detect_tissue_section
from pyslide.util.utilities import match_annotations_to_tissue_contour

#step=stride
#if step and size are the same then there will be no overlap of patches
#mag level 2:10x  
STEP=512
MAG_LEVEL=2
SIZE=(4096,4096)

WSI_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/test/wsi-masks'
SAVE_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/testing'
WSI_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/wsi/test'
ANNOTATIONS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/annotations'


def create_pngs(wsi_path, ann_path, save_path, wsi_mask_path):
    print("in create_pngs")
    wsi_paths=glob.glob(os.path.join(wsi_path,'*.ndpi'))
    annotations_paths=glob.glob(os.path.join(ann_path,'*'))

    
    
    for curr_path in wsi_paths:
        #remove ".ndpi" extension
        name=os.path.basename(curr_path)[:-5]
        print(name)

        #get the paths for all the annotations that match the name of the image
        ann_path=[a for a in annotations_paths if name in a]
        #print('length',len(ann_path))
        
        #skip if there are no annotations for this image
        if len(ann_path)==0:
            print("no annotation files!",name)
            continue

        #the patch masks don't have an extension??
        curr_save_path=os.path.join(save_path,name)
        #os.makedirs(curr_save_path,exist_ok=True)



        
        #retrieve all annotations for the specified classes
        annotate=Annotations(ann_path,source='qupath',labels=['GC','sinus'])

        #if there are no annotations then skip to next image
        if len(annotate._annotations)==0:
            print("no annotations")
            continue
        
        ## need to get border with sinuses as well
        wsi=Slide(curr_path,annotations=annotate)

        ## WRITE MASK FOR WSI
        #mask=wsi.slide_mask           
        #cv2.imwrite(os.path.join(wsi_mask_path,name+'_mask.png'),mask*255)


        ## DEFINE BORDER OF WHAT TO PATCH 
        #border = wsi.get_border()
        # we only want to use the LNs that we have annotations for

        border_annotate=Annotations(ann_path,source='qupath',labels=['border'])
        border_annotations=border_annotate._annotations
        if len(border_annotations)==0:
            #continue
            ## CONTOUR ALL LNs
            contours=detect_tissue_section(wsi)
            scale_factor = 64
        else:
            c=annotations['border'][0]
            c = np.array(c)
            contours=[c]
            scale_factor = 1
 
        #DRAW THUMBNAIL and CONTOURS
        print(wsi.level_downsamples[3])
        slide=wsi.get_thumbnail(wsi.level_dimensions[3])
        print(wsi.level_dimensions[3])
        print(wsi.level_count)
        slide=np.array(slide.convert('RGB'))
        slide=cv2.drawContours(slide,contours,-1,(0,0,255),3)

        ## DRAW A RECTANGLE around the LN that has annotations
        annotations=list(itertools.chain(*list(annotate.annotations.values())[0]))
        print(annotations)
        c=match_annotations_to_tissue_contour(contours,annotations,wsi.level_downsamples[3])
        
        rect = cv2.boundingRect(c)
        print(rect)
        x,y,w,h = rect
        slide=cv2.rectangle(slide,(x,y),(x+w,y+h),(0,255,0),2)
        # save a thumb image showing the countours, rectangle and selected LN
        cv2.imwrite(os.path.join(SAVE_PATH,name+'_slidethumb.png'),slide)
        
        ## RESTRICT PATCHING to only the area defined by the rectangle
        border=[[x*scale_factor,(x+w)*scale_factor],[y*scale_factor,(y+h)*scale_factor]]

        print("getting region")
        section_img,section_mask = wsi.generate_region(0,x,y,w,h,True,scale_factor)
        print("saving imgs and masks")
        cv2.imwrite(os.path.join(save_path,name+'_viewmask.png'),section_mask*255)
        cv2.imwrite(os.path.join(save_path,name+'_mask.png'),section_mask)
        cv2.imwrite(os.path.join(save_path,name+'.png'),section_img)

        ### FEATURE TO MASK & SAVE
        #annotate_feature=Annotations(ann_path,source='qupath',labels=['GC'])
        #annotations=annotate_feature._annotations 
        #wsi_feature=Slide(curr_path,annotations=annotate_feature)

        #patches=Patch(wsi_feature,mag_level=MAG_LEVEL,border=border,size=SIZE)
        #num=patches.generate_patches(STEP)
 
        #print('g','num patches: {}'.format(len(patches._patches)))
    
        ## SAVE PATCHES 
        #patches.save(curr_save_path,mask_flag=True)
              
        #patches.save_mask(curr_save_path, 'mask')
        #s=Stitching(os.path.join(save_path,'images'),mag_level=MAG_LEVEL,name=name)
        #canvas=s.stitch((2000,2000))
        #canvas=cv2.resize(canvas,(2000,2000))
        #cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'_stitch.png'),canvas)




if __name__=='__main__':
    
    #ap=argparse.ArgumentParser()
    #ap.add_argument('-cp','--configpath', required=True,help='path to config file')
    #args=vars(ap.parse_args())

    create_pngs(WSI_PATH, ANNOTATIONS_PATH, SAVE_PATH, WSI_MASK_PATH)



