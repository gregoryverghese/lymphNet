"""
" generate_patches.py
" uses the pyslide library to patch a WSI and create masks for the patches
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
SIZE=(1024,1024)
#/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/testing/baseline
WSI_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/test/wsi-masks'
SAVE_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/testing/baseline/patches'
WSI_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/wsi/all'
ANNOTATIONS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/annotations'
ANNOTATIONS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/testing/baseline/annotations'
FILTER_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/test/filter'
TISSUE_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/tissue_masks'

TESTIMS = True

def patch_slides(wsi_path, ann_path,tissue_mask_path, save_path, classes=[],wsi_mask=True, wsi_mask_path="", filter_path=""):
    print("in patch_images")
    draw_contours=False
    wsi_paths=glob.glob(os.path.join(wsi_path,'*.ndpi'))
    annotations_paths=glob.glob(os.path.join(ann_path,'*'))
    #print(annotations_paths)    
    ##what is filtered for?
    filtered=glob.glob(os.path.join(filter_path,'*'))
    filtered=[os.path.basename(f)[:-4] for f in filtered]
    #print(len(filtered))
    
    #presumably this is so that we can skip ones we have already done?
    saved=glob.glob(os.path.join(save_path,'*'))
    saved=[os.path.basename(s) for s in saved]
    print("saved: ",saved)    
    for curr_path in wsi_paths:
        #remove ".ndpi" extension
        name=os.path.basename(curr_path)[:-5]

        if name in saved:
            print('skipping: ',name)
            continue
        print('slide',name)

        ### skipping this image as it seems corrupt - doesn't load into QuPath
        #if '90405_02_R'==name:
        #    continue

        #get the paths for all the annotations that match the name of the image
        ann_path=[a for a in annotations_paths if name in a]
        #print('length',len(ann_path))
        print(ann_path)
        #skip if there are no annotations for this image
        if len(ann_path)==0:
            print("no annotation files!",name)
            continue
        print('slide',name)

        #the patch masks don't have an extension??
        curr_save_path=os.path.join(save_path,name)
        os.makedirs(curr_save_path,exist_ok=True)

        
        #retrieve all annotations for the specified classes
        annotate=Annotations(ann_path,source='qupath',labels=classes)

        #if there are no annotations then skip to next image
        if len(annotate._annotations)==0:
            if not TESTIMS:
                continue
        
        ## need to get border with sinuses as well
        wsi=Slide(curr_path,annotations=annotate)

        ## WRITE MASK FOR WSI
        #if wsi_mask:
        #    mask=wsi.slide_mask           
        #    cv2.imwrite(os.path.join(wsi_mask_path,name+'_mask.png'),mask*255)


        if(draw_contours):
            # GET CONTOURS - useful if we want to print out thumbnails but not using for actual border
            # we only want to use the LNs that we have annotations for

            border_annotate=Annotations(ann_path,source='qupath',labels=['border'])
            border_annotations=border_annotate._annotations
            if len(border_annotations)==0:
                ## CONTOUR ALL LNs
                contours=detect_tissue_section(wsi)
                scale_factor = 2**MAG_LEVEL
            else:
                c=border_annotations['border'][0]
                c = np.array(c)
                contours=[c]
                scale_factor = 1
 
            #DRAW THUMBNAIL and CONTOURS
            slide=wsi.get_thumbnail(wsi.level_dimensions[6])
            slide=np.array(slide.convert('RGB'))
            slide=cv2.drawContours(slide,contours,-1,(0,0,255),3)

            ## DRAW A RECTANGLE around the LN that has annotations
            annotations=list(itertools.chain(*list(annotate.annotations.values())[0]))
            c=match_annotations_to_tissue_contour(contours,annotations,wsi.level_downsamples[6])
        
            rect = cv2.boundingRect(c)
            #print(rect)
            x,y,w,h = rect
            slide=cv2.rectangle(slide,(x,y),(x+w,y+h),(0,255,0),2)
            # save a thumb image showing the countours, rectangle and selected LN
            cv2.imwrite(os.path.join(SAVE_PATH,name+'_slidethumb.png'),slide)

        ### GET THE BORDER for LN to save based on ALL the annotations (original wsi)
        #set the padding to 5% of avg width and height of the region
        if(TESTIMS):
            border_annotate=Annotations(ann_path,source='qupath',labels=['border'])
            border_annotations=border_annotate._annotations
            border=border_annotations['border'][0]
            border = np.array(border)
            x1 = np.min(border[:, 0])
            x2 = np.max(border[:, 0])
            y1 = np.min(border[:, 1])
            y2 = np.max(border[:, 1])
            border = [[x1,x2],[y1,y2]]
        else:
            border=wsi.get_border(space=500)
            (x1,x2),(y1,y2)=border
        print(border)
        ### FEATURE TO MASK & SAVE
        annotate_feature=Annotations(ann_path,source='qupath',labels=['GC'])
        annotations=annotate_feature._annotations 
        wsi_feature=Slide(curr_path,annotations=annotate_feature)

        ## Apply Tissue Mask
        tissue_mask=np.load(os.path.join(tissue_mask_path,name+".ndpi.npy"))

        # Convert True/False values to 0/1
        tissue_mask = tissue_mask.astype(np.uint8)*255
        #tissue_mask = np.transpose(tissue_mask)
        tissue_mask_mag = 2.5
        slide_mag=40
        #print("tissue_mask before scaling:",tissue_mask.shape)
        tissue_mask_scaled = cv2.resize(tissue_mask, (0, 0), fx=slide_mag/tissue_mask_mag, fy=slide_mag/tissue_mask_mag)
        wsi_feature.set_filter_mask(mask=tissue_mask_scaled)


        patches=Patch(wsi_feature,mag_level=MAG_LEVEL,border=border,size=SIZE)
        #print("slide dims:",wsi_feature.dimensions)        
        
        ##TESTING
        ##patch,filter_mask = wsi_feature.get_filtered_region((0,0), 2,(34000,18000))
        ##patch_path=os.path.join(curr_save_path,'images')
        ##os.makedirs(patch_path,exist_ok=True)
        ##image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        ##status=cv2.imwrite(os.path.join(patch_path,name+"_region.png"),image
        ##continue

        num=patches.generate_patches(STEP)
 
        print('g','num patches: {}'.format(len(patches._patches)))
        #print(patches._patches)
        
        ## SAVE PATCHES 
        patches.save(curr_save_path,mask_flag=True)
              
        #patches.save_mask(curr_save_path, 'mask')
        #s=Stitching(os.path.join(save_path,'images'),mag_level=MAG_LEVEL,name=name)
        #canvas=s.stitch((2000,2000))
        #canvas=cv2.resize(canvas,(2000,2000))
        #cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'_stitch.png'),canvas)




if __name__=='__main__':
    
    #ap=argparse.ArgumentParser()
    #ap.add_argument('-cp','--configpath', required=True,help='path to config file')
    #args=vars(ap.parse_args())

    classes=['GC','sinus']
    patch_slides(WSI_PATH, ANNOTATIONS_PATH, TISSUE_MASK_PATH,SAVE_PATH, classes, wsi_mask=True, wsi_mask_path=WSI_MASK_PATH)



