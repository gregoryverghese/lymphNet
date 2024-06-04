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
#mag downsample 4:10x  for a 40x slide
# for svs 
STEP=512
MAG_DS = 4
PATCH_SIZE=(1024,1024)

tissue_mask_mag = 2.5
SAVE_PATCH_MASKS=True
EXTERNAL_TEST = False
slide_mag = 40
classes=['GC']

#not used
WSI_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/test/wsi-masks'


#WSI_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/100-cohort/batch8'
#SAVE_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/100cohort/batch8'
ANNOTATIONS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/annotations/100cohort'
#ANNOTATIONS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/patches/10x/testing/barts/set2/baseline/annotations'
TISSUE_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/tissue_masks/100cohort'

TESTIMS = False
#TISSUE_MASK_EXT=".png_lnmask.png" #"_tissuemask.png"
TISSUE_MASK_EXT="_tissuemask.png"

def patch_slides(wsi_path, ann_path,tissue_mask_path, save_path, classes=[],wsi_mask=True, wsi_mask_path="", filter_path=""):
    print("in patch_images")
    draw_contours=False
    wsi_paths=glob.glob(os.path.join(wsi_path,'*.svs'))
    annotations_paths=glob.glob(os.path.join(ann_path,'*.json'))
    #print(annotations_paths)    
    
    #presumably this is so that we can skip ones we have already done?
    saved=glob.glob(os.path.join(save_path,'*'))
    saved=[os.path.basename(s) for s in saved]
    print("saved: ",saved)   
    
    for curr_path in wsi_paths:
        full_name=os.path.basename(curr_path)
        name=full_name.replace('.ndpi','')
        name=name.replace('.svs','')
        name=name.replace('.mrxs','')
        
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
        print('slide',name)

        #the patch masks don't have an extension??
        curr_save_path=os.path.join(save_path,name)
        os.makedirs(curr_save_path,exist_ok=True)

        
        #retrieve all annotations for the specified classes
        if len(ann_path)==0:
            annotate=None
        else:
            annotate=Annotations(ann_path,source='qupath',labels=classes)

            if len(annotate._annotations)==0:
                print("**** WARNING - NO ANNOTATIONS ***")
        
        ## need to get border with sinuses as well
        wsi=Slide(curr_path,annotations=annotate)
 
        ## svs files don't necessarily have 6 levels
        lvl = min(6,wsi.level_count-1)

        ## WRITE MASK FOR WSI
        #if wsi_mask:
        #    mask=wsi.slide_mask           
        #    cv2.imwrite(os.path.join(wsi_mask_path,name+'_mask.png'),mask*255)


        ## READ TISSUE MASK
        # Ensure mask is read in grayscale
        tissue_mask_name = os.path.join(TISSUE_MASK_PATH, full_name +TISSUE_MASK_EXT) # ".png_lnmask.png") #_tissuemask.png for combined with histoqc
        print(tissue_mask_name)
        tissue_mask = cv2.imread(tissue_mask_name, cv2.IMREAD_GRAYSCALE)  

        # Since your mask is binary, it should already be in 0/255 format, but in case it's not:
        # Convert any non-zero values to 1 and then multiply by 255 to ensure binary mask is 0/255
        tissue_mask = np.where(tissue_mask > 0, 1, 0).astype(np.uint8) * 255

        #transpose the mask?
        print(tissue_mask.shape)
        
        #do not need to transpose mask for combined
        #tissue_mask = np.transpose(tissue_mask)
        #print(tissue_mask.shape)

        dims= wsi.level_dimensions[lvl]
        print("wsi dims:",dims)
        print("tissue mask dims:",tissue_mask.shape)

        #set this based on slide properties
        slide_mag = int(wsi.properties.get('openslide.objective-power'))
        print(slide_mag)

        ## RESIZE tissue mask
        #tissue_mask_scaled = cv2.resize(tissue_mask, (0, 0), fx=wsi.level_downsamples[lvl], fy=wsi.level_downsamples[lvl])
        tissue_mask_scaled = cv2.resize(tissue_mask, wsi.level_dimensions[0], interpolation=cv2.INTER_NEAREST)

        ## APPLY tissue mask to the WSI
        print("tissue_mask after scaling:",tissue_mask_scaled.shape)
        wsi.set_filter_mask(mask=tissue_mask_scaled)

        ### GET THE BORDER for LN to save based on ALL the LNs
        #set the padding to 2% of avg width and height of the region
        if(TESTIMS):
            # for test images there won't be any annotations so need to set a region in QuPath first
            print("testims true")
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
            ## CONTOUR ALL LNs
            contours=detect_tissue_section(wsi)

            # First, we need to combine all the contours into one array
            all_points = np.concatenate([cnt for cnt in contours], axis=0)

            # add some padding to create a border around the countours
            xmin, ymin, w, h = cv2.boundingRect(all_points)
            padding = int((w+h)/2*.02)
            xmin = max(0,xmin - padding)
            xmax = max(0,xmin + w + padding)
            ymin = max(0,ymin - padding)
            ymax = max(0,ymin + h + padding)
            border_mask = [[xmin,xmax],[ymin,ymax]]

            #contours are at level 6 (or min level) so we need to scale them up to level 0
            scale_factor =  wsi.level_downsamples[lvl] /wsi.level_downsamples[0] 
            #print(scale_factor)

            xmin = int(xmin * scale_factor)
            xmax = int(xmax * scale_factor)
            ymin = int(ymin * scale_factor)
            ymax = int(ymax * scale_factor)
            border = [[xmin,xmax],[ymin,ymax]]

            # CROP to a multiple of the step size so patches are equal size
            w=int(xmax-xmin)
            h=int(ymax-ymin)
            #print(w,h)
            crop_w_by = w%STEP  #(STEP*scale_factor)
            xmax = xmax - crop_w_by
            crop_h_by = h%STEP  #(STEP*scale_factor)
            ymax = ymax - crop_h_by
            #print(crop_w_by)
            #print(crop_h_by)

            # ASSEMBLE FINAL BORDER
            border = [[xmin,xmax],[ymin,ymax]]



        print(border)
        ### FEATURE TO MASK & SAVE
        MAG_LEVEL = wsi.get_best_level_for_downsample(MAG_DS+0.1) #have to add .1 for svs
        patches=Patch(wsi,mag_level=MAG_LEVEL,border=border,size=PATCH_SIZE)
 
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
        patches.save(save_path,mask_flag=True,remove_white_patches=True)
 




if __name__=='__main__':
    
    ap=argparse.ArgumentParser()
    ap.add_argument('-wp','--wsi_path', required=True,help='path to WSIs')
    ap.add_argument('-op','--output_path', required=True,help='path to WSIs')
    #ap.add_argument('-mp','--mask_path', required=True,help='path to tissue masks')
    args=ap.parse_args()

    classes=['GC','sinus']
    patch_slides(args.wsi_path, ANNOTATIONS_PATH, TISSUE_MASK_PATH,args.output_path, classes, wsi_mask=True, wsi_mask_path=WSI_MASK_PATH)



