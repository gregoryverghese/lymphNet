import os
import glob
import argparse

import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from patching import Annotations,Slide,Patching,Stitching
#from utilities import mask2rgb

def mask2rgb(mask):
    n_classes=len(np.unique(mask))
    colors=sns.color_palette('hls',n_classes)
    rgb_mask=np.zeros(mask.shape+(3,))
    for c in range(1,n_classes):
        t=(mask==c)
        rgb_mask[:,:,0][t]=colors[c][0]
        rgb_mask[:,:,1][t]=colors[c][1]
        rgb_mask[:,:,2][t]=colors[c][2]
    return rgb_mask

STEP=128
MAG_LEVEL=4
SIZE=(512,512)

WSI_MASK_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-torch/data/masks'
GERMINAL_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-torch/data/patches/2.5x-512-f/train'
SINUS_PATH='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-torch/data/patches/2.5x-256'

wsi_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/wsi/train'
annotations_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/annotations/sum_ff_toms'
wsi_paths=glob.glob(os.path.join(wsi_path,'*'))
annotations_paths=glob.glob(os.path.join(annotations_path,'*'))

source_dict={'.xml':'imagej','.json':'qupath'}

print(len(wsi_paths))
for p in wsi_paths:
    name=os.path.basename(p)[:-5]
    print('slide',name)
    ann_paths=[a for a in annotations_paths if name in a]
    annotations=Annotations()
    for a in ann_paths:
        ext=os.path.splitext(a)[-1]
        source=source_dict[ext]
        print(source)
        annotations._generate_annotations(a,source)
        #annotations.filter_labels(['GC', 'sinus', 'GERMINAL CENTRE', 'SINUS'])
        annotations.encode_keys()
    
    ann=annotations._annotations
    if len(list(ann.keys()))==0:
        continue

    test=[]
    for k,v in ann.items():
        test=test+ann[k]
    if len(test)==0:
        continue

    wsi=Slide(p,annotations=annotations)
    borders=wsi.get_border()
    (x1,x2),(y1,y2)=borders
    print(borders)
    region=wsi.generate_region(mag=3,x=(x1,x2),y=(y1,y2))
    #region=region.convert('RGB')
    cv2.imwrite(os.path.join('/home/verghese/thumbnails',name+'.png'),region[0])
            #continue
    #img,borders=wsi.detect_component()
    """
    region=wsi.generate_region(mag=3,x=(x1,x2),y=(y1,y2))
    cv2.imwrite(os.path.join('/home/verghese/thumbnails',name+'.png'),region)
    
    #mask=wsi.slide_mask
    #plt.imshow(mask)
    #cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'.png'),mask)
    
    ################Get germinal centres#####################
    
    ann_obj=Annotations(ann_path,source=['imagej','qupath'])
    ann_obj.generate_annotations()
    ann_obj.filter_labels(['GC', 'GERMINAL CENTRE'])
    annotations=ann_obj._annotations
    new_annotations=annotations
    
        
    test=[]
    for k,v in annotations.items():
        test=test+annotations[k]
    if len(test)==0:
        continue

    if len(ann_path)==2:
        new_annotations={}
        new_annotations['germinals']=annotations['GC']+annotations['GERMINAL CENTRE']
    new_annotations={i: v for i, v in enumerate(new_annotations.values())}
    wsi_germs=Slide(p,annotations=new_annotations,draw_border=True)
    wsi_germs.get_border()
    patches=Patching(wsi_germs,mag_level=MAG_LEVEL,size=SIZE)
    num=patches.generate_patches(STEP, mask_flag=True,mode='focus')
    print('num patches: {}'.format(num))
    patches.save(GERMINAL_PATH,mask_flag=True)
    
    #################=Get sinuses###############
    
    ann_obj=Annotations(ann_path,source=['imagej','qupath'])
    ann_obj.generate_annotations()
    ann_obj.filter_labels(['sinus', 'SINUS'])
    annotations=ann_obj._annotations
    new_annotations=annotations
    if len(ann_path)==2:
        new_annotations={}
        new_annotations['sinus']=annotations['sinus']+annotations['SINUS']
    new_annotations={i: v for i, v in enumerate(new_annotations.values())}
    wsi_sinus=Slide(p,annotations=new_annotations,draw_border=True)
    print(new_annotations)
    wsi_sinus.get_border()
    print("border",wsi_sinus._border)
    patches=Patching(wsi_sinus,mag_level=MAG_LEVEL,size=SIZE)
    num=patches.generate_patches(STEP, mask_flag=True)
    print('num patches: {}'.format(num))
    patches.save(SINUS_PATH,mask_flag=True)

    #s=Stitching(os.path.join(PATCH_PATH,'masks'),mag_level=MAG_LEVEL,name=name)
    #canvas=s.stitch()
    #canvas=cv2.resize(canvas,(2000,2000))
    #cv2.imwrite(os.path.join(WSI_MASK_PATH,name+'_stitch.png'),canvas)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-cp','--configpath', required=True,help='path to config
                    file')
    args=vars(ap.parse_args())
"""
