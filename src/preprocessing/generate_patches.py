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
    test=[p_id for p_id in patches._patches if p_id['name'] in filtered]
    patches._patches=test
    print('g','num patches: {}'.format(len(patches._patches)))
    #patches.save(save_path,mask_flag=True)
    #patches.save(save_path,mask_flag=True)
    patches.save_mask(save_path, 'mask')


"""
from patching import Annotations,Slide,Patching,Stitching

#CODE FROM src/preprocessing/generate_patches.py
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


thumbnail_path='/home/verghese/thumbnails/*'
thumbnails=glob.glob(thumbnail_path)
thumbnails=[os.path.basename(t)[:-4] for t in thumbnails]

wsi_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/wsi/train'
annotations_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/lymphnode-keras/data/wsi/Guys/annotations/sum_ff_toms'
wsi_paths=glob.glob(os.path.join(wsi_path,'*'))
annotations_paths=glob.glob(os.path.join(annotations_path,'*'))

source_dict={'.xml':'imagej','.json':'qupath'}

print(len(wsi_paths))
for p in wsi_paths:
    name=os.path.basename(p)[:-5]
    if name in thumbnails:
        continue
    if name=="U_90420_5_X_LOW_1_L1":
        continue
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
