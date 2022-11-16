import os
import glob

import cv2
import itertools
import pandas as pd

from pyslide.slide import Slide
from pyslide.patching import Patching

MAG_LEVEL=1
SIZE=(224,224)
STEP=224

save_path='/SAN/colcc/Hormad1/data/patches/train/20x'
out_paths= '/home/verghese/hormad1/thumbnails'
tnbc_patients_path='/SAN/colcc/Hormad1/data/tnbc-primary-tumour/genomic-data/hormad1_tumour_wsi_genomics_v1_adj.csv'
tnbc_patients=pd.read_csv(tnbc_patients_path)
tnbc_patients=list(tnbc_patients['image_path'])
#enriched_patients=tnbc_patients[tnbc_patients['Sensory']=='Enriched']
#depleted_patients=tnbc_patients[tnbc_patients['Sensory']=='Depleted']
#enriched_paths=list(enriched_patients['image_path'])
#depleted_paths=list(depleted_patients['image_path'])

#print('enriched: n={}'.format(len(enriched_paths)))
#print('depleted: n={}'.format(len(depleted_paths)))
print('enriched: n={}'.format(len(tnbc_patients)))

folders=glob.glob(os.path.join(save_path,'*'))
folders=[os.path.basename(p) for p in folders]

for i, p in enumerate(tnbc_patients):
    print(p)
    name=os.path.basename(p)[:-5]
    if '48-17-90296 TUM' not in name:
        continue
    #if name in folders:
        #print('continuing')
        #continue
    wsi=Slide(p)
    components,borders= wsi.detect_components(num_component=1)
    #cv2.imwrite(os.path.join('/home/verghese/hormad1/thumbnails',name+'.png'),components[0])
    wsi._border=borders[0]
    print('border',borders[0])
    f=lambda x: wsi.resize_border(x,STEP)
    border=list(map(f,list(itertools.chain(*wsi._border))))
    border=iter(border)
    border=list(zip(border,border))
    patch=Patching(wsi,mag_level=MAG_LEVEL,size=SIZE,step=STEP)
    #patch.filter_patches(threshold=210)
    #patch.sample_patches(n=5000)
    os.makedirs(os.path.join(save_path,name),exist_ok=True)
    patch.save(path=os.path.join(save_path,name))

"""
for i, p in enumerate(depleted_paths):
   name=os.path.basename(p)
    print(name)
    wsi=Slide(p)
    components,borders= wsi.detect_components(num_component=1)
    #cv2.imwrite(os.path.join('/home/verghese/hormad1/thumbnails',name+'.png'),components[0])
    print(borders[0])
    wsi._border=borders[0]
    patch=Patching(wsi,size=(224,224),step=224)
    patch.sample_patches(n=5000)
    patch.save(path=os.path.join(save_path,'depleted'))
"""
