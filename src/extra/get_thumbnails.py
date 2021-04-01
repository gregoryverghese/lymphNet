import glob
import os

import numpy as np
import cv2
import pandas as pd
import openslide

wsi_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/wsi/guys_ln'
image_path='filenames.csv'
out_path='thumbnail_folder'

images=pd.read_csv(image_path)
check=list(images['Names'])

print('check',len(check))

total_images=[]

for path, subdirs, files in os.walk(wsi_path):
    for name in files:
        if name.endswith('ndpi'):
            total_images.append(os.path.join(path,name))

total_images=[i for i in total_images if any([j for j in check if j in i])]

print(len(total_images))

for i in total_images:
    print(i)
    name=os.path.basename(i)[:-4]
    slide=openslide.OpenSlide(i)
    thumbnail=slide.get_thumbnail(size=(2000,2000))
    thumbnail=np.array(thumbnail.convert('RGB'))
    cv2.imwrite(os.path.join(out_path,name+'_.png'),thumbnail)
