import numpy as np
import glob
import os
import cv2

maskPath  = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/data/masks'
basePath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation'



classes = glob.glob(os.path.join(maskPath, '*'))
step = np.floor(255/len(classes))
numClass = [(yield c+1) for c in range(len(classes))]

for c in classes:
    maskFiles = glob.glob(os.path.join(c, '*'))
    num = next(numClass)
    print(len(maskFiles))
    for m in maskFiles:
        
        mask = cv2.imread(m)
        mask[mask==255]=num*step
        name = m[len(c)+1:]

        print('here we go', os.path.join(basePath, 'masks', name))
        cv2.imwrite(os.path.join(basePath, 'masks', name), mask)
