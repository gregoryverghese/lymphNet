import cv2
import os
import numpy as np


def mergePatches(ndpi, boundaries, patchPath, modelName, magFactor=1, outPath='prediction/whole, 'imgSize=1024):

    name = os.path.basename(ndpi)[:-5]
    m = ((boundaries[0][1] - boundaries[0][0])//(imgSize*magFactor)*imgSize*magFactor)

    for h in range(boundaries[1][0], boundaries[1][1], 1024*16):
        for w in range(boundaries[0][0], boundaries[0][1], 1024*16):
    
            predPath = os.path.join(predsPath, modelName, name, 'patches',name +'_'+str(w)+'_'+str(h)+'_pred.png')
            patch = cv2.imread(predPath)
            patchNew = cv2.resize(patch, (500,500))

            if w == boundaries[0][0]:
                image = patchNew
            else:
                image = np.hstack((image, patchNew))

        if (w, h) == (boundaries[0][0]+m, boundaries[1][0]):
            final = image
        else:
            final = np.vstack((final, image))

    cv2.imwrite(os.path.join(outPath, modelname, name+'_pred.png'), final)
