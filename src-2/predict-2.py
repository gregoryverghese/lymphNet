import os
import glob

import numpy as np
import pandas as pd



def patching(step,xmin,xmax,ymin,ymax):
    """
    step across coordinate range
    """
    for x in range(xmin,xmax, step):
        for y in range(ymin,ymax,step):
            yield x, y


class Predict()
    def __init__(self,model,theshold):
        self.model=model
        self.threshold=threshold


    def _predict():
        canvas=np.zeros((int(ydim_new),int(xdim_new),3))
        for x, y in patching(step,xmin,xmax,ymin,ymax):
            patch=image[x:x+step,y:y+step]
            probs=self.model.predict(patch)
            pred=tf.cast((probs>self.threshold), tf.float32)
            canvas[x:x+step,y:y+step,:]=pred
    
    return canvas.astype(np.uint8)
