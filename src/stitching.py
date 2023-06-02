import cv2
import numpy as np
import matplotlib.pyplot as plt


class Canvas():
    def __init__(self,x_dim,y_dim):
        #only needs 1 channel as it is a binary mask
        canvas=np.zeros((1,int(x_dim),int(y_dim),1))
        self.canvas=canvas.astype(np.uint8)
    
    #@property
    #def canvas(self):
        #return self._canvas
       
    #@canvas.setter
    def stitch(self,m,x,y):
        self.canvas[:,x:x+m.shape[1],y:y+m.shape[2],:]=m
        #existing=self.canvas[:,x:x+m.shape[1],y:y+m.shape[2],:]
        #self.canvas[:,x:x+m.shape[1],y:y+m.shape[2],:]=np.maximum(m,existing)

def stitch(canvas, mask, x, y, h, w, t_dim, step, margin):
    #Top left
    if (y==0) and (x==0):
        m=mask[:,0:margin+step,0:margin+step,:]
        canvas.stitch(m,x,y)
    #Top right
    elif (y==h-t_dim) and (x==0):
        m=mask[:,0:margin+step,margin:t_dim,:]
        canvas.stitch(m,x,y+margin)
    #lower left
    elif (y==0 and x==w-t_dim):
        m=mask[:,margin:t_dim,0:margin+step,:]
        canvas.stitch(m,x+margin,y)
    #lower right
    elif (y==h-t_dim) and (x==w-t_dim):
        m=mask[:,margin:t_dim,margin:t_dim,:]
        canvas.stitch(m,x+margin,y+margin)
    #left
    elif y==0:
        m=mask[:,margin:margin+step,0:margin+step,:]
        canvas.stitch(m,x+margin,y)
    #right
    elif y==h-t_dim:
        m=mask[:,margin:margin+step,margin:t_dim,:]
        #print(m.shape)
        #print(margin)
        canvas.stitch(m,x+margin,y+margin)
    #top
    elif x==0:
        m=mask[:,0:margin+step,margin:margin+step,:]
        canvas.stitch(m,x,y+margin)
    #bottom
    elif x==w-t_dim:
        m=mask[:,margin:t_dim,margin:margin+step,:]
        canvas.stitch(m,x+margin,y+margin)
    #middle
    else:
        m=mask[:,margin:margin+step,margin:margin+step,:]
        canvas.stitch(m,x+margin,y+margin)
    return canvas


