'''
utilities.py: useful functions
'''
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import seaborn as sns
from itertools import chain


def mask2rgb(mask):
    n_classes=len(np.unique(mask))
    colors=sns.color_palette('hls',n_classes)
    rgb_mask=np.zeros(mask.shape+(3,))
    for c in range(1,n_classes+1):
        t=(mask==c)
        rgb_mask[:,:,0][t]=colors[c-1][0]
        rgb_mask[:,:,1][t]=colors[c-1][1]
        rgb_mask[:,:,2][t]=colors[c-1][2]
    return rgb_mask


def draw_boundary(annotations, offset=100):

    annotations = list(chain(*[annotations[f] for f in annotations]))
    coords = list(chain(*annotations))
    boundaries = list(map(lambda x: (min(x)-offset, max(x)+offset), list(zip(*coords))))
   
    return boundaries


def oneHotToMask(onehot):
    nClasses =  onehot.shape[-1]
    idx = tf.argmax(onehot, axis=-1)
    colors = sns.color_palette('hls', nClasses)
    multimask = tf.gather(colors, idx)
    multimask = np.where(multimask[:,:,:]==colors[0], 0, multimask[:,:,:])

    return multimask


#can we sample and return a new patching object
def sample_patches(patch,n,replacement=False):
    
    if replacement:
        patches=random.choice(patch._patches,n)
    else:
        patches=random.sample(patch._patches,n)

    new_patch =  Patch(patch.slide,
                       patch.size,
                       patch.mag_level,
                       patch.border,  
                       patch.step)

    new_patch.patches=patches
    return new_patches


def detect_tissue_section(slide):

    bilateral1_args={"d":9,"sigmaColor":10000,"sigmaSpace":150}
    bilateral2_args={"d":90,"sigmaColor":5000,"sigmaSpace":5000}
    bilateral3_args={"d":90,"sigmaColor":10000,"sigmaSpace":10000}
    bilateral4_args={"d":90,"sigmaColor":10000,"sigmaSpace":100}
    thresh1_args={"thresh":0,"maxval":255,"type":cv2.THRESH_TRUNC+cv2.THRESH_OTSU}
    thresh2_args={"thresh":0,"maxval":255,"type":cv2.THRESH_OTSU}

    slide=slide.get_thumbnail(slide.level_dimensions[6])
    slide=np.array(slide.convert('RGB'))
    img_hsv=cv2.cvtColor(slide,cv2.COLOR_RGB2HSV)
    lower_red=np.array([120,0,0])
    upper_red=np.array([180,255,255])
    mask=cv2.inRange(img_hsv,lower_red,upper_red)
    img_hsv=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
    m=cv2.bitwise_and(slide,slide,mask=mask)
    im_fill=np.where(m==0,233,m)
    mask=np.zeros(slide.shape)
    gray=cv2.cvtColor(im_fill,cv2.COLOR_BGR2GRAY)
    blur1=cv2.bilateralFilter(np.bitwise_not(gray),**bilateral1_args)
    blur2=cv2.bilateralFilter(np.bitwise_not(blur1),**bilateral2_args)
    blur3=cv2.bilateralFilter(np.bitwise_not(blur2),**bilateral3_args)
    blur4=cv2.bilateralFilter(np.bitwise_not(blur3),**bilateral4_args)
    blur_final=255-blur4

    _,thresh=cv2.threshold(blur_final,**thresh1_args)
    _,thresh=cv2.threshold(thresh,**thresh2_args)

    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=list(filter(lambda x: cv2.contourArea(x) > 4000, contours))
    return contours


def match_annotations_to_tissue_contour(
        contours,
        annotations,
        ds
        ):
    target=None
    for c in contours:
        for p in annotations:
            p=(int(p[0]/ds),int(p[1]/ds))
            if cv2.pointPolygonTest(c, p, False)==1: 
                target=1
                break
        if target==1:
            break
    return c

