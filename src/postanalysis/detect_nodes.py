import os
import glob

import cv2
import numpy as np
import openslide
import matplotlib.pyplot as plt

BILATERAL1_ARGS={"d":9,"sigmaColor":10000,"sigmaSpace":150}
BILATERAL2_ARGS={"d":90,"sigmaColor":5000,"sigmaSpace":5000}
BILATERAL3_ARGS={"d":90,"sigmaColor":10000,"sigmaSpace":10000}
BILATERAL4_ARGS={"d":90,"sigmaColor":10000,"sigmaSpace":100}
THRESH1_ARGS={"thresh":0,"maxval":255,"type":cv2.THRESH_TRUNC+cv2.THRESH_OTSU}
THRESH2_ARGS={"thresh":0,"maxval":255,"type":cv2.THRESH_OTSU}

outpath='/home/verghese'
wsi_paths='/SAN/colcc/WSI_LymphNodes_BreastCancer/wsi/guys_ln/32.90577/*'
wsi_paths=glob.glob(wsi_paths)

#print('analysing lymph nodes...',flush=True)
#totalImages=getFiles(wsiPath,'ndpi')
#print('Image N: {}'.format(len(totalImages),flush=True))
#totalMasks=getFiles(maskPath,'png')
#print('Mask N: {}'.format(len(totalMasks),flush=True))
#totalMasks=[t for t in totalMasks if 'image' not in t]

LN_count={}
num=0
for path in wsi_paths :

    patient_id=path.split("/")[-2]
    os.makedirs(os.path.join(outpath,patient_id),exist_ok=True)
    name=os.path.basename(path)[:-5]
    print('patient: {}, slide: {}'.format(name,patient_id))
    wsi=openslide.OpenSlide(path)
    img=np.array(wsi.get_thumbnail((2000,2000)))
    # color filter
    image=img.copy()
    shape=img.shape

    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    lower_red=np.array([120,0,0])
    upper_red=np.array([180,255,255])
    mask=cv2.inRange(img_hsv,lower_red,upper_red)
    img_hsv=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
    m=cv2.bitwise_and(img,img,mask=mask)
    im_fill=np.where(m==0,233,m)

    #convert to grayscale
    mask=np.zeros(shape)
    gray=cv2.cvtColor(im_fill,cv2.COLOR_BGR2GRAY)
    #generate the blur
    blur1=cv2.bilateralFilter(np.bitwise_not(gray),**BILATERAL1_ARGS)
    #step2: make the pixeldist and sigma space larger so that the content can be linked together
    blur2=cv2.bilateralFilter(np.bitwise_not(blur1),**BILATERAL2_ARGS)
    #step3: make each lymph node looks mor like a group
    blur3=cv2.bilateralFilter(np.bitwise_not(blur2),**BILATERAL3_ARGS)
    #step4: contain more color as possible
    blur4=cv2.bilateralFilter(np.bitwise_not(blur3),**BILATERAL4_ARGS)
    blur_final=255-blur4
    #threshold twice
    _,thresh=cv2.threshold(blur_final,**THRESH1_ARGS)
    _,thresh=cv2.threshold(thresh,**THRESH2_ARGS)
    #find contours
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=list(filter(lambda x: cv2.contourArea(x) > 9000, contours))
    new=cv2.drawContours(image,contours,-1,(0,0,255),3)
    outfile=os.path.join(outpath,patient_id,name+"_contour.png")
    print("number is ", len(contours))
    num+=len(contours)
    print("running",num)
print("final ...",str(num))
    #cv2.imwrite(outfile,new)
