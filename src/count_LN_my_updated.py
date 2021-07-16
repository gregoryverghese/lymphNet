import os,sys
import glob
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import gc
def getFiles(filesPath, ext):
    filesLst=[]
    for path, subdirs, files in os.walk(filesPath,followlinks=True):
        for name in files:
            if name.endswith(ext):
                filesLst.append(os.path.join(path,name))
    return filesLst
patient_index=sys.argv[1]
thumbnail_path=os.path.join("/SAN/colcc/WSI_LymphNodes_BreastCancer/Mengyuan/data/LNs/thumbnails",patient_index)
outpath="/SAN/colcc/WSI_LymphNodes_BreastCancer/Mengyuan/try/LN_compare/LN_count_my_04-Jun"
#path="/Volumes/ESD-USB/KCL_PHD/project/LN_tidy/Guys/thumbnail/01.90524/1.13.90524 C L2.51.ndpi.png"
LN_count={}
for path in getFiles(thumbnail_path,"png"):
    # read file
    patient_index=path.split("/")[-2]
    print(patient_index)
    try:
        os.mkdir(os.path.join(outpath,patient_index))
    except:
        print("")
    name=path.split("/")[-1].replace(".ndpi.png","")
    print(name)

    # color filter
    img=cv2.imread(path)
    image=img.copy()
    shape=img.shape
    plt.imshow(img)
    a=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    lower_red1=np.array([120,0,0])
    upper_red1=np.array([180,255,255])
    mask=cv2.inRange(a,lower_red1,upper_red1)
    a=cv2.cvtColor(a,cv2.COLOR_HSV2RGB)
    #print(a)
    m=cv2.bitwise_and(img,img,mask=mask)
    #print(m)
    #a=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)


    #fill with background color

    im_in=m
    im_fill=im_in.copy()
    h,w=im_in.shape[:2]
    ref_color=[233,233,233]
    for i in range(im_fill.shape[0]):
        for j in range(im_fill.shape[1]):
            if(im_fill[i][j][0]==0 and im_fill[i][j][1]==0):
                im_fill[i][j]=ref_color


    #prepare a mask with the same size of the image
    mask=np.zeros(shape)

    #convert to grayscale
    gray=cv2.cvtColor(im_fill,cv2.COLOR_BGR2GRAY)

    #gernerate the blur
        #step1: contain as much color as possible
    pixelDist=9
    sigmaColor=10000
    sigmaSpace=150
    blur=cv2.bilateralFilter(np.bitwise_not(gray),pixelDist,sigmaSpace,sigmaColor)

        #step2: make the pixeldist and sigma space larger so that the content can be linked together 
    pixelDist=90
    sigmaColor=5000
    sigmaSpace=5000
    blur1=cv2.bilateralFilter(np.bitwise_not(blur),pixelDist,sigmaSpace,sigmaColor)

        #step3: make each lymph node looks mor like a group
    pixelDist=90
    sigmaColor=10000
    sigmaSpace=10000
    blur2=cv2.bilateralFilter(np.bitwise_not(blur1),pixelDist,sigmaSpace,sigmaColor)

        #step4: contain more color as possible
    pixelDist=90
    sigmaColor=10000
    sigmaSpace=100
    blur3=cv2.bilateralFilter(np.bitwise_not(blur2),pixelDist,sigmaSpace,sigmaColor)

    blur_final=255-blur3

    #gerneate thresh
    minThresh=0
    maxThresh=255
    threshType=cv2.THRESH_TRUNC+cv2.THRESH_OTSU
    _,thresh=cv2.threshold(blur_final,minThresh,maxThresh,threshType)

    #gerneate thresh
    minThresh=0
    maxThresh=255
    threshType=cv2.THRESH_OTSU
    _,thresh=cv2.threshold(thresh,minThresh,maxThresh,threshType)


    #find contours
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=list(filter(lambda x: cv2.contourArea(x) > 9000, contours))

    print(len(contours))
    LN_count[name]=[len(contours),ref_color]

    for i in contours:
        new=cv2.drawContours(image,contours,-1,(0,0,255),3)
        
    outfile=os.path.join(outpath,patient_index,name+"_contour.png")
    cv2.imwrite(outfile,new)
    del blur,thresh,contours,image
    gc.collect()


outdict=os.path.join(outpath,"count",patient_index+"_LN_count.npy")
np.save(outdict,LN_count)
print(LN_count)
