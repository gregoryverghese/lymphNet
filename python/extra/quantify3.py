import os
import glob

import numpy as np

import cv2
import matplotlib.pyplot as plt
import openslide
import pandas as pd

import measure as me


def getFiles(filesPath, ext):
    filesLst=[]
    for path, subdirs, files in os.walk(filesPath):
        for name in files:
            if name.endswith(ext):
                filesLst.append(os.path.join(path,name))
    return filesLst


wsiPath='/SAN/colcc/WSI_LymphNodes_BreastCancer/HarryC/GUYS LN IMAGES NOV 2019 - FEB 2021'
maskPath1='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/testing/output/itt_1'
maskPath2='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/testing/output/itt_2'
plotPath='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/testing/output/quantify_plots'
totalImages=getFiles(wsiPath,'ndpi')
totalImages=getFiles(wsiPath,'ndpi')

print(len(totalImages))
#totalImages=[t for t in totalImages if '90495' in t]
totalMasks1=getFiles(maskPath1,'png')
totalMasks2=getFiles(maskPath2,'png')

totalMasks=totalMasks1+totalMasks2
print(len(totalMasks))
totalMasks=[t for t in totalMasks if 'image' not in t]
#totalMasks=[t for t in totalMasks if '90495' in t]
names=[]
lnIdx=[]
lnAreas=[]
germNum=[]
avgGermSizes=[]
avgGermAreas=[]
germTotalAreas2=[]
germTotalAreas=[]
sinusNum=[]
totalSinusArea=[]
avgGermW=[]
avgGermH=[]
centDist=[]
boundDist=[]
maxGerm=[]
minGerm=[]
shapes=[]

print('total number of images', len(totalImages))
print('total number of masks', len(totalMasks))

#totalMasks=[t for t in totalMasks if '90377' in t]
#print(totalMasks)
#totalMasks=totalMasks[0:1]
for maskF in totalMasks:
    name=os.path.basename(maskF)[:-4]
    print(name)
    #if '34.27.90513 C L1.3' not in name:
        #continue
    #wsiF=maskF[:-4]+'_image.png'
    try:
        wsiF=[s for s in totalImages if name in s][0]
    except Exception as e:
        continue
    #names.append(name)
    
    #image = cv2.imread(wsiF)
    mask = cv2.imread(maskF)
    wsi=openslide.OpenSlide(wsiF)
    dims=wsi.dimensions
    mdims=mask.shape

    image = wsi.get_thumbnail(size=(mdims[1],mdims[0]))
    image = np.array(image)
    #mask[:,:,0][mask[:,:,1]==255]=180
    mask=mask[:,:,0]
    #mask=mask[:,:,0]
    mShape=mask.shape
    iShape=image.shape
    #dims=dimensions.loc[dimensions['name']==name]
    #w=dims['w'].values
    #h=dims['w'].values
    #wnew=dims['wfinal'].values
    #hnew=dims['hfinal'].values
    w=dims[0]
    h=dims[1]
    wNew=mShape[0]
    hNew=mShape[1]
    slide = me.Slide(image,mask,w,h,wNew,hNew)
    num = slide.extractLymphNodes(255,128)
    #f,ax=plt.subplots(1,2,figsize=(15,15))
    #ax[0].imshow(mask,cmap='gray')
    #ax[0].axis('off')
    #ax[1].imshow(image,cmap='gray')
    #ax[1].axis('off')
    plt.show()
    #cv2.imwrite('plots/'+name+'_image.png',image)
    print('number of ln: {}'.format(num))

    for i, ln in enumerate(slide._lymphNodes):
        #names.append(name)

        #df=pd.DataFrame({'names':names})
        #df.to_csv('/home/verghese/names.csv')
        
        #f, ax = plt.subplots(1,2,figsize=(10,10))
        #ax[0].imshow(ln.mask)
        #ax[1].imshow(ln.image)
        #plt.show()
        mask=cv2.drawContours(ln.image,ln.contour,-1,(0,0,255),3)
        print(mask.shape)
        lnAreas.append(ln.area*1e6)
        cv2.imwrite(os.path.join(plotPath,name+str(i)+'_ln.png'),mask)
        numGerms=ln.germinals.detectGerminals()
        numSinuses=ln.sinuses.detectSinuses()
        #print('#Germinal:{}, #Sinus:{}'.format(numGerms,numSinuses))
        ln.germinals.measureSizes()
        ln.germinals.measureAreas()
        plotS=ln.sinuses.visualiseSinus()
        plotG=ln.germinals.visualiseGerminals()
        
        print('mask shape', mask.shape)
        binary_mask=np.zeros((mask.shape))
        binary_mask=cv2.fillPoly(binary_mask,pts=[ln.contour],color=(255,255,255))

        cv2.imwrite(os.path.join(plotPath,name+str(i)+'_binarymask.png'),binary_mask)
        #f,ax = plt.subplots(1,3,figsize=(15,25))
        #ax[0].imshow(ln.mask, cmap='gray')
        #ax[0].axis('off')
        #ax[1].imshow(plotG, cmap='gray')
        #ax[1].axis('off')
        #ax[2].imshow(plotS, cmap='gray')
        #ax[2].axis('off')
        #plt.show()
        
        #cv2.imwrite(os.path.join(plotPath,name+str(i)+'_sinus.png'),plotS)
        #cv2.imwrite(os.path.join(plotPath,name+str(i)+'_germs.png'),plotG)

        sizes=ln.germinals._sizes
        if sizes==[(0,0)]:
            avgSizes=[0,0]
        else:
            avgSizes=np.mean(list(zip(*sizes)),axis=1)
        areas=ln.germinals._areas
        if len(areas)==0:
            areas=[0]
        else:
            areas

        avgGermArea2=np.mean(areas)
        maxGermArea2=np.max(areas)
        minGermArea2=np.min(areas)
        germArea=ln.germinals.totalArea
        germArea2=ln.germinals.totalArea2
        sinusArea=ln.sinuses.totalArea2
        germDistCent=ln.germinals.distanceFromCenter()
        germDistBoundary=ln.germinals.distanceFromBoundary()
        germShape=np.mean(ln.germinals.circularity())
        names.append(name)
        lnIdx.append(i)
        germNum.append(numGerms)
        avgGermW.append(np.round(avgSizes[0]*1e6,2))
        avgGermH.append(np.round(avgSizes[1]*1e6,2))
        germTotalAreas.append(np.round(germArea*1e6,2))
        germTotalAreas2.append(np.round(germArea2*1e6,2))
        avgGermAreas.append(np.round(avgGermArea2*1e6,4))
        maxGerm.append(np.round(maxGermArea2*1e6,4))
        minGerm.append(np.round(minGermArea2*1e6,4))
        shapes.append(germShape)
        sinusNum.append(numSinuses)
        totalSinusArea.append(np.round(sinusArea*1e6,2))
        centDist.append(np.round(np.mean(germDistCent)))
        boundDist.append(np.round(np.mean(germDistBoundary)))


stats={
    'name':names,
    'ln_idx':lnIdx,
    'ln_area':lnAreas,
    'germ_number':germNum,
    'avg_germ_width':avgGermW,
    'avg_germ_height':avgGermH,
    #'total_germ_area':germTotalAreas,
    'total_germ_area2':germTotalAreas2,
    'avg_germ_area': avgGermAreas,
    'avg_germ_shape':shapes,
    'max_germ_area': maxGerm,
    'min_germ_area': minGerm,
    'germ_distance_to_centre':centDist,
    'germ_distance_to_boundary':boundDist,
    'sinus_number': sinusNum,
    'total_sinus_area':totalSinusArea

}

for k,v in stats.items():
    print(k, len(v))
statsDf=pd.DataFrame(stats)
#print(statsDf)
statsDf.to_csv('/home/verghese/shape.csv')

