import os
import glob
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import openslide
import pandas as pd

import measure as me
#from src.utilities.utils import getFiles

def getFiles(filesPath, ext):
    filesLst=[]
    for path, subdirs, files in os.walk(filesPath):
        for name in files:
            if name.endswith(ext):
                filesLst.append(os.path.join(path,name))
    return filesLst


def analyseNodes(wsiPath,maskPath,savePath):
    cancerPts='/home/verghese/cancer-points'
    print(maskPath)
    print('analysing lymph nodes...',flush=True)
    totalImages=getFiles(wsiPath,'ndpi')
    print('Image N: {}'.format(len(totalImages),flush=True))
    totalMasks=getFiles(maskPath,'png')
    print('Mask N: {}'.format(len(totalMasks),flush=True))
    cancerPoints=getFiles(cancerPts,'.csv')
    all_ln_status=pd.read_csv('/home/verghese/ln_status.csv',index_col=['image_name'])

    totalMasks=[t for t in totalMasks if 'image' not in t]
    names=[]
    lnIdx=[]
    lnAreas=[]
    germNum=[]
    avgGermSizes=[]
    avgGermAreas=[]
    avgGermAreas2=[]
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
    statusLst=[]
    subcapsular=[]

    print('total number of images', len(totalImages))
    print('total number of masks', len(totalMasks))

    for maskF in totalMasks:
        name=os.path.basename(maskF)[:-4]
        #if '100250' not in name:
            #continue
        print('image name: {}'.format(name))
        try:
            wsiF=[s for s in totalImages if name in s][0]
        except Exception as e:
            continue
        
        try:
            ln_status=int(all_ln_status.loc[name]['ln_status'])
        except Exception as e:
            continue
        if ln_status==0:
            cancerCoords=[]
        elif ln_status==1:
            point_files=[pt for pt in cancerPoints if name in pt]
            print(point_files)
            if len(point_files)>0:
                point_file=point_files[0]
                points=pd.read_csv(point_file,names=['probability','x','y'])
                points=points[points['probability']>0.85]
                cancerCoords=list(zip(list(points['x']),list(points['y'])))
            else:
                cancerCoords=[]

        mask = cv2.imread(maskF)
        wsi=openslide.OpenSlide(wsiF)
        dims=wsi.dimensions
        mdims=mask.shape

        #image = wsi.get_thumbnail(size=(mdims[1],mdims[0]))
        image=wsi.get_thumbnail(size=wsi.level_dimensions[6])
        mx,my=wsi.level_dimensions[6]
        mask=cv2.resize(mask,(mx,my))
        #mask=mask[:,:,0]

        image = np.array(image)
        mdims=image.shape
        mask[:,:,0][mask[:,:,0]==128]=0
        mask[:,:,1][mask[:,:,1]==255]=0
        mask[:,:,2][mask[:,:,2]==128]=0
        mask[:,:,2][mask[:,:,2]==255]=0
        
        print('unique 1',np.unique(mask[:,:,0]))
        print('unique 2',np.unique(mask[:,:,1]))
        #mask=cv2.resize(mask,(mdims[1],mdims[0]))
        print('unique 3',np.unique(mask[:,:,0]))
        print('unique 4',np.unique(mask[:,:,1]))
        mask[:,:,0][mask[:,:,0]!=0]=255
        mask[:,:,0][mask[:,:,1]!=0]=128
        print('unique 5',np.unique(mask[:,:,0]))
        mask=mask[:,:,0]
        print('unique 6',np.unique(mask))

        mShape=mask.shape
        iShape=image.shape
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
        #plt.show()
        #cv2.imwrite('plots/'+name+'_image.png',image)
        print('number of ln: {}'.format(num))

        for i, ln in enumerate(slide._lymphNodes):
            #####checking cancer section#######
            slide_contour=slide.contours[i]
            print('lengthhhhhhhhhh',len(cancerCoords))
            if len(cancerCoords)>0:
                check=[cv2.pointPolygonTest(slide_contour,pt,False) for pt in cancerCoords]
                check=list(filter(lambda x: x>0,check))
                if len(check)>0:
                    status="involved"
                else:
                    status="cf"
            elif len(cancerCoords)==0:
                if ln_status==0:
                    status='ln-neg'
                elif ln_status==1:
                    status='cf'


            mask=cv2.drawContours(ln.image,ln.contour,-1,(0,0,255),3)
            lnAreas.append(ln.area*1e6)
            cv2.imwrite(os.path.join(savePath,name+str(i)+'_ln.png'),mask)
            numGerms=ln.germinals.detectGerminals()
            numSinuses=ln.sinuses.detectSinuses()
            ln.germinals.measureSizes()
            ln.germinals.measureAreas()
            #plotS=ln.sinuses.visualiseSinus()
            #plotG=ln.germinals.visualiseGerminals()

            sinus_mask=ln.sinuses.sinusMask
            ann_mask=ln.sinuses.annMask

            ################## measure subcapsular sinus ################
            
            pixelDist=9
            sigmaColor=100
            sigmaSpace=100
            minThresh=0
            maxThresh=255
            #threshType=cv2.THRESH_BINARY+cv2.THRESH_OTSU
            threshType=cv2.THRESH_TRUNC+cv2.THRESH_OTSU
            slide_image=ln.sinuses.ln.image
            
            BILATERAL1_ARGS={"d":9,"sigmaColor":100,"sigmaSpace":100}
            BILATERAL2_ARGS={"d":90,"sigmaColor":10,"sigmaSpace":100}
            THRESH1_ARGS={"thresh":0,"maxval":255,"type":cv2.THRESH_TRUNC+cv2.THRESH_OTSU}
            THRESH2_ARGS={"thresh":0,"maxval":255,"type":cv2.THRESH_OTSU}

            gray=cv2.cvtColor(slide_image,cv2.COLOR_BGR2GRAY)
            blur1=cv2.bilateralFilter(np.bitwise_not(gray),**BILATERAL1_ARGS)
            blur2=cv2.bilateralFilter(np.bitwise_not(blur1),**BILATERAL2_ARGS)
            blur_final=255-blur2
            #plt.imshow(blur_final)
            #plt.show()
            _,thresh=cv2.threshold(blur_final,minThresh,maxThresh,threshType)
            _,thresh=cv2.threshold(thresh,**THRESH2_ARGS)
            #plt.imshow(thresh)
            #plt.show()
            contours_2,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            print(len(contours_2))
            contours_2 = list(filter(lambda x: cv2.contourArea(x) > 5000, contours_2))
            new=cv2.drawContours(slide_image,contours_2,-1,(0,0,255),2)
            #plt.figure(figsize=(20,20))
            #plt.imshow(new)
            #plt.show()
            
            test=np.zeros_like(slide_image)
            cv2.drawContours(test, contours_2, -1, (255,0,0), -1)

            kernel=np.ones((5,5),np.uint)
            erode=cv2.erode(test,kernel,(-1,-1),iterations=6)
            #plt.imshow(erode)
            #plt.show()
            erode_2=erode.copy()
            erode_2[:,:,0][erode[:,:,2]==255]=255
            erode_2[:,:,2][erode[:,:,2]==255]=0
            #plt.imshow(erode_2)
            np.unique(erode_2[:,:,2])
            new=test.copy()
            new[:,:,0][erode_2[:,:,0]==255]=255
            new[:,:,1][erode_2[:,:,0]==255]=255
            new[:,:,2][erode_2[:,:,0]==255]=255

            plt.figure(figsize=(10,10))
            plt.imshow(new)
            plt.show() 

            slide_image[:,:,0][erode_2[:,:,0]==255]=255
            slide_image[:,:,1][erode_2[:,:,0]==255]=255
            slide_image[:,:,2][erode_2[:,:,0]==255]=255
            #plt.figure(figsize=(20,20))
            #plt.imshow_(slide_image)
            #plt.show()
            
            mask_sinus=ln.sinuses.ln.mask
            mask_sinus[mask_sinus!=0]=128
            #plt.imshow(mask_sinus)
            #plt.show()
            print(mask_sinus.shape)
            print('values',np.unique(mask_sinus))

            mask_sinus[erode_2[:,:,0]==255]=255
            #plt.figure(figsize=(10,10))
            #plt.imshow(mask_sinus)
            subcapsular_area=len(mask_sinus[mask_sinus==128])*(ln.slide.wScale*ln.slide.hScale)
            
            print(mask_sinus[mask_sinus==128])
            
            print('checking shape',mask_sinus.shape)
            print('checking shape',len(mask_sinus))
            
            ##############################################################

            sinus_mask=ln.sinuses.sinusMask
            germinal_mask=ln.germinals.mask
            binary_mask=np.zeros((mask.shape))
            binary_mask=cv2.fillPoly(binary_mask,pts=[ln.contour],color=(255,255,255))

            germinal_mask = germinal_mask[:,:,None]*np.ones(3, dtype=int)[None,None,:]
            sinus_mask = sinus_mask[:,:,None]*np.ones(3, dtype=int)[None,None,:]

            binary_mask[germinal_mask==255]=0
            germinal_mask[:,:,0]=0
            germinal_mask[:,:,2]=0

            sinus_mask[sinus_mask==128]=255
            binary_mask[sinus_mask==255]=0
            sinus_mask[:,:,0]=0
            sinus_mask[:,:,1]=0

            binary_mask[:,:,1]=0
            binary_mask[:,:,2]=0

            print('b',np.unique(binary_mask[:,:,1]))
            print('g',np.unique(binary_mask[:,:,2]))
            binary_mask=binary_mask+germinal_mask+sinus_mask
            print(np.unique(binary_mask))
            cv2.imwrite(os.path.join(savePath,name+str(i)+'_binarymask.png'),binary_mask)
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

            if ln.germinals._num==0:
                avgGermArea=0
            else:
                avgGermArea=ln.germinals.totalArea2/ln.germinals._num

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
            statusLst.append(status)
            germNum.append(numGerms)
            avgGermW.append(np.round(avgSizes[0]*1e6,2))
            avgGermH.append(np.round(avgSizes[1]*1e6,2))
            germTotalAreas.append(np.round(germArea*1e6,2))
            germTotalAreas2.append(np.round(germArea2*1e6,2))
            avgGermAreas.append(np.round(avgGermArea2*1e6,4))
            avgGermAreas2.append(np.round(avgGermArea*1e6,4))
            maxGerm.append(np.round(maxGermArea2*1e6,4))
            minGerm.append(np.round(minGermArea2*1e6,4))
            shapes.append(germShape)
            sinusNum.append(numSinuses)
            totalSinusArea.append(np.round(sinusArea*1e6,2))
            centDist.append(np.round(np.mean(germDistCent)))
            boundDist.append(np.round(np.mean(germDistBoundary)))
            subcapsular.append(subcapsular_area)


    stats={
        'name':names,
        'ln_idx':lnIdx,
        'ln_area':lnAreas,
        'germ_number':germNum,
        'avg_germ_width':avgGermW,
        'avg_germ_height':avgGermH,
        #'total_germ_area':germTotalAreas,
        'total_germ_area2':germTotalAreas2,
        'ln_status':statusLst,
        'avg_germ_area': avgGermAreas,
        'avg_germ_area2': avgGermAreas2,
        'avg_germ_shape':shapes,
        'max_germ_area': maxGerm,
        'min_germ_area': minGerm,
        'germ_distance_to_centre':centDist,
        'germ_distance_to_boundary':boundDist,
        'sinus_number': sinusNum,
        'total_sinus_area':totalSinusArea,
        'subcapsularArea': subcapsular

    }

    for k,v in stats.items():
        print(k, len(v))
        statsDf=pd.DataFrame(stats)
    statsDf.to_csv('/home/verghese/node_details_subcapsular.csv')



if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-wp','--wsipath',required=True,help='path to wholeslide images')
    ap.add_argument('-mp','--maskpath',required=True,help='path to prediction masks')
    ap.add_argument('-sp','--savepath',required=True,help='path to save plots and stats')

    args=vars(ap.parse_args())
    wsiPath=args['wsipath']
    maskPath=args['maskpath']
    savePath=args['savepath']

    analyseNodes(wsiPath,maskPath,savePath)
