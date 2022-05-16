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
    cancerPts='/home/verghese/cancer-points-training'
    print(maskPath)
    print('analysing lymph nodes...',flush=True)
    totalImages=getFiles(wsiPath,'ndpi')
    print('Image N: {}'.format(len(totalImages),flush=True))
    totalMasks=getFiles(maskPath,'png')
    print('Mask N: {}'.format(len(totalMasks),flush=True))
    cancerPoints=getFiles(cancerPts,'.csv')
    all_ln_status=pd.read_csv('/home/verghese/ln_status_3.csv',index_col=['image_name'])

    totalMasks=[t for t in totalMasks if 'image' not in t]
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
    statusLst=[]
    patients=[]
    ln_statuses=[]

    print('total number of images', len(totalImages))
    print('total number of masks', len(totalMasks))

    for maskF in totalMasks:
        name=os.path.basename(maskF)[:-4]
        patient=maskF.split('/')[-2]
        print('image name: {}'.format(name))
        print('patient name:{}'.format(patient))
        
        try:
            wsiF=[s for s in totalImages if name in s][0]
        except Exception as e:
            print('missing wsi')
            continue
        
        try:
            ln_status=int(all_ln_status.loc[name]['ln_status'])
        except Exception as e:
            print('missing ln status')
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
                print('check_names:{}'.format(patient))
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
            plotS=ln.sinuses.visualiseSinus()
            plotG=ln.germinals.visualiseGerminals()

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
            maxGerm.append(np.round(maxGermArea2*1e6,4))
            minGerm.append(np.round(minGermArea2*1e6,4))
            shapes.append(germShape)
            sinusNum.append(numSinuses)
            totalSinusArea.append(np.round(sinusArea*1e6,2))
            centDist.append(np.round(np.mean(germDistCent)))
            boundDist.append(np.round(np.mean(germDistBoundary)))
            patients.append(patient)
            ln_statuses.append(ln_status)

    stats={
        'patient':patients,
        'name':names,
        'ln_status':ln_statuses,
        'ln_idx':lnIdx,
        'ln_area':lnAreas,
        'germ_number':germNum,
        'avg_germ_width':avgGermW,
        'avg_germ_height':avgGermH,
        'total_germ_area':germTotalAreas,
        'total_germ_area2':germTotalAreas2,
        'section_status':statusLst,
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
    statsDf.to_csv('/home/verghese/node_details_cancer_90552.csv')



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
