#!/usr/local/env python3

'''measure.py: Slide, LymphNode, Germinal and Sinuses
classes used to detect the predicted features and
quantify them using contouring methods
'''

import os
import glob

import cv2
import numpy as np


class Slide():

    bilateral1_args={"d":9,"sigmaColor":10000,"sigmaSpace":150}
    bilateral2_args={"d":90,"sigmaColor":5000,"sigmaSpace":5000}
    bilateral3_args={"d":90,"sigmaColor":10000,"sigmaSpace":10000}
    bilateral4_args={"d":90,"sigmaColor":10000,"sigmaSpace":100}
    thresh1_args={"thresh":0,"maxval":255,"type":cv2.THRESH_TRUNC+cv2.THRESH_OTSU}
    thresh2_args={"thresh":0,"maxval":255,"type":cv2.THRESH_OTSU}

    def __init__(self, slide, mask,w,h,wNew=1,hNew=1,
                 pixWidth=0.23e-6, pixHeight=0.23e-6):

        self.slide=slide
        self.mask=mask
        self.w=w
        self.h=h
        self.wNew=wNew
        self.hNew=hNew
        self.pixWidth=pixWidth
        self.pixHeight=pixHeight
        self.contours=None
        self._lymphNodes=None

    @property
    def wScale(self):
        return (self.w/self.wNew)*self.pixWidth

    @property
    def hScale(self):
        return (self.h/self.hNew)*self.pixHeight


    def extractLymphNodes1(self, germLabel, sinusLabel, pixelDist=9, sigmaColor=100,
                          sigmaSpace=100, minThresh=0, maxThresh=255,
                          threshType=cv2.THRESH_BINARY+cv2.THRESH_OTSU):

        gray=cv2.cvtColor(self.slide,cv2.COLOR_BGR2GRAY)
        blur=cv2.bilateralFilter(np.bitwise_not(gray),pixelDist,sigmaSpace,sigmaColor)
        _,thresh=cv2.threshold(blur,minThresh,maxThresh,threshType)
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = list(filter(lambda x: cv2.contourArea(x) > 40000, contours))

        self._lymphNodes=[self._createLymphNode(c, thresh, germLabel,sinusLabel)
                          for c in contours]
        self._lymphNodes=list(filter(lambda x: x is not None, self._lymphNodes))

        return len(self._lymphNodes)


    def extractLymphNodes(self, germLabel, sinusLabel):

        img_hsv=cv2.cvtColor(self.slide,cv2.COLOR_RGB2HSV)
        lower_red=np.array([120,0,0])
        upper_red=np.array([180,255,255])
        mask=cv2.inRange(img_hsv,lower_red,upper_red)
        img_hsv=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
        m=cv2.bitwise_and(self.slide,self.slide,mask=mask)
        im_fill=np.where(m==0,233,m)
        mask=np.zeros(self.slide.shape)
        gray=cv2.cvtColor(im_fill,cv2.COLOR_BGR2GRAY)
        #generate the blur
        blur1=cv2.bilateralFilter(np.bitwise_not(gray),**Slide.bilateral1_args)
        #step2: make the pixeldist and sigma space larger so that the content can be linked together
        blur2=cv2.bilateralFilter(np.bitwise_not(blur1),**Slide.bilateral2_args)
        #step3: make each lymph node looks mor like a group
        blur3=cv2.bilateralFilter(np.bitwise_not(blur2),**Slide.bilateral3_args)
        #step4: contain more color as possible
        blur4=cv2.bilateralFilter(np.bitwise_not(blur3),**Slide.bilateral4_args)
        blur_final=255-blur4
        #threshold twice
        _,thresh=cv2.threshold(blur_final,**Slide.thresh1_args)
        _,thresh=cv2.threshold(thresh,**Slide.thresh2_args)
        #find contours
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours=list(filter(lambda x: cv2.contourArea(x) > 9000, contours))
        self.contours=contours
        self._lymphNodes=[self._createLymphNode(c, thresh, germLabel,sinusLabel) for c in contours]
        self._lymphNodes=list(filter(lambda x: x is not None, self._lymphNodes))
        return len(self._lymphNodes)


    def _createLymphNode(self, contour, thresh, germLabel, sinusLabel):

        x,y,lnW,lnH=cv2.boundingRect(contour)
        if lnW<100 or lnH<100:
            return None

        #TODO: do I need to keep lnImage
        lnMask=self.mask[y:y+lnH,x:x+lnW]
        lnImage=self.slide[y:y+lnH,x:x+lnW]
        new=thresh[y:y+lnH,x:x+lnW]

        #we update the contour of the ln based on new mask/image
        contours,_=cv2.findContours(new,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        areas=[cv2.contourArea(c) for c in contours]
        lnContour = contours[areas.index(max(areas))]

        return LymphNode(lnContour,self,lnMask,new,lnImage,'test',germLabel,sinusLabel)

    def filterLymphNodes(self):
        #TODO: how do we decide what to filter - blob detection????
        contours = list(filter(lambda x: cv2.contourArea(x) > 2000, contours))
        contours = list(filter(lambda x: cv2.contourArea(x) < 100000, contours))


class Germinals():
    def __init__(self, ln, mask, label):
        self.ln=ln
        mask[mask!=label]=0
        self.mask=mask
        self.label=label
        self.annMask=mask
        self._germinals=None
        self._num=None
        self._boundingBoxes=None
        self._sizes=None
        self._areas=None

    @property
    def locations(self):
        if self._germinals is None:
            raise ValueError('No germinals detected')
        return [self.ln.calculateCentre(g) for g in self._germinals]

    @property
    def totalArea(self):
        if self._areas is None:
            raise ValueError('germinal areas not calculated')
        #print('areas', self._areas)
        return sum(self._areas)

    @property
    def totalArea2(self):
        return (len(self.mask[self.mask==self.label])
                *self.ln.slide.wScale*self.ln.slide.hScale)


    def circularity():
        return [self.ln.calculateCentre(g) for g in self._germinals]


    def detectGerminals(self):

        if len(self.mask.shape)==3:
            gray=cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        #edges=cv2.Canny(self.mask,30,200)
        blur=cv2.bilateralFilter(self.mask,9,100,100)
        _,thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

        self._germinals=contours
        self._num=len(self._germinals)
        self.annMask=thresh

        return self._num


    #TODO: should we use diagonal bounding box to measure
    def measureSizes(self, pixels=False):

        #TODO how to handle no germinals for sizes
        if len(self._germinals)==0:
            #print('here')
            self._sizes=[(0,0)]
        else:
            self._boundingBoxes=list(map(cv2.boundingRect, self._germinals))
            self._sizes=[(b[2],b[3]) for b in self._boundingBoxes]

            if not pixels:
                f = lambda x: (x[0]*self.ln.slide.wScale,x[1]*self.ln.slide.hScale)
                self._sizes=list(map(f, self._sizes))

        return self._sizes


    def measureAreas(self, pixels=False):

        self._areas=list(map(cv2.contourArea, self._germinals))
        if not pixels:
            f = lambda x: (x*self.ln.slide.wScale*self.ln.slide.hScale)
            self._areas=list(map(f, self._areas))

        return self._areas


    def circularity(self):
        areas=self.measureAreas(pixels=True)
        f = lambda x: cv2.arcLength(x,True)
        perimeters = list(map(f, self._germinals))
        f = lambda x: (4*np.pi*x[0])/np.square(x[1])
        c= list(map(f,zip(areas,perimeters)))
        return c



    def distanceFromCenter(self, pixels=False):

        lnPnt=np.asarray(self.ln.centre)
        locations=[np.asarray(l) for l in self.locations]
        f = lambda x: np.linalg.norm(lnPnt-x)
        distances = list(map(f, self.locations))

        return distances


    def distanceFromBoundary(self, pixels=False):

        pnts=np.asarray([p for p in self.locations])
        points=[np.asarray([list(p[0]) for p in self.ln.contour])
                           for c in self._germinals]

        f = lambda x: np.sqrt(np.sum((x[0]-x[1])**2,axis=1))
        dist = list(map(f, zip(points,pnts)))
        minIdx = list(map(np.argmin,dist))
        return [d[i] for i,d in zip(minIdx,dist)]

        '''
        dist=[]
        for l in self.locations:
            lnPnt=np.asarray(l)
            points=np.asarray([list(p[0]) for p in self.ln.contour])
            f = lambda x: np.linalg.norm(lnPnt-x)
            distances = list(map(f, points))
            dist.append(min(distances))

        return dist
        '''

    def visualiseGerminals(self, color=(0,0,255)):

        plot=self.annMask

        if self._germinals != 0 and len(self.annMask.shape)==2:
            #self.annMask=cv2.cvtColor(self.annMask,cv2.COLOR_GRAY2BGR)
            plot=cv2.drawContours(self.annMask, self._germinals, -1, color, 3)

        if self._sizes != [(0,0)]:
            colorReverse=color[::-1]
            for b in self._boundingBoxes:
                #print('what is going on')
                x,y,w,h = b
                plot=cv2.rectangle(plot,(x,y),(x+w,y+h), 180,1)

        return plot


class Sinuses():
    def __init__(self, ln, mask, label):
        self.ln=ln
        mask[mask!=label]=0
        self.sinusMask=mask
        self.annMask=mask
        self.label=label
        self._sinuses = None
        self._num=None
        self._areas = None


    @property
    def totalArea(self):
        return sum(self._areas)


    @property
    def totalArea2(self):
        return (len(self.sinusMask[self.sinusMask==self.label])
                   *self.ln.slide.wScale*self.ln.slide.hScale)


    def detectSinuses(self):

        if len(self.sinusMask.shape)==3:
            self.annMask=cv2.cvtColor(self.annMask, cv2.COLOR_BGR2GRAY)

        edges=cv2.Canny(self.annMask,30,200)

        #blur=cv2.bilateralFilter(np.bitwise_not(self.sinusMask),9,100,100)
        #_,thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

        contours = list(filter(lambda x: cv2.contourArea(x) > 0, contours))
        self._sinuses=contours
        self._num=len(self._sinuses)
        self.annMask=edges

        return self._num


    def measureAreas(self):
        self._areas=[cv2.contourArea(c) for c in self._sinuses]
        return self._areas


    def visualiseSinus(self, color=(0,0,255)):

        plot=self.annMask
        #plot = np.bitwise_not(plot)

        if self._sinuses != None and len(self.annMask.shape)==2:

            self.annMask=cv2.cvtColor(self.annMask,cv2.COLOR_GRAY2BGR)
            plot=cv2.drawContours(self.annMask, self._sinuses, -1, color,1)

        return plot


class LymphNode():
    def __init__(self, contour, slide, mask,new,image,
                 name, germLabel, sinusLabel):

        self.slide=slide
        self.contour=contour
        self.mask=mask
        self.new=new
        self.image=image
        self.centre=self.calculateCentre(contour)
        #self.area=cv2.contourArea(contour)
        self.germinals = Germinals(self,mask.copy(), germLabel)
        self.sinuses = Sinuses(self,mask.copy(), sinusLabel)

    @property
    def area(self):
        a=cv2.contourArea(self.contour)
        return a*self.slide.wScale*self.slide.hScale


    @staticmethod
    def calculateCentre(point):

        M = cv2.moments(point)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

        return x, y


    def visualise(self):

        plot=cv2.cvtColors(self.mask,cv2.COLOR_GRAY2BGR)
        plot=cv2.drawContours(plot, self.node, -1, (0,0,255),1)
        x,y,w,h = self.boundingBox
        plot=cv2.rectangle(plot,(x,y),(x+w,y+h), (255,0,0),1)

        return plot
