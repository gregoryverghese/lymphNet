#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
patching.py: extracts patches, labels or masks from a whole slide image: CODE NEEDS REFACTORING
'''

import os
import json
import glob
import argparse
import xml.etree.ElementTree as ET

import cv2
import openslide
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from itertools import chain


class WSITiling():
    '''
    class for tiling a whole slide image into patches. Uses 
    pathologist annotations to generate masks
    '''
    def __init__(self, tileDim, resizeDim, magLevel, magFactor, step, masks,
        imageDir, maskDir, outPath, software, drawBorder, feature):
        self.tileDim = tileDim
        self.resizeDim = resizeDim
        self.magLevel = magLevel
        self.magFactor =magFactor
        self.step = step
        self.classKey=classKey
        self.classLabel = {v:k for k, v in self.classKey.items()}
        self.masks = masks
        self.imageDir = imageDir
        self.maskDir = maskDir
        self.outPath = outPath
        self.software = software
        self.drawBorder = drawBorder
        self.feature = feature


    def getRegions(self, xmlFileName):
        '''
        extract annotations from xml format
        generated in ImageScope
        '''
        tree = ET.parse(xmlFileName)
        root = tree.getroot()
        pixelSpacing = float(root.get('MicronsPerPixel'))

        regions = {n.attrib['Name']: n[1].findall('Region') for n in root}
        labelAreas = {}

        for a in regions:
            region = {}
            for r in regions[a]:
                iD = int(r.get('Id'))
                area = r.attrib['AreaMicrons']
                length = r.attrib['LengthMicrons']
                vertices = r[1].findall('Vertex')
                f = lambda x: (int(float(x.attrib['X'])), int(float(x.attrib['Y'])))
                coords = list(map(f, vertices))
                region[iD] = dict(zip(('area', 'length', 'coords'), (area, length, coords)))

            labelAreas[a] = region

        return labelAreas


    def getPatchMasks(self, scan, ndpi, boundaries, annotations):
        '''
        generates patches and corresponding masks at a given 
        magnification level
        '''

        dim = scan.dimensions
        img = np.zeros((dim[1], dim[0]), dtype=np.uint8)
        size = self.tileDim*self.magFactor
        step = self.step*self.magFactor

        for k in annotations:
            v = annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(img, v, color=k)

        if self.magLevel != 0:
            wMin, wMax = boundaries[0]
            wMinNew = self.magFactor*int(np.floor((wMin/self.magFactor)))
            wMaxNew = self.magFactor*int(np.ceil((wMax/self.magFactor)))
            
            hMin, hMax = boundaries[1]
            hMinNew = self.magFactor*int(np.floor((hMin/self.magFactor)))
            hMaxNew = self.magFactor*int(np.ceil((hMax/self.magFactor)))

            boundaries  = [(wMin, wMax), (hMin,hMax)]

        for w in range(boundaries[0][0], boundaries[0][1], step):
            for h in range(boundaries[1][0], boundaries[1][1], step):

                x = int(w-(size*.5))
                y = int(h-(size*.5))
                patch = scan.read_region((x, y), self.magLevel,(self.tileDim, self.tileDim))
                mask = img[h-int(size*0.5):h+int(size*0.5),w-int(size*0.5):w+int(size*0.5)]
                ################################################################################
                for v in list(annotations.values())[0]:
                    p = Path(v)
                    contains = p.contains_point([w, h])
                    if (contains and mask.shape == (size, size) and np.mean(patch) < 200):
                       print((w, h))
                       patch = patch.convert('RGB')
                       patch = patch.resize((self.resizeDim, self.resizeDim))
                       if self.resizeDim!=self.tileDim :
                           patch.resize((self.resizeDim, self.resizeDim)) 
                       try:
                           mask = cv2.resize(mask,(self.resizeDim,self.resizeDim))
                       except:
                           print(e)
                           continue
                #################################################################################


                       imgpath = os.path.join(self.outPath,self.imageDir)
                       maskpath = os.path.join(self.outPath, self.maskDir)
                       filename = os.path.basename(ndpi)[:-5]+'_'+str(w)+'_'+str(h)
                       patch.save(os.path.join(imgpath, filename+'.png'))
                       cv2.imwrite(os.path.join(maskpath, filename + '_masks.png'),mask)
                       break  

        ##########################Code to create sparse dataset##############################
                '''
                patch = scan.read_region((int(w), int(h)), self.magLevel, (self.tileDim, self.tileDim))
                mask = img[h:h+int(self.tileDim*self.magFactor), w:w+int(self.tileDim*self.magFactor)]  
               
                if np.mean(patch) < 200 and (mask.shape == (self.tileDim*self.magFactor, self.tileDim*self.magFactor)):
                    print((w, h))

                    try:
                        mask = cv2.resize(mask,(self.resizeDim,self.resizeDim))
                    except:
                        print(e)
                        continue

                    patch = patch.convert('RGB')
                    patch = patch.resize((self.resizeDim, self.resizeDim)) if self.resizeDim!=self.tileDim else patch
                    #ToDo: deal with edge cases where patch is greater than
                    #mask dimensions
        #######################################################################################
        
                    imgpath = os.path.join(self.outPath,self.imageDir)
                    maskpath = os.path.join(self.outPath, self.maskDir)
                    filename = os.path.basename(ndpi)[:-5]+'_'+str(w)+'_'+str(h)
                    patch.save(os.path.join(imgpath, filename+'.png'))
                    cv2.imwrite(os.path.join(maskpath, filename + '_masks.png'),mask)
         
                    '''
    def filterPatches(self, p, w, h, tolerance=0.75):
        '''
        removes patches from given class set if area 
        belonging to class is not met
        '''
        xx,yy=np.meshgrid(np.arange(self.tileDim),np.arange(self.tileDim))
        #print('Shape of xx is:{}{}'.format(xx.shape[1],xx.shape[0]))

        ys=np.array([i for i in range(h-(int(self.tileDim/2)), h+(int(self.tileDim/2)))])
        xs=np.array([i for i in range(w-(int(self.tileDim/2)), w+(int(self.tileDim/2)))])

        for i in range(len(xx)):
            xx[i]=xs

        for i in range(len(yy[0])):
            yy[:,i]=ys

        xnew, ynew =xx.flatten(),yy.flatten()

        points=np.vstack((xnew,ynew)).T
        grid=p.contains_points(points)

        num=(tolerance*grid.shape[0])
        x = len(grid[grid==True])

        return x>=num, x


    def getPatches(self, scan, ndpi, boundaries, annotations):
        '''
        generates patches and corresponding labels
        '''
        #annotations = {k:v2 for k, v in annotations.items() for v2 in v}
 
        print('Number of annotations{}'.format(len(list(annotations.values())[0])))
        for w in range(boundaries[0][0], boundaries[0][1], self.step*self.magFactor):
            for h in range(boundaries[1][0], boundaries[1][1], self.step*self.magFactor):
                 
                paths = [Path(v) for v in list(annotations.values())[0]]
                contains = [p.contains_point([w, h]) for p in paths]
                overlaps = [self.filterPatches(p,w,h)[1] for p in paths]
                if False in contains and sum(overlaps) == 0:
                        x = int(w-(self.tileDim*0.5))
                        y = int(h-(self.tileDim*0.5))
                        patch = scan.read_region((x, y), self.magLevel, (self.tileDim, self.tileDim))
                        patch.convert('RGB')
                        if np.mean(patch) < 200:
                            folder = 'ifr'
                            print(w,h, folder)
                            patch.save(os.path.join(os.path.join(self.outPath,self.feature, folder, os.path.basename(ndpi)[:-5])+'_'+str(w)+'_'+str(h)+'.png')),
        
                #maskDim = tf.shape(mask).numpy()
                #patch = scan.read_region((int(w-(self.tileDim*0.5)), int(h-(self.tileDim*0.5))), self.magLevel, (self.tileDim, self.tileDim))
                #patch.convert('RGB')
                #patch = patch.resize((self.resizeDim, self.resizeDim)) if self.resizeDim!=tileDim else patch
                #for k, v in annotations.items():
                 
                #overlaps = []
                #for i,v in enumerate(list(annotations.values())[0]): 
                    #p = Path(v)
                    #contains = p.contains_point([w, h])
                    #if contains==True:   
                        #overlap = self.filterPatches(p, w, h)
                        #overlaps.append(overlap[1])
                        #if overlap[0]==True:
                            #folder = self.classLabel[k]
                            #patch = scan.read_region((int(w-(self.tileDim*0.5)), int(h-(self.tileDim*0.5))), self.magLevel, (self.tileDim, self.tileDim))
                            #patch.convert('RGB')
                            #patch = patch.resize((self.resizeDim, self.resizeDim)) if self.resizeDim!=tileDim else patch
                            #folder = self.feature
                            #print(folder, w, h, overlaps)
                            #patch.save(os.path.join(os.path.join(self.outPath, folder), os.path.basename(ndpi)[:-5])+'_'+str(w)+'_'+str(h)+'.png')
                            #break
            
    def drawBoundary(self, annotations):
        '''
        draw boundary around all annotations
        '''
        allAnn = list(chain(*[annotations[f] for f in annotations]))
        coords = list(chain(*allAnn))
        boundaries = list(map(lambda x: (min(x)-2000, max(x)+2000), list(zip(*coords))))
        #print('boundaries: {}'.format(boundaries))
        return boundaries


    def getImageJAnnotations(self, ndpi, xmlPath):
        '''
        get annotations from ImageJ xml file
        '''
        #self.classKey = {'SINUS': 1}

        if self.feature == 'sinus':
            self.classKey = {'SINUS':1}
        elif self.feature == 'germinal':
            self.classKey = {'GERMINAL CENTRE':1}
        elif self.feature == 'follicle':
            self.classKey = {'FOLLICLE':1}
        elif self.feature == ['germinal', 'sinus']:
            self.classKey = {"GERMINAL":1, "SINUS":2}


        xmlFile = os.path.join(xmlPath, os.path.basename(ndpi)[:-5]) + '.xml'
        if not os.path.exists(xmlFile):
            print('no ImageJ xml file')
            return None, None

        print('Getting annotations from ImageJ xml')
        xmlAnnotations = self.getRegions(xmlFile)

        border = xmlAnnotations[''][1]['coords']
        boundaries = list(map(lambda x: (min(x), max(x)), list(zip(*border))))

        keys = xmlAnnotations.keys()
        for k in list(keys):
            if k not in self.classKey:
                del xmlAnnotations[k]

        if not xmlAnnotations:
            print('No {} annotations in ImageJ'.format(self.feature))
            return None, None

        values = sum([len(xmlAnnotations[k]) for k in keys])
        if values==0:
            print('No coordinates for {} annotations in ImageJ'.format(self.feature))
            return None, None

        print('ImageJ annotations exist for {}'.format(self.feature))
        annotations = {self.classKey[k]: [v2['coords'] for k2, v2 in v.items()] 
                                             for k, v in xmlAnnotations.items()}
        return annotations, boundaries


    def getQupathAnnotations(self, ndpi, jsonPath):
        '''
        get annotations from qupath json
        '''
    #hack but need to change to make it more elegant

        if self.feature == 'sinus':
            self.classKey = {'sinus':1}
        elif self.feature == 'germinal':
            self.classKey = {'GC':1}
        elif self.feature == 'follicle':
            self.classKey = {'follicle':1}
        elif self.feature == ['germinal', 'sinus']:
            self.classKey = {"GC":1, "sinus":2}

        print('Getting annotations from json')
        jsonFiles = os.path.join(jsonPath, os.path.basename(ndpi)[:-5])+'.json'
        print('jsonFile: {}'.format(jsonFiles))

        try:
            with open(jsonFiles) as jsonFile:
                jsonAnnotations = json.load(jsonFile)

        except Exception as e:
            print('error',e)
            print('no json file')
            return None, None

        keys = list(jsonAnnotations.keys())

        for k in keys:
            if k not in self.classKey:
                del jsonAnnotations[k]

        if not jsonAnnotations:
            print('dict empty')
            return None, None

        #hack problem is here if we have a key in self.classKey that isnt in
        #jsonAnnotations it bugs out
        values = sum([len(jsonAnnotations[k]) for k in self.classKey])
        if values==0:
            print('empty annotations')
            return None, None

        print('dict is not empty')
        annotations = {self.classKey[k]: [[[int(i['x']), int(i['y'])] for i in v2] for k2, v2 in v.items()]for k, v in jsonAnnotations.items()}

        if self.drawBorder:
            boundaries = self.drawBoundary(annotations)
        else:
            border = jsonAnnotations['other']
            border = [[int(c['x']), int(c['y'])] for c in border['1']]
            boundaries = list(map(lambda x: (min(x), max(x)), list(zip(*border))))

        return annotations, boundaries


    def getTiles(self, ndpiPath, annotationPath):
        '''
        loops over whole slide images, gets annotations
        and generates patches and masks
        '''
        ndpiFiles = [f for f in glob.glob(os.path.join(ndpiPath, '*')) if '.ndpi' in f]
        print('ndpiFiles: {}'.format(ndpiFiles))  

        for i, ndpi in enumerate(ndpiFiles):
            if i<0:
                continue
            print('{}: loading {} '.format(i, ndpi), flush=True)
            try:
                scan = openslide.OpenSlide(ndpi)
            except Exception as e:
                print(e)
                continue

            for program in self.software:
                print('loading software: {}'.format(program))
                annotation, boundaries = getattr(self, 'get'+program+'Annotations')(ndpi, annotationPath)
                if annotation is None:
                    continue
                if self.masks:
                    #if patches:
                    print('Getting tiles and corresponding masks')
                    self.getPatchMasks(scan, ndpi,  boundaries, annotation)
                    #elif not patches:
                    #print('Getting whole image and mask')
                    #self.getWSIMasks(scan,ndpi, boundaries, annotation)
                else:
                    print('Getting just masks')
                    self.getPatches(scan, ndpi, boundaries, annotation)


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-np', '--ndpipath', required=True, help='path to WSI files')
    ap.add_argument('-op', '--outpath', required=True, help= 'path for images and masks to be saved to')
    ap.add_argument('-xp', '--annotationpath', required=True, help= 'path for xml annotations')
    ap.add_argument('-cp', '--configpath', required=True, help='path to config file')
    ap.add_argument('-id', '--imagedir', help='image directory')
    ap.add_argument('-md', '--maskdir', help='mask  directory')

    args = vars(ap.parse_args())

    ndpiPath = args['ndpipath']
    annotationPath = args['annotationpath']
    outPath = args['outpath']
    maskDir = args['maskdir']
    imageDir = args['imagedir']

    with open(args['configpath']) as jsonFile:
        config = json.load(jsonFile)

    tileDim = config['tileDim']
    resizeDim = config['resizeDim']
    step = config['step']
    classKey = config['classKey']
    classKey = {k: int(v) for k, v in classKey.items()}
    masks = config['masks']
    magLevel = config['magnification']
    magFactor = config['downsample']
    software = config['software']
    drawBorder = config['drawborder']
    feature = config['feature']

    tiling = WSITiling(tileDim, resizeDim, magLevel,  magFactor, step, masks,
    imageDir, maskDir, outPath, software, drawBorder, feature)
    tiling.getTiles(ndpiPath, annotationPath)
