import os
import cv2
import matplotlib.pyplot as plt
import json
import openslide
import numpy as np
from itertools import chain
import glob
from pathlib import Path
import xml.etree.ElementTree as ET

color = [(255,0,0), (0,0,255),(255,0,0)]
#feature = ['germinal', 'sinus']
feature = 'germinal'
ndpiPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/all/testing/*'
annotationsPath='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/sum_swap_toms/annotations'
outPath='/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/10x/one/testing'
#outPath = '/home/verghese/scratch'
configPath = '/home/verghese/breastcancer_ln_deeplearning/scripts/config/config_tf.json'
testingPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/wsi/Guys/all/testing/*'
#testingPath = '/home/verghese/scratch'
magFactor = 4
mag = 2

def getRegions(xmlFileName):

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


def drawBoundaries(annotations):

    allAnn = list(chain(*[annotations[f] for f in annotations]))
    coords = list(chain(*allAnn))
    boundaries = list(map(lambda x: (min(x)-2000, max(x)+2000), list(zip(*coords))))
    print('boundaries: {}'.format(boundaries))

    return boundaries


def getQupathAnnotations(jsonPath, feature):

    if feature == 'sinus':
        classKey = {'sinus':1}
    elif feature == 'germinal':
        classKey = {'GC':1}
    elif feature == 'germinal_sinus':
        classKey = {'GC':1, 'sinus':2}


    with open(jsonPath) as jsonFile:
        jsonAnnotations = json.load(jsonFile)

    keys = list(jsonAnnotations.keys())

    for k in keys:
        if k not in classKey:
            del jsonAnnotations[k]

    annotations = {classKey[k]: [[[int(i['x']), int(i['y'])] for i in v2] for k2, v2 in v.items()]for k, v in jsonAnnotations.items()}

    return annotations


def getImageJAnnotations(xmlPath, feature):

    if feature == 'sinus':
        classKey = {'SINUS':1}
    elif feature == 'germinal':
        classKey = {'GERMINAL CENTRE':1}
    elif feature == 'germinal_sinus':
        classKey = {'GERMINAL CENTRE':1, 'SINUS':2}

    xmlAnnotations = getRegions(xmlPath)

    keys = xmlAnnotations.keys()

    for k in list(keys):
        if k not in classKey:
             del xmlAnnotations[k]

    if not xmlAnnotations:
        #print('No {} annotations in ImageJ'.format(feature))
        return None, None

    values = sum([len(xmlAnnotations[k]) for k in keys])
    if values==0:
        #print('No coordinates for {} annotations in ImageJ'.format(feature))
        return None, None

    #print('ImageJ annotations exist for {}'.format(feature))
    annotations = {classKey[k]: [v2['coords'] for k2, v2 in v.items()] for k, v in xmlAnnotations.items()}

    return annotations


def resizeDims(currDim, magfactor):
        
    t = np.ceil(currDim/magfactor)

    #power2 = [2**i for i in range(15)]
    power2  = [2**i for i in range(20)]
    #power2 = power2 + [2048*i for i in range(10)]
    power2 = [p for p in power2 if t-p<300] 
    diff = list(map(lambda x: abs(t-x), power2))
    imgsize = power2[diff.index(min(diff))]
    newDim = imgsize*magFactor
                                
    return newDim


with open(configPath) as configFile:
    config = json.load(configFile)

#filenames = config['testFiles']
#files = [os.path.join(ndpiPath,f+'.npdi') for f in filenames]
print(testingPath)
files = glob.glob(testingPath)
files = [f for f in files if '.ndpi' in f]
print(files)
files = [f for f in files if '100188_02_R' not in f]
print(len(files))
for i, f in enumerate(files):
    #print('{}: loading {} '.format(i, f), flush=True)
    name = os.path.basename(f)[:-5]
    print(f)
    scan = openslide.OpenSlide(f)

    jsonPath = os.path.join(annotationsPath, name + '.json')
    xmlPath = os.path.join(annotationsPath, name + '.xml')
    print(xmlPath)
    print(jsonPath)
    keys = []
    annotations = {}

    if os.path.exists(jsonPath):
        annotationsQupath = getQupathAnnotations(jsonPath, feature)
        keys = keys + list(annotationsQupath.keys())
        print('Q keys', keys)
        for k in keys:
            if k in annotations:
                annotations[k] = annotations[k] + annotationsQupath[k]
            else:
                annotations[k] = annotationsQupath[k]

    if os.path.exists(xmlPath):
        annotationsXML = getImageJAnnotations(xmlPath, feature)
        keys = keys + list(annotationsXML.keys())
        keys = list(set(keys))
        print('X keys', keys)
        for k in keys:
            if k in annotations:
                annotations[k] = annotations[k] + annotationsXML[k]
            else:
                annotations[k] = annotationsXML[k]

    if not annotations:
        print('No annotations for thi feature')
        continue

    values = sum([len(annotations[k]) for k in keys])
    print('number', values)
    if values==0:
        print('no annotations for feature: {}'.format(feature))
        continue
    
    boundaries = drawBoundaries(annotations)
    scan = openslide.OpenSlide(f)
    dim = scan.dimensions
    print('Scan dimensions:{}'.format(dim))
    wBoundaries = [magFactor*int(np.floor((boundaries[0][0]/magFactor))), magFactor*int(np.ceil((boundaries[0][1]/magFactor)))]
    hBoundaries = [magFactor*int(np.floor((boundaries[1][0]/magFactor))), magFactor*int(np.ceil((boundaries[1][1]/magFactor)))]
    boundaries  = [wBoundaries, hBoundaries]

    ySize = boundaries[1][1]-boundaries[1][0]
    xSize = boundaries[0][1]-boundaries[0][0]

    print('ySize: {}, xSize:{}'.format(ySize, xSize)) 
    xSize = resizeDims(xSize, magFactor)
    ySize = resizeDims(ySize, magFactor)

    print('ySize: {}, xSize:{}'.format(ySize, xSize)) 
    img = np.zeros((dim[1], dim[0], 3), dtype=np.uint8)
 
    print('Y coordinates check',(boundaries[1][0] + (ySize)) > dim[1])


    if (boundaries[1][0] + (ySize)) > dim[1]:
 
        print('adjusting y dimension')
        print('scan max height: {}'.format(dim[1]))
        print('calculated height: {}'.format(boundaries[1][0]+ySize)),
        print('new dim:{}'.format(dim[1] - boundaries[1][0])) 
        boundaries[1][0] = dim[1] - ySize

    print('X coordinatest check', (boundaries[0][0] + (xSize)) > dim[0])

    if (boundaries[0][0] + (xSize)) > dim[0]:
                       
        print('adjusting x dimension')
        print('scan max height: {}'.format(dim[1]))
        print('calculated height: {}'.format(boundaries[0][0]+xSize)),
        print('new dim:{}'.format(dim[1] - boundaries[0][0])) 
        boundaries[0][0] = dim[0] - xSize

    patch = scan.read_region((boundaries[0][0], boundaries[1][0]), mag, (np.int(xSize/magFactor),np.int(ySize/magFactor)))
    print(np.int(xSize/magFactor),np.int(ySize/magFactor))
    for k in annotations:
        print(k)
        v = annotations[k]
        v = [np.array(a) for a in v]
        #c = color[k]

        #cv2.fillPoly(img, v, color=color[k])
        cv2.fillPoly(img, v, color=k)

    #print(np.unique(img))
    img2 = img[boundaries[1][0]:boundaries[1][0]+ySize,boundaries[0][0]:boundaries[0][0]+xSize]
    img2 = cv2.resize(img2, (np.int(xSize/magFactor),np.int(ySize/magFactor)))
    print(np.unique(img2))
    print(img2.shape)
    #:img2[img2!=0]=1
    print('values 0', np.unique(img2[:,:,0]))
    print('values 1', np.unique(img2[:,:,1]))
    print('values 2', np.unique(img2[:,:,2]))
    cv2.imwrite(os.path.join(outPath, 'masks', feature, name+'_masks.png'),img2)
    patch.convert('RGB').save(os.path.join(outPath, 'images', feature, name+'.png'))
