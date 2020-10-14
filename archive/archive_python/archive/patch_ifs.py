import cv2

import openslide
import imageio
import numpy as np
import itertools
import os
from matplotlib.path import Path
import matplotlib.pyplot as plt

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

xmlPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/xml/'
#lnDir = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/'
path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/ndpi/images'
labelKey = {'GERMINAL CENTRE': 1, 'SINUS': 2, 'FOLLICLE':3, 'ADIPOSE':4}
#labelKey = {'GERMINAL CENTRE': 1}
#labelKey = {'SINUS': 2}
labelFileKey={1:'GERMINAL CENTRE', 2:'SINUS', 3:'FOLLICLE', 4:'ADIPOSE'}

'''
Read XML file and returns nested dictionary with following structure
{'Region Type': {Region Id: {Area: 1000: length: 1000, 'Vertex': (X, Y)}}}
'''
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


def getShapePixels(xmlFile, ndpiFile):

    scan = openslide.OpenSlide(ndpiFile)
    dims = scan.dimensions

    labelAreas = getRegions(xmlFile)

    v = labelAreas[''][1]['coords']

    keys = list(labelAreas.keys())

    for k in keys:

        if k not in labelKey:
            del labelAreas[k]

    return labelAreas, scan, v


'''
get mask labels for image segmentation
'''
def getMask(annotation, label, dims, xy, ndpiFile, folder):

    print('Getting the mask')
    colors = [(0,0,0),(255,0,0),(0,255,0),(0,0,255)]
    img = np.zeros((dims[1], dims[0], 3),dtype=np.uint8)
    img1 = cv2.fillPoly(img, [np.int32(np.stack(annotation[0]))], color=colors[label])
    img2 = img1[xy[1]-2000:xy[1]+2000, xy[0]-2000:xy[0]+2000,:]

    name = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/masks/'+folder+'/'+ndpiFile+str(xy[0])+'__'+str(xy[1])+'__mask.png'

    cv2.imwrite(name, img2)


'''
Saves down a patch with given coordinates as centre and stores in
the folder corresponding to its label
'''
def savePatch(xy, scan, ndpiFile, labelFileKey={0: 'FOLLICLE', 1:'GERMINAL', 2:'SINUS'}):

    folder = 'IFS'
    
    print('This is folder {}'.format(folder))

    patch = scan.read_region((xy[0]-224, xy[1]-224), 0, (448, 448))
    patchRGB = patch.convert('RGB')

    if np.mean(patchRGB) < 200:
        name = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/classification/'+folder+'/'+ndpiFile+str(xy[0])+'__'+str(xy[1])+'__'+'.png'
        patchRGB.save(name)

    #getMask(annotation, label, scan.dimensions, xy, ndpiFile, folder)


def getPatches(xmlAnnotations, scan, ndpiFile, v):

    annotations = [(v2['coords'], labelKey[k]) for k, v in xmlAnnotations.items() for k2, v2 in v.items()]
    labels = [a[1] for a in annotations]
    dims = scan.dimensions

    boundaries = list(map(lambda x: (min(x), max(x)), list(zip(*v))))

    for w in range(boundaries[0][0], boundaries[0][1], 224):
        for h in range(boundaries[1][0], boundaries[1][1], 224):
            paths = [(Path(a[0]).contains_point([w, h]), l) for a, l in zip(annotations, labels)]
            paths = list(filter(lambda x: x[0]==True, paths))
            if len(paths) == 0:
                savePatch((w,h), scan, ndpiFile)



if __name__ == '__main__':

    xmlFiles = [f[:-4] for f in os.listdir(xmlPath) if 'xml' in f]

    #for p in filePaths:
        #path = lnDir + p

    ndpiFiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and '.ndpi' in f]



    for ndpi in ndpiFiles:
        print(ndpi[61:-5])
        if ndpi[61:-5] in xmlFiles:
            print('loading ' + ndpi)
            xmlFile = xmlPath + ndpi[61:-5]+'.xml'
            xmlAnnotations, scan, v = getShapePixels(xmlFile, ndpi)
            getPatches(xmlAnnotations, scan, ndpi[61:], v)
