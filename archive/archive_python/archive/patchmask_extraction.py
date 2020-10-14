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

filePaths = ['U_100165', 'U_100229', 'U_100278', 'U_90364', 'U_90475', 'U_100195', 'U_100233', 'U_90122', 'U_90405', 'U_90581', 'U_100203', 'U_100246', 'U_90183', 'U_90420', 'U_90603', 'U_100219', 'U_100254', 'U_90360', 'U_90424', 'U_90651', 'U_90112', 'U_90163', 'U_90264', 'U_90653', 'U_90670', 'U_100042', 'U_100188', 'U_100237', 'U_90157', 'U_90238', 'U_100152']
xmlPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/xml/'
lnDir = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/ndpi/'
labelKey = {'GERMINAL CENTRE': 1, 'SINUS': 2}
#labelKey = {'GERMINAL CENTRE': 1}
#labelKey = {'SINUS': 2}
labelFileKey={1:'GERMINAL', 2:'SINUS'}

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

    keys = list(labelAreas.keys())

    for k in keys:

        if k not in labelKey:
            del labelAreas[k]

    return labelAreas, scan


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
def savePatch(label, xy, annotation, scan, ndpiFile, labelFileKey={0: 'FOLLICLE', 1:'GERMINAL', 2:'SINUS'}):

    folder = labelFileKey[label]

    print('This is folder {}'.format(folder))

    ndpiFile = ndpiFile[62:-5]

    patch = scan.read_region((xy[0]-2000, xy[1]-2000), 0, (4000, 4000))
    name = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/images/'+folder+'/'+ndpiFile+str(xy[0])+'__'+str(xy[1])+'__'+'.png'
    patch.convert('RGB').save(name)

    getMask(annotation, label, scan.dimensions, xy, ndpiFile, folder)


def getPatches(xmlAnnotations, scan, ndpiFile):

    annotations = [(v2['coords'], labelKey[k]) for k, v in xmlAnnotations.items() for k2, v2 in v.items()]
    labels = [a[1] for a in annotations]
    dims = scan.dimensions

    for w in range(0, dims[0], 400):
        for h in range(0, dims[1], 400):
            for a, l in zip(annotations, labels):
                p = Path(a[0])
                contains = p.contains_point([w, h])
                if contains==True:
                    savePatch(l, (w,h), a, scan, ndpiFile)
                    continue


if __name__ == '__main__':

    xmlFiles = [f[:-4] for f in os.listdir(xmlPath) if 'xml' in f]

    for p in filePaths:
        path = lnDir + p
        ndpiFiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and '.ndpi' in f]

        for ndpi in ndpiFiles:
            print(ndpi)
            if ndpi[62:-5] in xmlFiles:
                print('loading ' + ndpi)
                xmlFile = xmlPath + ndpi[62:-5]+'.xml'
                xmlAnnotations, scan = getShapePixels(xmlFile, ndpi)
                getPatches(xmlAnnotations, scan, ndpi)
