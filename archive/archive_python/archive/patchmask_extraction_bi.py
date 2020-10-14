import os
import glob
import cv2
import openslide
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


labelKey = {'GERMINAL CENTRE': 0, 'SINUS': 1}
labelFileKey={0: 'GERMINAL CENTRE', 1: 'SINUS'}

xmlPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/xml/'
ndpiPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/ndpi/images'
patchPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/two/images'
maskPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/patches/segmentation/two/masks'


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


def getPatchMasks(ndpiFiles, xmlPath, patchPath, maskPath):

    for ndpi in ndpiFiles:
        print('loading ' + ndpi, flush=True)
        xmlFile = os.path.join(xmlPath, os.path.basename(ndpi)[:-5]) + '.xml'

        scan = openslide.OpenSlide(ndpi)
        xmlAnnotations = getRegions(xmlFile)

        v = xmlAnnotations[''][1]['coords']
        boundaries = list(map(lambda x: (min(x), max(x)), list(zip(*v))))


        keys = xmlAnnotations.keys()

        for k in list(keys):
            if k not in labelKey:
                del xmlAnnotations[k]

        annotations = {labelKey[k]: [v2['coords'] for k2, v2 in v.items()] for k, v in xmlAnnotations.items()}
        img = np.zeros((boundaries[1][1]+2500, boundaries[0][1]+2500, 3), dtype=np.uint8)
        
        colors = [(255,0,0),(0,255,0),(0,0,255)]
        for k in annotations:
            c=colors[k]
            v = annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(img, v, color=c)

        for w in range(boundaries[0][0], boundaries[0][1], 500):
            for h in range(boundaries[1][0], boundaries[1][1], 500):
                patch = scan.read_region((w-2000, h-2000), 0, (4000, 4000))
                img2 = img[h-2000:h+2000, w-2000:w+2000,:]
                print(np.mean(patch), img2.shape)
                if (img2.shape == (4000, 4000, 3) and np.mean(patch) < 200):
                    patch.convert('RGB').save(os.path.join(patchPath, os.path.basename(ndpi)[:-5])+'_'+str(w)+'_'+str(h)+'.png')
                    cv2.imwrite(os.path.join(maskPath, os.path.basename(ndpi)[:-5])+'_'+str(w)+'_'+str(h)+'_masks.png', img2)


if __name__=='__main__':

    ndpiFiles = [f for f in glob.glob(os.path.join(ndpiPath, '*')) if '.ndpi' in f]
    getPatchMasks(ndpiFiles, xmlPath, patchPath, maskPath)


