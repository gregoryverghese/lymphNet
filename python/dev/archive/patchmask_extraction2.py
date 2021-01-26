import os
import json
import glob
import cv2
import openslide
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.path import Path
from itertools import chain
import xml.etree.ElementTree as ET


class WSITiling():
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
		dim = scan.dimensions 
		#img = np.zeros((boundaries[1][1]+2500, boundaries[0][1]+2500), dtype=np.uint8)
		img = np.zeros((dim[1], dim[0]), dtype=np.uint8)
		for k in annotations:
			v = annotations[k]
			v = [np.array(a) for a in v]
			cv2.fillPoly(img, v, color=k)
                
		#print('Annotations', annotations)
		if self.magLevel != 0:
			wBoundaries = [200*int(np.floor((boundaries[0][0]/200))), 200*int(np.ceil((boundaries[0][1]/200)))]
			hBoundaries = [200*int(np.floor((boundaries[1][0]/200))), 200*int(np.ceil((boundaries[1][1]/200)))]
			boundaries  = [wBoundaries, hBoundaries]
		#print('boundaries', boundaries)
		for w in range(boundaries[0][0], boundaries[0][1], self.step*self.magFactor):
			for h in range(boundaries[1][0], boundaries[1][1], self.step*self.magFactor):
				patch = scan.read_region((int(w-(self.tileDim*0.5*self.magFactor)), int(h-(self.tileDim*0.5*self.magFactor))), self.magLevel, (self.tileDim, self.tileDim))
				#patch = scan.read_region((int(w), int(h)), magLevel, (tileDim, tileDim))
				mask = img[h-int(self.tileDim*0.5*self.magFactor):h+int(self.tileDim*0.5*self.magFactor), w-int(self.tileDim*0.5*self.magFactor):w+int(self.tileDim*0.5*self.magFactor)]
				#paths = [Path(v) for a in list(annotations.values()) for v in a]
				#contains = list(map(lambda x: x.contains_point([w, h]), paths))
				for v in list(annotations.values())[0]:
					p = Path(v)
					contains = p.contains_point([w, h])
					if contains==True and (mask.shape == (self.tileDim*self.magFactor, self.tileDim*self.magFactor) and np.mean(patch) < 200):
						#if (mask.shape == (self.tileDim*self.magFactor, self.tileDim*self.magFactor) and np.mean(patch) < 200):
						print((w, h))
						patch = patch.convert('RGB')
						patch = patch.resize((self.resizeDim, self.resizeDim)) if self.resizeDim!=self.tileDim else patch
						mask = cv2.resize(mask, (self.resizeDim,self.resizeDim))
						patch.save(os.path.join(os.path.join(self.outPath, self.imageDir), os.path.basename(ndpi)[:-5])+'_'+str(w)+'_'+str(h)+'.png')
						cv2.imwrite(os.path.join(os.path.join(self.outPath, self.maskDir), os.path.basename(ndpi)[:-5])+'_'+str(w)+'_'+str(h)+'_masks.png', mask)
						break
'''
    def getWSIMasks(self, scan, ndpi, boundaries, annotations):

        dim = scan.dimensions
        img = np.zeros((dim[1], dim[0]), dtype=np.uint8)
        for k in annotations:
            v = annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillpoly(img, v, color=k)

		if self.magLevel != 0:
			wBoundaries = [200*int(np.floor((boundaries[0][0]/200))), 200*int(np.ceil((boundaries[0][1]/200)))]
			hBoundaries = [200*int(np.floor((boundaries[1][0]/200))), 200*int(np.ceil((boundaries[1][1]/200)))]
			boundaries  = [wBoundaries, hBoundaries]

        patch = scan.read_region((int(wBoundaries[0]), 6, (512,512)))
        mask = img[h:h-512,w:w-512]
        patch = patch.convert('RGB')
        patch.save(os.path.join(os.path.join(self.outPath,self.imageDir),os.path.basename(ndpi)))
        cv2.imwrite(os.path.join(os.path.join(self.outPath,self.imageDir),os.path.basename(ndpi)),write)
'''



	def filterPatches(self, p, w, h, tolerance=0.75):

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

		return x>=num


	def getPatches(self, scan, ndpi, boundaries, annotations):
		annotations = {k:v2 for k, v in annotations.items() for v2 in v}
		print('Number of annotations {}'.format(len(annotations)))
		for w in range(boundaries[0][0], boundaries[0][1], self.step*self.magFactor):
			for h in range(boundaries[1][0], boundaries[1][1], self.step*self.magFactor):
				patch = scan.read_region((int(w-(self.tileDim*0.5)), int(h-(self.tileDim*0.5))), self.magLevel, (self.tileDim, self.tileDim))
				for k, v in annotations.items():
					p = Path(v)
					contains = p.contains_point([w, h])
					if contains==True and np.mean(patch) < 200:
						if self.filterPatches(p, w, h):
							print((w, h))
							folder = self.classLabel[k]
							print(folder)
							patch = patch.convert('RGB')
							patch = patch.resize((self.resizeDim, self.resizeDim)) if self.resizeDim!=tileDim else patch
							patch.save(os.path.join(os.path.join(self.outPath, folder), os.path.basename(ndpi)[:-5])+'_'+str(w)+'_'+str(h)+'.png')
							break


	def drawBoundary(self, annotations):

		allAnn = list(chain(*[annotations[f] for f in annotations]))
		coords = list(chain(*allAnn))
		boundaries = list(map(lambda x: (min(x)-2000, max(x)+2000), list(zip(*coords))))
		#print('boundaries: {}'.format(boundaries))   
		return boundaries


	def getImageJAnnotations(self, ndpi, xmlPath):
		#self.classKey = {'SINUS': 1}

        
		if self.feature == 'sinus':
			self.classKey = {'SINUS':1}
		elif self.feature == 'germinal':
			self.classKey = {'GERMINAL CENTRE':1}
		elif self.feature == 'follicle':
			self.classKey = {'FOLLICLE':1}
		print('Getting annotations from xml')

		xmlFile = os.path.join(xmlPath, os.path.basename(ndpi)[:-5]) + '.xml'
		if not os.path.exists(xmlFile):
			return None, None

		xmlAnnotations = self.getRegions(xmlFile)

		border = xmlAnnotations[''][1]['coords']
		boundaries = list(map(lambda x: (min(x), max(x)), list(zip(*border))))

		keys = xmlAnnotations.keys()
		for k in list(keys):
			if k not in self.classKey:
				del xmlAnnotations[k]

		if not xmlAnnotations:
			print('dict empty')
			return None, None
        
		values = sum([len(xmlAnnotations[k]) for k in keys])
		if values==0:
			return None, None

		print('dict is not empty')
		annotations = {self.classKey[k]: [v2['coords'] for k2, v2 in v.items()] for k, v in xmlAnnotations.items()}
		return annotations, boundaries


	def getQupathAnnotations(self, ndpi, jsonPath):

		#hack but need to change to make it more elegant

		if self.feature == 'sinus':
			self.classKey = {'sinus':1}
		elif self.feature == 'germinal':
			self.classKey = {'GC':1}
		elif self.feature == 'follicle':
			self.classKey = {'follicle':1}


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

		ndpiFiles = [f for f in glob.glob(os.path.join(ndpiPath, '*')) if '.ndpi' in f]
	#	ndpiFiles = [f for f in ndpiFiles if 'U_90264' in f]                

		for i, ndpi in enumerate(ndpiFiles):
				
			print('{}: loading {} '.format(i, ndpi), flush=True)
			scan = openslide.OpenSlide(ndpi)

			for program in self.software:
				print('loading software: {}'.format(program))

				annotation, boundaries = getattr(self, 'get'+program+'Annotations')(ndpi, annotationPath)
			    
				if annotation is None:
					continue

				if self.masks:
                    if patches:
					    print('Getting tiles and corresponding masks')
					    self.getPatchMasks(scan, ndpi,  boundaries, annotation)
                    elif not patches:
                        print('Getting whole image and mask')
                        self.getWSIMasks(scan,ndpi, boundaries, annotation)
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