#!usr/bin/env python3

"""
patching.py: contains Patching class for splitting WSIs
into a set of smaller tiles based on annotations
"""

import os
import glob
import json
import xml.etree.ElementTree as ET

import numpy as np
import openslide
import cv2
import seaborn as sns
from matplotlib.path import Path
from openslide import OpenSlide
import pandas as pd
import seaborn as sns
from itertools import chain
import operator as op
from src.utilities.utils import mask2rgb


__author__='Gregory Verghese'
__email__='gregory.verghese@gmail.com'


class Slide(OpenSlide):
    """
    WSI object that enables annotation overlay wrapper around 
    openslide.OpenSlide class. Generates annotation mask.
    Attributes:
        _slide_mask: ndarray mask representation
        dims: dimensions of WSI
        name: string name
        draw_border: boolean to generate border based on annotations
        _border: list of border coordinates [(x1,y1),(x2,y2)]
    """
    MAG_fACTORS={0:1,1:2,3:4,4:16,5:32}

    def __init__(self, filename, draw_border=False,
                 annotations=None, annotations_path=None):
        super().__init__(filename)

        if annotations_path is not None:
            ann=Annotations(annotations_path)
            self.annotations=ann.generate_annotations()
        else:
            self.annotations=annotations

        self.dims = self.dimensions
        self.name = os.path.basename(filename)[:-5]
        self.draw_border=draw_border
        self._border=None


    @property
    def slide_mask(self):
       mask=self.generate_mask((2000,2000))
       mask=mask2rgb(mask)
       return mask


    def generate_mask(self, size=None):
        """
        generates mask representation of annotations
        Args:
            size: tuple of size dimensions for mask
        Returns:
            self._slide_mask: ndarray mask
        """
        #colors=sns.color_palette('hls',len(self.annotations))
        #colors=[(int(c[0]*255),int(c[1]*255),int(c[2]*255)) for c in colors]
        x, y = self.dims[0], self.dims[1]
        slide_mask=np.zeros((y, x), dtype=np.uint8)
        for k in self.annotations:
            v = self.annotations[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=k)

        if size is not None:
            slide_mask=cv2.resize(slide_mask, size)

        return slide_mask


    def generate_annotations(self,path):
        """
        generate annotations object based on json or xml

        Args:
            path: path to json or xml annotation files
            file_type: xml or json
        Returns:
            self.annotations: dictionary of annotation coordinates
        """
        ann_obj=Annotations(path)
        self.annotations = ann_obj.generate_annotations()
        return self.annotations


    @staticmethod
    def resize_border(dim, factor=1, threshold=None, operator='=>'):
        """
        resize and redraw annotations border - useful to cut out
        specific size of WSI and mask

        Args:
            dim: dimensions
            factor: border increments
            threshold: min/max size
            operator: threshold limit

        Returns:
            new_dims: new border dimensions [(x1,y1),(x2,y2)]

        """
        if threshold is None:
            threshold=dim

        operator_dict={'>':op.gt,'=>':op.ge,'<':op.lt,'=<':op.lt}
        operator=operator_dict[operator]
        multiples = [factor*i for i in range(100000)]
        multiples = [m for m in multiples if operator(m,threshold)]
        diff = list(map(lambda x: abs(dim-x), multiples))
        new_dim = multiples[diff.index(min(diff))]

        return new_dim


    #TODO: function will change with format of annotations
    #data structure accepeted
    def get_border(self,space=100):
        """
        generate border around annotations on WSI

        Args:
            space: border space
        Returns:
            self._border: border dimensions [(x1,y1),(x2,y2)]
        """

        coordinates = list(chain(*[self.annotations[a] for a in
                                   self.annotations]))
        coordinates=list(chain(*coordinates))
        f=lambda x: (min(x)-space, max(x)+space)
        self._border=list(map(f, list(zip(*coordinates))))

        return self._border


    def detect_component(self,down_factor=10):

        f = lambda x: round(x/100)
        new_dims=list(map(f,self.dims))
        image=np.array(self.get_thumbnail(new_dims))

        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur=cv2.bilateralFilter(np.bitwise_not(gray),9,100,100)
        _,thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        x_scale=self.dims[0]/new_dims[0]
        y_scale=self.dims[1]/new_dims[1]

        x1=round(x_scale*x)
        x2=round(x_scale*(x+w))
        y1=round(y_scale*y)
        y2=round(y_scale*(y+h))

        self._border=[(x1,x2),(y1,y2)]
        image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

        return image, self._border


    def generate_region(self, mag=0, x=None, y=None, x_size=None, y_size=None,
                        scale_border=False, factor=1, threshold=None, operator='=>'):
        """
        extracts specific regions of the slide

        Args:
            mag: magnfication level 1-8
            x:
            y:
            x_size: x dim size
            y_size: y dim size
            scale_border: resize border
            factor: increment for resizing border
            threshold: limit for resizing border
            operator: operator for threshold
        Returns:
            region: ndarray image of extracted region
            mask: ndarray mask of annotations in region

        """
        if x is None:
            self.get_border()
            x, y = self.border

        x_min, x_max=x
        y_min, y_max=y
        x_size=x_max-x_min
        y_size=y_max-y_min
        x_size=int(x_size/Slide.MAG_fACTORS[mag])
        y_size=int(y_size/Slide.MAG_fACTORS[mag])
        if scale_border:
            x_size = Slide.resize_border(x_size, factor, threshold, operator)
            y_size = Slide.resize_border(y_size, factor, threshold, operator)

        print('x_size:{}'.format(x_size))
        print('y_size:{}'.format(y_size))
        region=self.read_region((x_min,y_min),mag,(x_size, y_size))
        mask=self.slide_mask()[x_min:x_min+x_size,y_min:y_min+y_size]

        return np.array(region.convert('RGB')), mask


    def save(self, path, size=(2000,2000), mask=False):
        """
        save thumbnail of slide in image file format
        Args:
            path:
            size:
            mask:
        """

        if mask:
            cv2.imwrite(path,self._slide_mask)
        else:
            image = self.get_thumbnail(size)
            image = image.convert('RGB')
            image = np.array(image)
            cv2.imwrite(path,image)


class Annotations():

    """
    returns dictionary of coordinates of ROIs

    reads annotation files in either xml and json format
    and returns a dictionary containing x,y coordinates
    for each region of interest in the annotation

    Attributes:
        path: string path to annotation file
        annotation_type: file type
        labels: list of ROI names ['roi1',roi2']
        _annotations: dictonary with return files
                      {roi1:[[x1,y1],[x2,y2],...[xn,yn],...roim:[]}
    """
    def __init__(self, path, source=None,labels=[]):
        self.paths=path
        if source is None:
            self.source==[None]*len(self.paths)
        else:
            self.source=source
        self.labels = labels
        self._annotations={}


    @property
    def class_key(self):
        self.labels=list(set(self.labels))
        if self.labels is not None:
            class_key={l:i for i, l in enumerate(self.labels)}
        return class_key


    def generate_annotations(self):
        """
        calls appropriate method for file type
        Returns:
            annotations: dictionary of coordinates
        """
        class_key=self.class_key
        if not isinstance(self.paths,list):
            self._paths=[self.paths]
        #for p, source in zip(self.paths,self.source):
        for p in self.paths:
            #if source=='imagej':
                #annotations=self._imagej(p)
            #elif source=='asap':
                #annotations=self._asap(p)
            if p.endswith('xml'):
                anns=self._imagej(p)
            elif p.endswith=='csv':
                anns=self._csv(p)
            elif p.endswith('json'):
                anns=self._json(p)
            elif p.endswith('csv'):
                anns=self._csv(p)
            else:
                raise ValueError('provide source or valid filetype')
            for k, v in anns.items():
                if k in self._annotations:
                    self._annotations[k].append(anns[k])
                else:
                    self._annotations[k]=anns[k]
        if self.labels is not None:
            #annotations=self.filter_labels(annotations)
            pass
        return self._annotations


    def filter_labels(self, labels):
        """
        remove labels from annotations
        Returns:
            annotations: filtered dict of coordinates
        """
        keys = list(self._annotations.keys())
        for k in keys:
            if k not in labels:
                self.labels.remove(k)
                del self._annotations[k]
        return self._annotations


    def rename_labels(self, label_names):
        for k,v in label_names.items():
            self._annotations[v] = self._annotations.pop(k)
    

    def encode_keys(self):
        print('keys',self.class_key)
        self._annotations={self.class_key[k]: v for k,v in self._annotations.items()}


    def _imagej(self,path):
        """
        parses xml files

        Returns:
            annotations: dict of coordinates
        """
        tree=ET.parse(path)
        root=tree.getroot()
        anns=root.findall('Annotation')
        labels=list(root.iter('Annotation'))
        labels=list(set([i.attrib['Name'] for i in labels]))
        self.labels.extend(labels)
        annotations={l:[] for l in labels}
        for i in anns:
            label=i.attrib['Name']
            instances=list(i.iter('Vertices'))
            for j in instances:
                coordinates=list(j.iter('Vertex'))
                coordinates=[(c.attrib['X'],c.attrib['Y']) for c in coordinates]
                coordinates=[(round(float(c[0])),round(float(c[1]))) for c in coordinates]
                #annotations[label].append([coordinates])
                annotations[label]=annotations[label]+[coordinates]
        #annotations = {self.class_key[k]: v for k,v in annotations.items()}
        return annotations


    def _asap(self,path):

        tree=ET.parse(path)
        root=tree.getroot()
        ns=root[0].findall('Annotation')
        labels=list(root.iter('Annotation'))
        self.labels=list(set([i.attrib['PartOfGroup'] for i in labels]))
        annotations={l:[] for l in labels}
        for i in ns:
            coordinates=list(i.iter('Coordinate'))
            coordinates=[(float(c.attrib['X']),float(c.attrib['Y'])) for c in coordinates]
            coordinates=[(round(c[0]),round(c[1])) for c in coordinates]
            label=i.attrib['PartOfGroup']
            annotations[label]=annotations[label]+[coordinates]

        annotations = {self.class_key[k]: v for k,v in annotations.items()}
        return annotations


    def _json(self,path):
        """
        parses json file

        Returns:
            annotations: dict of coordinates
        """
        with open(path) as json_file:
            json_annotations=json.load(json_file)
        
        labels=list(json_annotations.keys())
        self.labels.extend(labels) 
        annotations = {k: [[[int(i['x']), int(i['y'])] for i in v2] 
                       for v2 in v.values()] for k, v in json_annotations.items()}
        return annotations


    def _dataframe(self):
        pass


    def _csv(self):
        anns_df=pd.read_csv(path)
        anns_df.fillna('undefined', inplace=True)
        anns_df.set_index('labels',drop=True,inplace=True)
        self.labels=list(set(anns_df.index))
        annotations={l: list(zip(anns_df.loc[l].x,anns_df.loc[l].y)) for l in
                     self.labels}

        annotations = {self.class_key[k]: v for k,v in annotations.items()}
        self._annotations=annotations
        return annotations


    def df(self):
        """
        returns dataframe of annotations

        """
        key={v:k for k,v in self.class_key.items()}
        labels=[[l]*len(self._annotations[l]) for l in self._annotations.keys()]
        labels=chain(*labels)
        labels=[key[l] for l in labels]
        x_values=[xi[0] for x in list(self._annotations.values()) for xi in x]
        y_values=[yi[1] for y in list(self._annotations.values()) for yi in y]
        df=pd.DataFrame({'labels':labels,'x':x_values,'y':y_values})

        return df


    def save(self,save_path):
        """
        save down annotations in csv file
        Args:
            save_path:string save path
        """
        self.df().to_csv(save_path)


class Patching():

    MAG_FACTORS={0:1,1:2,2:4,3:8,4:16}

    def __init__(self, slide, annotations=None, size=(256, 256),
                 mag_level=0,border=None, mode=False):

        super().__init__()
        self.slide=slide
        self.mag_level=mag_level
        self.size=size
        self._number=None
        self._patches=[]
        self._masks=[]

    @property
    def masks(self):
        return self._masks

    @property
    def patches(self):
        return self._patches

    @property
    def annotations(self):
        return _self.annotations

    @property
    def mag_factor(self):
        return Patching.MAG_FACTORS[self.mag_level]

    @property
    def slide_mask(self):
        return self.slide._slide_mask

    @property
    def config(self):
        config={'name':self.slide.name,
                'mag':self.mag_level,
                'size':self.size,
                'border':self.slide.border,
                'mode':None,
                'number':self._number}
        return config


    def __repr__(self):
        return str(self.config)


    @staticmethod
    def patching(step,xmin,xmax,ymin,ymax):
        for x in range(xmin,xmax, step):
            for y in range(ymin,ymax,step):
                yield x, y


    def _remove_edge_cases(self,x,y):
        x_size=int(self.size[0]*self.mag_factor*.5)
        y_size=int(self.size[1]*self.mag_factor*.5)
        xmin=self.slide._border[0][0]
        xmax=self.slide._border[0][1]
        ymin=self.slide._border[1][0]
        ymax=self.slide._border[1][1]
        remove=False

        if x+x_size>xmax:
            remove=True
        if x-x_size<xmin:
            remove=True
        if y+y_size>ymax:
            remove=True
        if y-y_size<ymin:
            remove=True
        return remove


    def generate_patches(self,step, mode='sparse',mask_flag=False):
        self._patches=[]
        self._masks=[]
        step=step*self.mag_factor
        xmin=self.slide._border[0][0]
        xmax=self.slide._border[0][1]
        ymin=self.slide._border[1][0]
        ymax=self.slide._border[1][1]

        for x, y in self.patching(step,xmin,xmax,ymin,ymax):
            name=self.slide.name+'_'+str(x)+'_'+str(y)
            if self._remove_edge_cases(x,y):
                #continue
                pass
            self.patches.append({'name':name,'x':x,'y':y})
            if mask_flag:
                mask=self.slide.slide_mask[y:y+self.size[0],x:x+self.size[1]]
                if mode == 'focus':
                    classes = len(np.unique(mask))
                    self._masks.append({'x':x, 'y':y, 'classes':classes})
                    self.focus()
                else:
                    self._masks.append({'x':x, 'y':y})
        self._number=len(self._patches)
        return self._number


    def focus(self, task='classes'):

        if task=='classes':
            index=[i for i in range(len(self._patches)) if
                  self._masks[i][task] >1]
        elif task=='labels':
            index=[i for i in range(len(self._patches)) if
                   self._masks[i][task]!=9]

        self._patches = [self.patches[i] for i in index]
        self._masks = [self.masks[i] for i in index]

        return len(self._patches)


    @staticmethod
    def __filter(y_cnt,cnts,threshold):
        ratio=y_cnt/float(sum(cnts))
        return ratio>=threshold


    #TODO:how do we set a threshold in multisclass
    def generate_labels(self,threshold=1):
        labels=[]
        for i, (m,x,y) in enumerate(self.extract_masks()):
            cls,cnts=np.unique(m, return_counts=True)
            y=cls[cnts==cnts.max()]
            y_cnt=cnts.max()
            if self.__filter(y_cnt,cnts,threshold):
                self.masks[i]['labels']=y[0]
                labels.append(y)
            else:
                self.masks[i]['labels']=9
                #TODO:do we want a labels attribute
                labels.append(y)

        return np.unique(np.array(labels),return_counts=True)


    def plotlabeldist(self):
        labels=[self.masks[i]['labels'] for i in range(len(self.masks))]
        return sns.distplot(labels)


    #TODO: maybe we don't need .5 - should check
    def extract_patch(self, x=None, y=None):
        x_size=int(self.size[0]*self.mag_factor*.5)
        y_size=int(self.size[1]*self.mag_factor*.5)

        patch=self.slide.read_region((x-x_size,y-y_size), self.mag_level,
                                     (self.size[0],self.size[1]))
        patch=np.array(patch.convert('RGB'))
        return patch


    def extract_patches(self):
        for p in self._patches:
            patch=self.extract_patch(p['x'],p['y'])
            yield patch,p['x'],p['y']


    def extract_mask(self, x=None, y=None):

        x_size=int(self.size[0]*self.mag_factor*.5)
        y_size=int(self.size[1]*self.mag_factor*.5)
        mask=self.slide.generate_mask()[y-y_size:y+y_size,x-x_size:x+x_size]
        mask=cv2.resize(mask,(self.size[0],self.size[1]))

        return mask


    def extract_masks(self):
        for m in self._masks:
            mask=self.extract_mask(m['x'],m['y'])
            yield mask,m['x'],m['y']


        #TODO: how to save individiual patch and mask
    @staticmethod
    def saveimage(image,path,filename,x=None,y=None):

        if y is None and x is not None:
            raise ValueError('missing y')
        elif x is None and y is not None:
            raise ValueError('missing x')
        elif (x and y) is None:
            image_path=os.path.join(path,filename)
        elif (x and y) is not None:
             filename=filename+'_'+str(x)+'_'+str(y)+'.png'
             image_path=os.path.join(path,filename)
        print('image_path',image_path)
        status=cv2.imwrite(image_path,image)
        return status


    #TODO fix masks. Currently saving only first mask over and over
    def save(self, path, mask_flag=False):

        patchpath=os.path.join(path,'images')
        try:
            os.mkdir(patchpath)
        except OSError as error:
            print(error)

        if mask_flag:
            maskpath=os.path.join(path,'masks')
            try:
                os.mkdir(os.path.join(maskpath))
            except OSError as error:
                print(error)

            mask_generator=self.extract_masks()
        for patch,x,y in self.extract_patches():
            #if np.mean(patch[:,:,1])>210:
                #continue
            #test=patch[:,:,1]

            #if len(test[test>220])>(0.3*self.size[0]**2):
                #print('here')
                #continue
            patchstatus=self.saveimage(patch,patchpath,self.slide.name,x,y)
            if mask_flag:
                mask,x,y=next(mask_generator)
                maskstatus=self.saveimage(mask,maskpath,self.slide.name,x,y)


class Stitching():

    MAG_FACTORS={0:1,1:2,2:4,3:8,4:16}

    def __init__(self,patch_path,slide=None,patching=None,name=None,
             step=None,border=None,mag_level=0):

        self.patch_path=patch_path
        patch_files=glob.glob(os.path.join(self.patch_path,'*'))
        print('found {} patches'.format(len(patch_files)))
        self.fext=patch_files[0].split('.')[-1]
        self.slide=slide
        self.coords=self._get_coords()

        if patching is not None:
            self.name=self.patching.slide.name
        elif slide is not None:
            self.name=self.slide.name
        elif name is not None:
            self.name=name
        else:
            self.name='pyslide_wsi'

        if border is not None:
            self.border=border
        elif patching is not None:
            self.border=patching.slide.border
        elif slide is not None:
            self.border=slide.border
        else:
            self.border=self._get_border()
        print('border',self.border)

        if patching is not None:
            self.mag_level=patching.mag_level
        else:
            self.mag_level=mag_level

        self.step=self._get_step() if step is None else step


    @property
    def mag_factor(self):
         return Stitching.MAG_FACTORS[self.mag_level]


    def _get_coords(self):
        patch_files=glob.glob(os.path.join(self.patch_path,'*'))
        coords=[(int(f.split('_')[-2:][0]),int(f.split('_')[-2:][1][:-4]))
                for f in patch_files]

        self._coords=coords
        return self._coords


    def _get_border(self):
        coords=self._get_coords()
        xmax=max([c[0] for c in coords])
        xmin=min([c[0] for c in coords])
        ymax=max([c[1] for c in coords])
        ymin=min([c[1] for c in coords])

        return [[xmin,xmax],[ymin,ymax]]


    def _get_step(self):
        coords=self._get_coords()
        xs=[c[0] for c in coords]
        step=min([abs(x1-x2) for x1, x2 in zip(xs, xs[1:]) if abs(x1-x2)!=0])
        return int(step/self.mag_factor)


    def stitch(self,size=None):
        step=self.step*self.mag_factor
        xmin,xmax=self.border[0][0],self.border[0][1]
        ymin,ymax=self.border[1][0],self.border[1][1]
        z=1500*self.mag_factor
        xnew=(xmax+z-xmin)/self.mag_factor
        ynew=(ymax+z-ymin)/self.mag_factor
        canvas=np.zeros((int(ynew),int(xnew),3))
        step=self.step*self.mag_factor
        for x in range(xmin,xmax+step,step):
            for y in range(ymin,ymax+step,step):
                filename=self.name+'_'+str(x)+'_'+str(y)+'.'+self.fext
                p=cv2.imread(os.path.join(self.patch_path,filename))
                xsize,ysize,_=p.shape
                xnew=int((x-xmin)/self.mag_factor)
                ynew=int((y-ymin)/self.mag_factor)
                canvas[ynew:ynew+ysize,xnew:xnew+xsize,:]=p
        return canvas.astype(np.uint8)
