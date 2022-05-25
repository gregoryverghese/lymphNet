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
#from src.utilities.utils import mask2rgb


__author__='Gregory Verghese'
__email__='gregory.verghese@gmail.com'

def mask2rgb(mask):
    n_classes=len(np.unique(mask))
    colors=sns.color_palette('hls',n_classes)
    rgb_mask=np.zeros(mask.shape+(3,))
    for c in range(1,n_classes):
        t=(mask==c)
        rgb_mask[:,:,0][t]=colors[c][0]
        rgb_mask[:,:,1][t]=colors[c][1]
        rgb_mask[:,:,2][t]=colors[c][2]
    return rgb_mask


class Slide(OpenSlide):
    """
    WSI object that enables annotation overlay wrapper around 
    openslide.OpenSlide class. Generates annotation mask.

    :param _slide_mask: ndarray mask representation
    :param dims dimensions of WSI
    :param name: string name
    :param draw_border: boolean to generate border based on annotations
    :param _border: list of border coordinates [(x1,y1),(x2,y2)]
    """
    MAG_fACTORS={0:1,1:2,2:4,3:8,4:16,5:32}

    def __init__(self,filename,mag=0,annotations=None,
                 annotations_path=None,labels=None,source=None):
        super().__init__(filename)

        self.mag=mag
        self.dims=self.dimensions
        self.name=os.path.basename(filename)
        self._border=None
        if annotations_path is not None:
            self.annotations=Annotations(annotations_path,source=source,
                                         labels=labels,encode=True)
        else:
            self.annotations=annotations

    @property
    def slide_mask(self):
       mask=self.generate_mask((2000,2000))
       mask=mask2rgb(mask)
       return mask


    def generate_mask(self, size=None):
        """
        Generates mask representation of annotations.

        :param size: tuple of mask dimensions
        :return: self._slide_mask ndarray. single channel
            mask with integer for each class
        """
        x, y = self.dims[0], self.dims[1]
        slide_mask=np.zeros((y, x), dtype=np.uint8)
        self.annotations.encode=True
        coordinates=self.annotations.annotations
        keys=sorted(list(coordinates.keys()))
        for k in keys:
            v = coordinates[k]
            v = [np.array(a) for a in v]
            cv2.fillPoly(slide_mask, v, color=k)
        if size is not None:
            slide_mask=cv2.resize(slide_mask, size)
        return slide_mask


    @staticmethod
    def resize_border(dim, factor=1, threshold=None, operator='=>'):
        """
        Resize and redraw annotations border. Useful to trim wsi 
        and mask to specific size

        :param dim: dimensions
        :param factor: border increments
        :param threshold: min/max size
        :param operator: threshold limit
        :return new_dims: new border dimensions [(x1,y1),(x2,y2)]
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
        Generate border around max/min annotation points
        :param space: gap between max/min annotation point and border
        :self._border: border dimensions [(x1,y1),(x2,y2)]
        """
        if self.annotations is None:
            self._border=[[0,self.dims[0]],[0,self.dims[1]]]
        else:
            coordinates=self.annotations._annotations
            coordinates = list(chain(*list(coordinates.values())))
            coordinates=list(chain(*coordinates))
            f=lambda x: (min(x)-space, max(x)+space)
            self._border=list(map(f, list(zip(*coordinates))))
        mag_factor=Slide.MAG_fACTORS[self.mag]
        f=lambda x: (int(x[0]/mag_factor),int(x[1]/mag_factor))
        self._border=list(map(f,self._border))

        return self._border

    #Need to do min size in terms of micrometers not pixels
    def detect_components(self,level_dims=6,num_component=None,min_size=None):
        """
        Find the largest section on the slide
        :param down_factor: 
        :return image: image containing contour around detected section
        :return self._border: [(x1,x2),(y1,y2)] around detected section
        """
        new_dims=self.level_dimensions[6]
        image=np.array(self.get_thumbnail(self.level_dimensions[6]))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur=cv2.bilateralFilter(np.bitwise_not(gray),9,100,100)
        _,thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if num_component is not None:
            idx=sorted([(cv2.contourArea(c),i) for i,c in enumerate(contours)])
            contours=[contours[i] for c, i in idx]
            contours=contours[-num_component:]
        if min_size is not None:
            contours=list(map(lambda x, y: cv2.contourArea(x),contours))
            contours=[c for c in contours if c>min_area]
        borders=[]
        components=[]
        image_new=image.copy()
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            x_scale=self.dims[0]/new_dims[0]
            y_scale=self.dims[1]/new_dims[1]
            x1=round(x_scale*x)
            x2=round(x_scale*(x+w))
            y1=round(y_scale*y)
            y2=round(y_scale*(y-h))
            self._border=[(x1,x2),(y1,y2)]
            image_new=cv2.rectangle(image_new,(x,y),(x+w,y+h),(0,255,0),2)
            components.append(image_new)
            borders.append([(x1,x2),(y1,y2)])            
        return components, borders 
    

    def generate_region(self, mag=0, x=None, y=None, x_size=None, y_size=None,
                        scale_border=False, factor=1, threshold=None, operator='=>'):
        """
        Extracts specific regions of the slide
        :param mag: magnification level
        :param x: min x coordinate
        :param y: min y coordinate
        :param x_size: x dim size 
        :param y_size: y dim size
        :param scale_border: resize border
        :param factor:
        :param threshold:
        :param operator:
        :return: extracted region (RGB ndarray)
        """
        if x is None:
            self.get_border()
            x, y = self._border        
        if x is not None:
            if isinstance(x,tuple):
                if x_size is None:
                    x_min, x_max=x
                    x_size=x_max-x_min
                elif x_size is not None:
                    x_min=x[0]
                    x_max=x_min+x_size
            elif isinstance(x,int):
                x_min=x
                x_max=x+x_size
        if y is not None:
            if isinstance(y,tuple):
                if y_size is None:
                    y_min, y_max=y
                    y_size=y_max-y_min
                elif y_size is not None:
                    y_min=y[0]
                    y_max=y_min+y_size
            elif isinstance(y,int):
                y_min=y
                y_max=y_min+y_size
        if scale_border:
            x_size = Slide.resize_border(x_size, factor, threshold, operator)
            y_size = Slide.resize_border(y_size, factor, threshold, operator)
        if (x_min+x_size)>self.dimensions[0]:
            x_size=self.dimensions[0]-x_min
        if (y_min+y_size)>self.dimensions[1]:
            y_size=self.dimensions[1]-y_min
        x_size_adj=int(x_size/Slide.MAG_fACTORS[mag])
        y_size_adj=int(y_size/Slide.MAG_fACTORS[mag])
        region=self.read_region((x_min,y_min),mag,(x_size_adj, y_size_adj))
        mask=self.generate_mask()[x_min:x_min+x_size,y_min:y_min+y_size]
        return np.array(region.convert('RGB')), mask

    def save(self, path, size=(2000,2000), mask=False):
        """
        Save thumbnail of slide in image file format
        :param path:
        :param size:
        :param mask:
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
    Returns dictionary of coordinates of ROIs. Reads annotation 
    files in either xml and json format and returns a dictionary 
    containing x,y coordinates for each region of interest in the 
    annotation

    :param path: string path to annotation file
    :param annotation_type: file type
    :param labels: list of ROI names ['roi1',roi2']
    :param _annotations: dictonary with return files
                      {roi1:[[x1,y1],[x2,y2],...[xn,yn],...roim:[]}
    """
    def __init__(self, encode=False):
        #self.paths=path if isinstance(path,list) else [path]
        #self.source=source
        #self.labels=labels
        self.encode=encode
        self._annotations={}
        self.labels=[]
        #self._generate_annotations()

    def __repr__(self):
        numbers=[len(v) for k, v in self._annotations.items()]
        df=pd.DataFrame({"classes":self.labels,"number":numbers})
        return str(df)

    @property
    def keys(self):
        return list(self.annotations.keys())

    @property
    def values(self):
        return list(self.annotations.values())

    @property
    def annotations(self):
        if self.encode:
            annotations=self.encode_keys()
            self.encode=False
        else:
            annotations=self._annotations
        return annotations

    @property
    def class_key(self):
        print('labels',self.labels)
        if self.labels == []:
            self.labels=list(self._annotations.keys())
        class_key={l:i+1 for i, l in enumerate(self.labels)}
        return class_key

    @property
    def numbers(self):
        numbers=[len(v) for k, v in self._annotations.items()]
        return dict(zip(self.labels,numbers))


    #def __call__(self, path, source, labels=[]):
        #self.paths=path if isinstance(path,list) else [path]
        #self.source=source
        #self.labels=labels
        #self._generate_annotations()


    def _generate_annotations(self,path, source, labels=[]):
        """
        Calls appropriate method for file type.
        return: annotations: dictionary of coordinates
        """
        #self._annotations={}
        if not isinstance(path,list):
            paths=[path] 
        if source is not None:
            for p in paths:
                annotations=getattr(self,'_'+source)(p)
                for k, v in annotations.items():
                    if k in self._annotations:
                        self._annotations[k].append(v)
                    else:
                        self._annotations[k]=v
            #print(self._annotations)
        if len(labels)>0:
            self._annotations=self.filter_labels(labels)
        else:
            labels=list(self._annotations.keys())
        

    def filter_labels(self, labels):
        """
        remove labels from annotations
        :param labels: label list to remove
        :return annotations: filtered annotation dictionary
        """
        self.labels=self.labels+labels
        keys = list(self._annotations.keys())
        for k in keys:
            if k not in self.labels:
                del self._annotations[k]
        return self._annotations


    def rename_labels(self,names):
        """
        rename annotation labels
        :param names: dictionary {current_labels:new_labels}
        """
        for k,v in names.items():
            self._annotations[v]=self._annotations.pop(k)
        self.labels=list(self._annotations.keys())        
    

    def encode_keys(self):
        """
        encode labels as integer values
        """
        print(self.class_key)
        print(self._annotations.keys())
        annotations={self.class_key[k]: v for k,v in self._annotations.items()}
        return annotations


    def _imagej(self,path):
        """
        Parses xml files
        :param path:
        :return annotations: dict of coordinates
        """
        tree=ET.parse(path)
        root=tree.getroot()
        anns=root.findall('Annotation')
        labels=list(root.iter('Annotation'))
        labels=list(set([i.attrib['Name'] for i in labels]))
        self.labels.extend(labels)
        #self.labels.extend(labels)
        annotations={l:[] for l in labels}
        for i in anns:
            label=i.attrib['Name']
            instances=list(i.iter('Vertices'))
            for j in instances:
                coordinates=list(j.iter('Vertex'))
                coordinates=[[c.attrib['X'],c.attrib['Y']] for c in coordinates]
                coordinates=[[round(float(c[0])),round(float(c[1]))] for c in coordinates]
                annotations[label]=annotations[label]+[coordinates]
        #print('greg', annotations)
        return annotations


    def _asap(self,path):
        """
        Parses _asap files
        :param path:
        :return annotations: dict of coordinates
        """
        tree=ET.parse(path)
        root=tree.getroot()
        ns=root[0].findall('Annotation')
        labels=list(root.iter('Annotation'))
        labels=list(set([i.attrib['PartOfGroup'] for i in labels]))
        annotations={l:[] for l in labels}
        for i in ns:
            coordinates=list(i.iter('Coordinate'))
            coordinates=[[float(c.attrib['X']),float(c.attrib['Y'])] for c in coordinates]
            coordinates=[[round(c[0]),round(c[1])] for c in coordinates]
            label=i.attrib['PartOfGroup']
            annotations[label]=annotations[label]+[coordinates]
        #annotations = {self.class_key[k]: v for k,v in annotations.items()}
        return annotations

    
    def _qupath(self,path):
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


    """
    def _qupath(self,path):
        
        Parses qupath annotation json files
        :param path: json file path
        :return annotations: dictionary of annotations
        
        print(path)
        annotations={}
        with open(path) as json_file:
            j=json.load(json_file)
        for a in j:
            c=a['properties']['classification']['name']
            geometry=a['geometry']['type']
            coordinates=a['geometry']['coordinates']
            if c not in annotations:
                annotations[c]=[]
            if geometry=="LineString":
                points=[[int(i[0]),int(i[1])] for i in coordinates]
            elif geometry=="Polygon":    
                for a2 in coordinates:
                    points=[[int(i[0]),int(i[1])] for i in a2]
            elif geometry=="MultiPolygon":
                for a2 in coordinates:
                    for a3 in a2:
                        points=[[int(i[0]),int(i[1])] for i in a3]
            annotations[c].append(points)
        return annotations
    """ 

    def _json(self,path):
        """
        Parses json file with following structure.
        :param path:
        :return annotations: dict of coordinates
        """
        with open(path) as json_file:
            json_annotations=json.load(json_file)
        
        labels=list(json_annotations.keys())
        self.labels.extend(labels) 
        annotations = {k: [[[int(i['x']), int(i['y'])] for i in v2] 
                       for v2 in v.values()] for k, v in json_annotations.items()}
        return annotations


    def _dataframe(self):
        """
        Parses dataframe with following structure
        """ 
        anns_df=pd.read_csv(path)
        anns_df.fillna('undefined', inplace=True)
        anns_df.set_index('labels',drop=True,inplace=True)
        self.labels=list(set(anns_df.index))
        annotations={l: list(zip(anns_df.loc[l].x,anns_df.loc[l].y)) for l in
                     self.labels}

        annotations = {self.class_key[k]: v for k,v in annotations.items()}
        self._annotations=annotations


    def _csv(self,path):
        """
        Parses csv file with following structure
        :param path: 
        :return annotations: dict of coordinates
        """
        anns_df=pd.read_csv(path)
        anns_df.fillna('undefined', inplace=True)
        anns_df.set_index('labels',drop=True,inplace=True)
        labels=list(set(anns_df.index))
        annotations={l: list(zip(anns_df.loc[l].x,anns_df.loc[l].y)) for l in
                     labels}

        #annotations = {self.class_key[k]: v for k,v in annotations.items()}
        self._annotations=annotations
        return annotations


    def df(self):
        """
        Returns dataframe of annotations.
        :return :dataframe of annotations
        """
        #key={v:k for k,v in self.class_key.items()}
        labels=[[l]*len(self._annotations[l][0]) for l in self._annotations.keys()]
        labels=list(chain(*labels))
        #labels=[key[l] for l in labels]
        x_values=[xi[0] for x in list(self._annotations.values()) for xi in x[0]]
        y_values=[yi[1] for y in list(self._annotations.values()) for yi in y[0]]
        df=pd.DataFrame({'labels':list(labels),'x':x_values,'y':y_values})

        return df


    def save(self,save_path):
        """
        Save down annotations in csv file.
        :param save_path: path to save annotations
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
        x_size=int(self.size[0]*self.mag_factor)
        y_size=int(self.size[1]*self.mag_factor)
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
                continue
                #pass
            self.patches.append({'name':name,'x':x,'y':y})
            if mask_flag:
                #mask=self.slide.slide_mask[y:y+self.size[0],x:x+self.size[1]]
                mask=self.slide.generate_mask()[y:y+self.size[0],x:x+self.size[1]]
                if mode == 'focus':
                    classes = len(np.unique(mask))
                    #print('classes:{}'.format(classes))
                    self._masks.append({'x':x, 'y':y, 'classes':classes})
                    #self.focus()
                else:
                    self._masks.append({'x':x, 'y':y})
        if mode=='focus':
            self.focus()
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
        x_size=int(self.size[0]*self.mag_factor)
        y_size=int(self.size[1]*self.mag_factor)

        patch=self.slide.read_region((x,y), self.mag_level,
                                     (self.size[0],self.size[1]))
        patch=np.array(patch.convert('RGB'))
        return patch


    def extract_patches(self):
        for p in self._patches:
            patch=self.extract_patch(p['x'],p['y'])
            yield patch,p['x'],p['y']


    def extract_mask(self, x=None, y=None):

        x_size=int(self.size[0]*self.mag_factor)
        y_size=int(self.size[1]*self.mag_factor)
        #mask=self.slide.slide_mask()[y-y_size:y+y_size,x-x_size:x+x_size]
        mask=self.slide.generate_mask()[y:y+y_size,x:x+x_size]
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
            if np.mean(patch)>220:
                continue
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
                print(filename)
                p=cv2.imread(os.path.join(self.patch_path,filename))
                xsize,ysize,_=p.shape
                xnew=int((x-xmin)/self.mag_factor)
                ynew=int((y-ymin)/self.mag_factor)
                canvas[ynew:ynew+ysize,xnew:xnew+xsize,:]=p
        return canvas.astype(np.uint8)
