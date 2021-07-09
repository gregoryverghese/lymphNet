import os
import json

import cv2
import openslide
import numpy as np
import pandas as pd
import operator as op
from itertools import chain


class Slide(openslide.OpenSlide):
    """
    WSI object that enables annotation overlay

    wrapper around openslide.OpenSlide class loads WSIs 
    and provides additional functionality to generate 
    masks and mark with user loaded annotations

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
        

        self.dims = self.dimensions
        self.name = os.path.basename(filename)[:-4]
        self.draw_border=draw_border
        self._border=None
        self._slide_mask=None

    @property
    def border(self):
        if self._border is None:
            self._border=[[0,self.dims[0]],[0,self.dims[1]]]
        return self._border

    @border.setter
    def border(self,value):
        #Todo: if two values we treat as max_x and max_y
        assert(len(value)==4)

    @property
    def draw_border(self):
        return self.draw_border

    @draw_border.setter 
    def draw_border(self, value):
        if value:            
            self._border=self.get_border()
            #self.draw_border=value
        elif not value:
            self._border=[[0,self.dims[0]],[0,self.dims[1]]]
            #self.draw_border=value
        else:
            raise TypeError('Boolean type required')
        
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
    def get_border(self, space=100):
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
            self.draw_border()
            x, y = self.border
        
        x_min, x_max=x
        y_min, y_max=y

        x_size=x_max-x_min
        y_size=y_max-y_min

        #Adjust sizes - treating 0 as base
        #256 size in mag 0 is 512 in mag 1
        x_size=int(x_size/Slide.MAG_fACTORS[mag])
        y_size=int(y_size/Slide.MAG_fACTORS[mag])
        
        if scale_border:
            x_size = Slide.resize_border(x_size, factor, threshold, operator)
            y_size = Slide.resize_border(y_size, factor, threshold, operator)
        
        print('x_size:{}'.format(x_size))
        print('y_size:{}'.format(y_size))

        region=self.read_region((x_min,y_min),mag,(x_size, y_size))
        mask=self.slide_mask()[x_min:x_min+x_size,y_min:y_min+y_size]

        return region, mask

    
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
        self._patch_names=[]
        self._annotations=annotations

    @property
    def masks(self):
        return self._masks
    
    @property
    def patches(self):
        return self._patches

    @property
    def annotations(self):
        return self._annotations

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
    
    
    def _remove_edge_cases(self,x,y,xmin,xmax,ymin,ymax):
        x_size=int(self.size[0]*self.mag_factor*.5)
        y_size=int(self.size[1]*self.mag_factor*.5)
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
            

    #TODO discard patches at borders that do not match size
    def generate_patches(self,step, mode='sparse', masks=False):

        self._patches=[]
        self._masks=[]
        #DO we leave the step multiplier here
        step=step*self.mag_factor
        xmin, xmax = self.slide.border[0][0], self.slide.border[0][1]
        ymin, ymax = self.slide.border[1][0], self.slide.border[1][1]

        for x, y in self.patching(step,xmin,xmax,ymin,ymax):
            if self._remove_edge_cases(x,y,xmin,xmax,ymin,ymax):
                continue

            name=self.slide.name+'_'+str(x)+'_'+str(y)
            self.patches.append({'name':name, 'x':x,'y':y})
            if masks:
                mask=self.slide._slide_mask[y:y+self.size[0],x:x+self.size[1]]
                if mode=='focus':
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
        mask=self.slide_mask[y-y_size:y+y_size,x-x_size:x+x_size][:,:,0]
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
            masks_generator=self.extract_masks()

        for patch,x,y in self.extract_patches(): 
            if np.mean(patch)>190:
                continue
            patchstatus=self.saveimage(patch,patchpath,self.slide.name,x,y)
            if mask_flag: 
                mask,x,y=next(mask_generator)
                maskstatus=self.saveimage(mask,maskpath,self.slide.name,x,y)

