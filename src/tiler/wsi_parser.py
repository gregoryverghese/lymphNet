import os
import glob
import json
import random

import numpy as np
import cv2
import seaborn as sns
from matplotlib.path import Path
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
import seaborn as sns
from itertools import chain
import operator as op

from utilities import mask2rgb
#from exceptions import StitchingMissingPatches
from lmdb_io import LMDBWrite
from feature_extractor import FeatureGenerator
#from pyslide.io.tfrecords_io import TFRecordWrite


class WSIParser():
    def __init__(
            self,
            slide, 
            tile_dim,
            border,
            mag_level=0,  
            ):

        super().__init__()
        self.slide = slide
        self.mag_level = mag_level
        self.tile_dims = (tile_dim,tile_dim)
        #y_size=int(self.size[1]*self.mag_factor*.5)
        self.border = border

        self._x_min = int(self.border[0][0])
        self._x_max = int(self.border[0][1])
        self._y_min = int(self.border[1][0])
        self._y_max = int(self.border[1][1])
        self._downsample = int(slide.level_downsamples[mag_level])
        self._x_dim = int(tile_dim*self._downsample) 
        self._y_dim = int(tile_dim*self._downsample)
        self._tiles = []
        self._features = []
        self._number = len(self._tiles)
        

    @property
    def number(self):
        return len(self._tiles)

    @property
    def tiles(self):
        return self._tiles

    @tiles.setter
    def tiles(self,value):
        self._tiles=value

    @property
    def features(self):
        return self._features

    @property
    def config(self):
        config={'name':self.slide.name,
                'mag':self.mag_level,
                'size':self.size,
                'border':self.border,
                'number':self._number}
        return config


    def __repr__(self):
        """
        Object representation

        :return str(self.config)
        """
        return str(self.config)

        
    def _remove_edge_case(self,x,y):
        """
        Remove edge cases based on dimensions of patch

        :param x: base x coordinate to test 
        :param y: base y coordiante to test
        :return remove: boolean remove patch or not
        """
        remove=False
        if x+self._x_dim>self._x_max:
           remove=True
        if y+self._y_dim>self._y_max:
            remove=True
        return remove


    def tiler(self, stride=None, edge_cases=False):

        """
        Generate tile coordinates based on border
        mag_level, and stride.

        :param step: integer: step size
        :param mode: sparse or focus
        :param mask_flag: include masks
        :return len(self._patches): Number of patches
        """
       
        stride = self.tile_dims[0] if stride is None else stride
        stride = stride * self._downsample
        self._tiles = []
        for x in range(self._x_min,self._x_max, stride):
            for y in range(self._y_min, self._y_max, stride):
                #if self._remove_edge_case(x,y):
                    #continue
                self._tiles.append((x,y))

        self._number=len(self._tiles)
        return self._number


    def extract_features(self,model_name,model_path):

        encode=FeatureGenerator(model_name,model_path)
        for i, t in enumerate(self.tiles):
            tile = self.extract_tile(t[0],t[1])
            feature_vec=encode.forward_pass(tile)
            self.features.append(feature_vec.detach().numpy())


    def filter_tissue(self,slide_mask,label,threshold=0.5):
        print('greg',slide_mask.shape)
        slide_mask[slide_mask!=label]=0
        slide_mask[slide_mask==label]=1
        tiles=self._tiles.copy()
        for t in self._tiles:
            x,y=(t[1],t[0])
            t_mask=slide_mask[x:x+self._x_dim,y:y+self._y_dim]
            if np.sum(t_mask) < threshold * (self._x_dim * self._y_dim):
                print('removing')
                tiles.remove(t)

        self._tiles = tiles
        print(f'Sampled tiles:{len(self._tiles)}')
        return len(self._tiles)


    def filter_tiles(self,filter_func): 
        """
        Filter tiles using filtering function

        :param filter_func: intensity threshold value
        """
        tiles = self._tiles.copy()

        for i, tile in enumerate(self.extract_tiles()):
            if filter_func(tile):
                tiles.remove(tile)

        print(f'Removed {self.number - len(tiles)} tiles')
        self._tiles = tiles.copy()
        return removed


    def extract_tile(self, x=None, y=None):
        """
        Extract individual patch from WSI.

        :param x: int x coordinate
        :param y: int y coodinate
        :return patch: ndarray patch
        """

        #if we want x,y coordinate of point to be central
        #points in read_region (x-x_size,y-y_size)
        #x_size=int(self.size[0]*self.mag_factor*.5)
        #y_size=int(self.size[1]*self.mag_factor*.5)
        tile=self.slide.read_region((x,y), self.mag_level,
                (self._x_dim, self._y_dim))
        tile=np.array(tile.convert('RGB'))
        return tile


    def extract_tiles(self):
        """
        Generator to extract all tiles

        :yield tile: ndarray tile
        :yield p: tile dict metadata
        """
        for t in self._tiles:
            tile=self.extract_tile(t[0],t[1])
            yield t


    @staticmethod
    def _save_to_disk(
        image, 
        path,
        filename,
        x=None,
        y=None):
        """
        Save tile only if WSI x and y position
        known.

        :param image: ndarray tile
        :param path: path to save
        :param filename: str filename
        :param x: int x coordindate for filename
        :param y: int y coordinate for filename
        """
        assert isinstance(y, int) and isinstance(x, int) 
        filename=filename+'_'+str(x)+'_'+str(y)+'.png'
        image_path=os.path.join(path,filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        status=cv2.imwrite(image_path,image)
        return status
   

    def save(
        self, 
        path, 
        label_dir=False, 
        label_csv=False
        ):

        """
        Extracts and saves down tiles

        :param path: save path
        :param masK_flag: boolean to save masks
        :param label_dir: label directory
        :param label_csv: boolean to save labels in csv
        """
        tile_path=os.path.join(path,'tiles')
        os.makedirs(tile_path,exist_ok=True)
        #CHECK TILE
        for t in self.extract_tiles():
            self._save_disk(tile,path,filename,t[0],t[1])

        if label_csv:
            df=pd.DataFrame(self._tiles,columns=['names','x','y'])
            df.to_csv(os.path.join(path,'labels.csv'))


    def to_lmdb(self, db_path, map_size, write_frequency=100):
        #size_estimate=len(self._patches)*self.size[0]*self.size[1]*3
        print(db_path)
        db_write=LMDBWrite(db_path,map_size,write_frequency)
        db_write.write(self)
            

    def to_tfrecords(
            self, 
            db_path,
            shard_size=0.01,
            unit=1e9
            ):
        TFRecordWrite(db_path,self,shard_size,unit).convert()
