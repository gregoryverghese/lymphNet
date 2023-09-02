import os
import glob
import argparse

import cv2
import openslide
import numpy as np

from wsi_parser import WSIParser
from utilities import TissueDetect, visualise_wsi_tiling


def parse_wsi(args, wsi_path):
     
    wsi = openslide.OpenSlide(wsi_path)
    detector = TissueDetect(wsi)
    thumb=detector.tissue_thumbnail


    """
    cv2.imwrite(os.path.join(args.save_path,'thumb.png',thumb))
    tis_mask = detector.detect_tissue(3)
    border = detector.border(3)
    cv2.imwrite(os.path.join(args.save_path,'mask.png'), tis_mask)    
    parser = WSIParser(wsi,args.tile_dims,border)
    num = parser.tiler(args.stride)
    print('Tiles: {}'.format(num))

    parser.filter_tissue(
        tis_mask,
        label=1,
        threshold=0.5)
    
    print(f'Sampled tiles: {parser.number}') 
    visualise_wsi_tiling(
            wsi,
            parser,
            args.vis_path,
            viewing_res=3
            )

    if args.parser != 'tiler':
        parser.extract_features(args.parser,
                args.model_path)
    
    parser.to_lmdb(
            os.path.join(args.tile_path,
                         os.path.basename(wsi_path)), 
            map_size=2e9
    )
    
    #patches=remove_black(patches)
    #patches=remove_blue(patches)
    
    #patches.save(save_path,mask_flag=False)
    #patches.to_lmdb(save_path,map_size=10e9)
    #print('                         ',end='\r')

    #if feature_method is not None:
        #pass
        #getattr(ParseWSI,ars.method+'feature')

    #if args.database:
        #lmdb.save
    """

if __name__=='__main__':
    
    ap=argparse.ArgumentParser()

    ap.add_argument('-wp','--wsi_path',
            required=True, help='whole slide image directory')

    ap.add_argument('-ap','--annotation_path',
            required=True, help='annotations directory')

    ap.add_argument('-sp','--save_path',
            required=True, help='directoy to write tiles and features')
    
    ap.add_argument('-s','--stride',default=1024,
            help='distance to step across WSI')

    ap.add_argument('-ml','--mag_level',default=0,
            help='magnification level of tiling')

    ap.add_argument('-td','--tile_dims',default=1024,
            help='dimensions of tiles')
    
    ap.add_argument('-tf','--tfrecords',default=False,
            help='store tiles as tf records')
    
    args=ap.parse_args()
    
    args.tile_path = os.path.join(args.save_path, dir_)
    args.vis_path = os.path.join(args.save_path, 'vis')
    args.mask_path = os.path.join(args.save_path, 'masks')

    os.makedirs(args.tile_path, exist_ok=True)
    os.makedirs(args.vis_path, exist_ok=True)
    os.makedirs(args.mask_path, exist_ok=True)

    wsi_paths=glob.glob(os.path.join(args.wsi_path,'*'))
    ann_paths=glob.glob(os.path.join(ann_path,'*'))

    for f in wsi_paths:
        #parse_wsi(args,f) 
        
        wsi_path, wsi_ext = os.path.splittext(f)
        name=os.path.basename(wsi_path)
        print(f'Slide name: {name}')
        ann_path=[a for a in ann_paths if name in a]
    
        if len(ann_path)==0:
            print(f'No {name} annotation file')
            continue

        #wsi_save_path=os.path.join(save_path,name)
        #os.makedirs(curr_save_path,exist_ok=True)








