import os
import glob
import argparse

import cv2
import openslide
import numpy as np

from wsi_parser import WSIParser
from slide import Slide, Annotations
from utilities import ( 
    TissueDetect, 
    visualise_wsi_tiling,
    average_stack_masks
)
from tfrecords_io import TFRecordWrite 


def parse_wsi(args, wsi_path, annotate, mask):
    
    wsi = Slide(
        wsi_path, mask = mask, annotations = annotate)
    
    #wsi.mask = mask
    print(f'WSI dimensions: {wsi.dimensions}')
    detector = TissueDetect(wsi)
    thumb = detector.tissue_thumbnail
    tis_mask = detector.detect_tissue(6)
    #cv2.rectangle(thumb,(x,y),(x+w,y+h),(255,0,0),5)
    #cv2.imwrite(os.path.join(args.save_path,'thumb.png'),thumb)
    #border = detector.border(3) 
    #cv2.imwrite(os.path.join(args.save_path,'tis.png'),tis_mask*255)
    
    border = wsi.get_border(space=1000)
    (x1,x2),(y1,y2) = border
    
    f=lambda x: (int(x[0]/8),int(x[1]/8))
    new_border=list(map(f,border))
    (x1_,x2_),(y1_,y2_) = new_border
    cv2.rectangle(thumb,(x1_,y1_),(x2_,y2_),(255,0,0),3)
    cv2.imwrite(os.path.join(args.vis_path,args.name+'_thumb.png')
                ,thumb)

    #for c in classes:
    #annotate_feature = Annotations(ann_path, source='qupath', labels=['GC'])
    #annotations = annotate._annotations 
    #wsi_feature = Slide(curr_path, annotations = annotate_feature)
        
    parser = WSIParser(wsi, args.tile_dims, border, 2)
    num = parser.tiler(args.stride)

    #print('Tiles: {}'.format(num))
    parser.filter_tissue(
            tis_mask,
            label=1,
            threshold=0.05
            )
        
    print(f'Sampled tiles: {parser.number}') 
    visualise_wsi_tiling(
            wsi,
            parser,
            os.path.join(args.vis_path,args.name+'.png'),
            viewing_res=3
                )

    if args.parser != 'tiler':
        parser.extract_features(args.parser,
                    args.model_path)
    
    classes = [1]
    for c in classes:
        parser.slide.filter_mask_classes(c)
        #parser.save(args.tile_path, mask_flag=True)
        parser.to_tfrecords(
            os.path.join(args.tile_path) 
            )


if __name__=='__main__':
    
    ap=argparse.ArgumentParser()

    ap.add_argument('-wp','--wsi_path',
            required=True, help='whole slide image directory')

    ap.add_argument('-ap','--annotation_paths',
            required=False, help='annotations directory')

    ap.add_argument('-mp', '--mask_paths',
            required=False, help='masks directory')

    ap.add_argument('-sp','--save_path',
            required=True, help='directoy to write tiles and features')
    
    ap.add_argument('-s','--stride',default=512,
            help='distance to step across WSI')

    ap.add_argument('-ml','--mag_level',default=2,
            help='magnification level of tiling')

    ap.add_argument('-td','--tile_dims',default=512,
            help='dimensions of tiles')
    
    ap.add_argument('-tf','--tfrecords',default=False,
            help='store tiles as tf records')

    ap.add_argument('-at','--annotate_type',default='qupath',
            help='software used to generate annotations')

    ap.add_argument('-c', '--consensus', default = 1,
                    type = int, help='mask consensus agreement')

    ap.add_argument('-p','--parser', default='tiler',
            help='type of feature extractor')
    #ap.add_argument('-cl','--classes',default='qupath',
            #help='software used to generate annotations')

    args=ap.parse_args()
    
    args.classes = ['GC','sinus']
    dir_ = 'tfrecords' if args.tfrecords else 'tiles'
    args.tile_path = os.path.join(args.save_path, dir_)
    args.vis_path = os.path.join(args.save_path, 'vis')
    os.makedirs(args.tile_path, exist_ok=True)
    os.makedirs(args.vis_path, exist_ok=True)

    wsi_paths=glob.glob(os.path.join(args.wsi_path,'*'))

    if args.annotation_paths is not None:
        ann_paths=glob.glob(os.path.join(args.annotation_paths,'*'))

    if args.mask_paths is not None:
        mask_paths=glob.glob(os.path.join(args.mask_paths,'*'))
    
    for f in wsi_paths:
        wsi_path, wsi_ext = os.path.splitext(f)
        args.name = os.path.basename(wsi_path)
        print(f'Parsing {args.name}') 
        #wsi = openslide.OpenSlide(f)

        if args.annotation_paths is not None:
            ann_path = [a for a in ann_paths if args.name in a]
            if len(ann_path)!=0:
                print('missing annotations')
                continue
            annotate = Annotations(
                ann_path, 
                source = args.annotate_type, 
                labels = args.classes
            )
        else:
            annotate = None

        if args.mask_paths is not None:
            mask_path = [m for m in mask_paths if args.name in m]
            print(mask_path)
            if len(mask_path)==0:
                continue
                print('missing mask')
            mask = cv2.imread(mask_path[0])
            mask = average_stack_masks(mask,args.consensus)
            print('mask shape', mask.shape)
        else:
            mask = None
        
        parse_wsi(args, f, annotate, mask)


        #wsi_save_path=os.path.join(save_path,name)
        #os.makedirs(curr_save_path,exist_ok=True)








