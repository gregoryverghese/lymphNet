import argparse
import pandas as pd
import itertools
import os

from hormad1_patching import Patching,Slide


def main(data_csv_path):

    data=pd.read_csv(data_csv_path)
    patients=list(data['patient'])
    all_labels=[]
    all_patch_names=[]

    for i,p in enumerate(patients):
        label=data.iloc[i]['HORMAD1status']
        scan_path=data.iloc[i]['image_path']
        if '66-25-90172 TUM - 2021-02-10 17.06.50' in scan_path:
            continue
        slide_obj=Slide(scan_path)
        slide_obj.detect_component()
        patch_obj=Patching(slide_obj,mag_level=0,size=(128,128))
        num_patches=patch_obj.generate_patches(1024)
        patches=patch_obj.patches
        patch_obj.save('/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/hormad1/data/patches')
        #patch_names=[p['name'] for p in patches]
        #all_patch_names.append(patch_names)
        #all_labels.append([label]*num_patches)
     
    #all_labels=list(itertools.chain(*all_labels))
    #all_patch_names=list(itertools.chain(*all_patch_names))
    #label_df=pd.DataFrame({'patch_names':all_patch_names,
    #                       'patch_labels': all_labels})

    #label_df.to_csv('/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/hormad1/data/patches/labels.csv')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-p','--data_path',required=True,
                    help='path to csv file containing image details')


    args=vars(ap.parse_args())
    data_csv_path=args['data_path']

    main(data_csv_path)
