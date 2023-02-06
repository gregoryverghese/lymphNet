
"""
finetune.py: select ideal values for variables
"""

from predict import test_predictions


import os
import glob
import argparse
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


def find_optimal_value(model,test_path,save_path,feature="germinal", threshold=0.5, step=1024):
    print("\n\n**** HOLLY ****   in find_optimal_value")
    #loop over 
    dicescores = []
    data = {}
    data['names'] = ["14.90610 C L2.11.png","32.90577 C L1.2.png","38.90861 LA L2.png","42.90144 C L2.2.png","48.90239 C L1.3.png"]
    multiple = True
    if(multiple):
        thresholds = [round(x,3) for x in np.linspace(0.75, 0.95, num=9)]
        for i in thresholds:    
            print("\n\n****START HOLLY threshold: "+str(i))
            #set up paths for models, training curves and predictions
            curr_save_path = os.path.join(save_path,'finetune_'+feature+'_'+str(step)+'_'+str(i))
            os.makedirs(curr_save_path,exist_ok=True)

            save_predict_path=os.path.join(curr_save_path,'predictions')
            os.makedirs(save_predict_path,exist_ok=True)
            print(i)
            print(round(i,2))
            dices=test_predictions(model,
                                test_path,
                                curr_save_path,
                                feature,
                                i,
                                step,normalize=["Scale"],channel_means=[0.633,0.383,0.659],channel_std=[0.143,0.197,0.19])
        
            dicescores.append(dices)
        
            #convert list of [1,] tensors to list of floats so easy to view in csv
            da = [dt.numpy() for dt in dices]
            dices_vals = [(list(db))[0] for db in da]
            data[str(i)] = dices_vals

            print("****END\n")

    else:
        print("**** !!!!  single test\n\n")
        curr_save_path = os.path.join(save_path,'finetune_'+feature+'_'+str(step)+'_'+str(threshold))
        os.makedirs(curr_save_path,exist_ok=True)

        save_predict_path=os.path.join(curr_save_path,'predictions')
        os.makedirs(save_predict_path,exist_ok=True)

        dices=test_predictions(model,
                                test_path,
                                curr_save_path,
                                feature,
                                threshold,                                
                                step,normalize=["Scale"],channel_means=[0.633,0.383,0.659],channel_std=[0.143,0.197,0.19])

 
        dicescores.append(dices)
        da = [dt.numpy() for dt in dices]
        dices_vals = [(list(db))[0] for db in da]
        data[str(threshold)] = dices_vals



    print(dicescores)


    df = pd.DataFrame.from_dict(data, orient='columns')
    print(df.head())
    df.to_csv(os.path.join(save_path,'dicescores.csv'))

    print("\n****END find_optimal_value \n")


if __name__=='__main__':
    print("in finetune.py")
    ap=argparse.ArgumentParser(description='model inference')
    ap.add_argument('-mp','--model_path',required=True,help='path to trained model')
    ap.add_argument('-tp','--test_path',required=True,help='path to test images and masks')
    ap.add_argument('-sp','--save_path',required=True,help='experiment folder for saving results')
    ap.add_argument('-f', '--feature', default="germinal",help='feature')
    ap.add_argument('-th','--threshold',default=0.95,help='activation threshold')
    ap.add_argument('-s','--step',default=1024,help='sliding window size')
    
    args=ap.parse_args()
    print("parsed args")
    print("mp: "+args.model_path)
    print("tp: "+args.test_path)
    print("op: "+args.save_path)
    print("f: "+args.feature)
    print("th: "+str(args.threshold))
    print("s: "+str(args.step))

    model=load_model(args.model_path)
    print("loaded model")
    find_optimal_value(model,
                     args.test_path,
                     args.save_path,
                     args.feature,
                     float(args.threshold),
                     int(args.step))




