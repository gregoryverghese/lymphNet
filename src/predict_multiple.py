#call predict multiple times
from predict import test_predictions
from tensorflow.keras.models import load_model
import datetime
import pandas as pd
import os
import argparse
import ast

def predict_multiple(model, test_path, save_path, feature,step,normalize,means,std):

    test_files = os.listdir(os.path.join(test_path,'images','germinal'))    

    print(test_files)
    # Count the number of files
    #for i in len(test_files):
    #    results[i] = []

    results={str(i): [] for i,v in enumerate(test_files)}
    
    results['threshold'] = [i / 100 for i in range(50, 101, 5)]
    #results['threshold'] = [i / 100 for i in range(90, 95, 5)]
    results['dices']=[]
    print(results)
    #results['14']=[]
    #results['32']=[]
    #results['38']=[]
    #results['42']=[]
    #results['48']=[]
    #go from 0.5 to 1 in increments of 
    # Loop through threshold values from 0.5 to 1 with 0.05 increments
    for threshold in results['threshold']:
        #curr_save_path=(save_path+'_'+str(threshold))
        curr_save_path=f"{save_path}_{threshold:.2f}"
        os.makedirs(curr_save_path,exist_ok=True)
        print(f"Running test_predictions with threshold: {threshold}")
        print("saving to: ",curr_save_path)
        dices=test_predictions(model,
                         test_path,
                         curr_save_path,
                         args.feature,
                         threshold,
                         1024,
                         step,
                         normalize,
                         means,    
                         std
                         )
        results['dices'].append(dices)
        print("num dices:",len(dices))
        print("num test files:",len(test_files))
        for j,_ in enumerate(dices):
            results[str(j)].append(dices[j])
        #results['1'].append(dices[1])
        #results['2'].append(dices[2])
        #results['3'].append(dices[3])
        #results['4'].append(dices[4])

    print(results)
    #results_df = pd.DataFrame(results)
    #results_df.to_csv(os.path.join(save_path,'results-multiple.csv'))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='model inference')
    ap.add_argument('-mp','--model_path',required=True,help='path to trained model')
    ap.add_argument('-tp','--test_path',required=True,help='path to test images and masks')
    ap.add_argument('-sp','--save_path',required=True,help='experiment folder for saving results')
    ap.add_argument('-f','--feature',required=True,help='morphological feature')
    ap.add_argument('-s','--step',default=512,help='sliding window size')
    ap.add_argument('-n','--normalize',nargs='+',default=["Scale","StandardizeDataset"],help='normalization methods')
    ap.add_argument('-cm','--means',nargs='+',default=[0.791, 0.663, 0.825],help='channel mean')
    ap.add_argument('-cs','--std',nargs='+', default=[0.164,  0.248, 0.178],help='channel std')
    args=ap.parse_args()
    cm=[float(x) for x in args.means]
    cs=[float(x) for x in args.std] 
    print(cm)
    print(cs)
    print(args.model_path)
    model=load_model(args.model_path)
    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H:%M')
    save_path= os.path.join(args.save_path,curr_date+"-multiple")
    os.makedirs(save_path,exist_ok=True)
    save_path= os.path.join(save_path,'predictions')
    os.makedirs(save_path,exist_ok=True)

    predict_multiple(model,
                     args.test_path,
                     save_path,
                     args.feature,
                     int(args.step),
                     args.normalize,
                     cm,
                     cs)



