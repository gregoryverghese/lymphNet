import pandas as pd
import os
import glob
from shutil import copyfile,copy2

patients=list(pd.read_csv('patients2.csv')['patients'])

wsipath='/SAN/colcc/WSI_LymphNodes_BreastCancer/hdd1/UNMARKED SLIDES'

totalImages=[]
test=[]

for path, subdirs, files in os.walk(wsipath):
    for name in files:
        print(name)
        check=[str(p) for p in patients if str(p) in name]
        test.append(check)
        if any([str(p) for p in patients if str(p) in name]):
            totalImages.append(os.path.join(path,name))


print(len(totalImages))
test=pd.DataFrame(list((set([t[0] for t in test if t!=[]]))))

test.to_csv('patients2.csv')



dst = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/testing/triple_negative'

for t in totalImages:
    copy2(t, dst)



