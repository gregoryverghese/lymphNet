from pathlib import Path
import shutil
import os
import glob

files = []
ndpis = []
xmlPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/xml/*'
ndpiPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/ndpi'
ndpiDest = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/ndpi/images'
ndpiPath2 = '/SAN/colcc/WSI_LymphNodes_BreastCancer/MARKED SLIDES LAST BATCH'

xmls = glob.glob(xmlPath)
ndpiCurr = os.listdir(ndpiDest)

for xml in xmls:
    print(xml)
    ndpi = xml[53:-4]+'.ndpi'
    ndpis.append(ndpi)

for ndpi in ndpis:
    if ndpi not in ndpiCurr:
        for filename in Path(ndpiPath2).rglob(ndpi):
            files.append(filename)

for f in files:
    shutil.copy(f, ndpiDest)
