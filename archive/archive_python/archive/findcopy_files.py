from pathlib import Path
import shutil
import os
import glob

files = []
ndpis = []
xmlPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/xml/*'
ndpiPath = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/ndpi'
ndpiDest = '/SAN/colcc/WSI_LymphNodes_BreastCancer/Greg/data/ndpi/images'
xmls = glob.glob(xmlPath)

for xml in xmls:
    print(xml)
    ndpi = xml[53:-4]+'.ndpi'
    ndpis.append(ndpi)

for ndpi in ndpis:
    for filename in Path(ndpiPath).rglob(ndpi):
        files.append(filename)

for f in files:
    shutil.copy(f, ndpiDest)
