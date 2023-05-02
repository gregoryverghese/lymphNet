#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
xml_to_json_annotation.py: converts xml annotations (eg created in Aperio) to json that can be imported to QuPath

input: xml file in the format

output: json file in a format acceptable to QuPath

NB: this script will close all annotations by ensuring that the first vertex/coord matches the last in a region

'''
import glob
import os
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np

from pathlib import Path
from shapely.geometry import Polygon
from shapely.strtree import STRtree



def convert_multiple(input_path, output_path):
    #read all images names in directories
    input_paths=glob.glob(os.path.join(input_path,'*'))
    print("converting: "+str(len(input_paths))+" xml files")
    for ip in input_paths:
        xml_to_json(ip, output_path)
    print("done converting")


def xml_to_json(xml_file_path, json_file_path):
    # Parse the Aperio Imagescope XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Initialize the QuPath JSON format
    qupath_json = {"type": "FeatureCollection","features": []}

    # Loop through all the regions of interest (ROIs) in the Aperio XML file
    for annot in root.iter('Annotation'):
        ##if the annotation does not have any vertices then skip
        ##need to include this to avoid empty elements being created in QuPath
        vertices_element = annot.find('Regions/Region/Vertices')
        if vertices_element is None:
            print("no vertices")
            continue

        #print("region"+annot.attrib.get("Name"))
        roi = {"type": "Feature", "color": "#FF0000", "class": "", "properties": {}}
        roi_properties = {}

        # Get the name of the ROI
        roi_properties["object_type"] = "annotation"
        classification = {}
        classification["name"] = annot.attrib.get("Name")
        if classification["name"]=="GERMINAL CENTRE":
            classification["name"]="GC"
        elif classification["name"]=="SINUS":
            classification["name"]="sinus"

        classification["colorRGB"] =-15481405
        roi_properties["classification"] = classification

        # Get the description of the ROI
        #roi_properties["Description"] = region.attrib.get("Memo")

        # Get the vertices of the ROI
        regions = []
        for region in annot.iter('Vertices'):
            vertices = []
            for vertex in region.iter('Vertex'):
                vertices.append([float(vertex.attrib.get("X")), float(vertex.attrib.get("Y"))])
            
            #need to make sure the polygons are closed
            #if last coord does not match first coord then add a final coord at the end
            if vertices[0] != vertices[-1]:
                vertices.append(vertices[0])
            regions.append(vertices)
        roi["geometry"] = {"type": "Polygon", "coordinates": regions}

        # Add the ROI properties to the ROI
        roi["properties"] = roi_properties

        # Add the ROI to the QuPath JSON format
        
        qupath_json["features"].append(roi)

    #file_name, file_extension = os.path.splitext(os.path.basename(xml_file_path))
    # Write the QuPath JSON format to a file
    with open(os.path.join(json_file_path,os.path.basename(xml_file_path))+".json", 'w') as f:
        json.dump(qupath_json, f,indent=4)



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to xml files to be converted' )
    ap.add_argument('-op', '--output_path', required=True, help='path where jsons should be saved')
    ap.add_argument('-m', '--method', default='multiple', help='can be multiple or single')
    ap.add_argument('-fn', '--file_name', default='file.xml', help='FILENAME.xml used for converting a single file within a folder')

    args = ap.parse_args()

    os.makedirs(args.output_path,exist_ok=True)
    if(args.method == 'multiple'):
        convert_multiple(args.input_path, args.output_path)
    elif(args.method=='single'):
        print("converting a single file")
        #we are assuming that the user has specified an xml file even if the extension is not .xml
        xml_to_json(os.path.join(args.input_path,args.file_name),args.output_path)
    else:
        print("invalid method - should be one of single/multiple")


