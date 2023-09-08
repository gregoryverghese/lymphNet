"""
feature_extractor.py
"""

import os
import glob
import random
import argparse
import itertools
 
import pandas as pd
import cv2
import torch
#import staintools
from PIL import Image
import numpy as np
import torchvision.models as models
from torchvision import transforms as T

#from lmdb_data import LMDBRead, LMDBWrite



class FeatureGenerator():
    encoders= {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50
              }
    def __init__(
            self,
            model_name,
            model_path,
            encoder_name='resnet18',
            contrastive=None):

        self.model_path=model_path
        self.encoder_name=encoder_name
        self.model = model_name
        

    @property
    def model(self):
        return self._model


    @model.setter
    def model(self,value):
        self._model=getattr(self,'_'+value)()
        

    @property
    def encoder(self):
        encoder=FeatureGenerator.encoders[self.encoder_name]
        return encoder
        

    @property
    def checkpoint_dict(self):
        return torch.load(self.model_path,map_location=torch.device('cpu'))


    #Load pretrained MoCO model from
    def _moco(self):
        state_dict=self.checkpoint_dict['state_dict']
        model=self.encoder()
        model.load_state_dict(state_dict,strict=False)
        #remove final linear layer
        model=torch.nn.Sequential(*list(model.children())[:-1])
        return model

    
    #Load pretrained simclr model from 
    #https://github.com/ozanciga/self-supervised-histopathology/blob/main/README.md
    def _ciga(self):
        state_dict=self.checkpoint_dict['state_dict']
        for k in list(state_dict.keys()):
            k_new=k.replace('model.', '').replace('resnet.', '')
            state_dict[k_new] = state_dict.pop(k)

        model=self.encoder()
        model_dict=model.state_dict()
        state_dict={k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        return model


    def _vgg16(self):
        net=models.vgg16(pretrained=True)
        model=torch.nn.Sequential(*(list(net.children())[:-1]))
        return model
        

    def _simclr(self):
        for k in list(checkpoint_dict.keys()):
            if k.startswith('backbone'): 
                if not k.startswith('backbone.fc'):
                    checkpoint_dict[k[len(layer_name):]] = checkpoint_dict[k]
            del checkpoint_dict[k]

        model=self.encoder()
        model.load_state_dict(checkpoint_dict,strict=False)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))    
        return model


    def forward_pass(self, image): 
        transform=T.Compose([T.ToTensor()])
        image=Image.fromarray(image)
        image=transform(image)
        image=torch.unsqueeze(image,0)
        with torch.no_grad():
            features=self.model(image)
        features=torch.squeeze(features)

        return features


