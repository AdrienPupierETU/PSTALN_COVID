#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:28:58 2020

@author: Adrien
"""

import json
import numpy as np

DATA_PATH='../data/'

def importDataFromJson(fileName):
    '''
    @param fileName : the name of the json file to be loaded; need to be in folder DATA_PATH
    @return json : a json object with the data inside
    ''' 
    with open(DATA_PATH+fileName) as file :
        return json.loads(file.read())

def GetDataWithAbstractBibliovid(bibliovidData):
    '''
    @param bibliovidData : a json object loaded from bibliovid file
    
    @return  X_bibliovid     : list of the abstract of the document
             YCat_bibliovid  : list of the categorie of the document
             YCat_bibliovid  : list of the specialty of the document
             label_YCat_dict : dict with all the categorie
             label_YSpe_dict : dict with all the specialty
    '''
    X_bibliovid=[]
    YCat_bibliovid=[]
    YSpe_bibliovid=[]
    Y_AllCat=[]
    Y_AllSpe=[]
    for x in bibliovidData:
      X=x.get('abstract')
      if X is None :
        Y_AllCat.append(x.get('category').get('name'))
        temp=[]
        for y in x.get('specialties'):
          temp.append(y.get('name'))
        Y_AllSpe.append(temp)
      else:
        X_bibliovid.append(X)
        YCat_bibliovid.append(x.get('category').get('name'))
        temp=[]
        for y in x.get('specialties'):
          temp.append(y.get('name'))
        YSpe_bibliovid.append(temp)
    label_YCat_dict={label: i for i, label in enumerate(sorted(set(Y_AllCat)))}
    label_YSpe_dict={label: i for i, label in enumerate(sorted(set(np.concatenate(Y_AllSpe).ravel())))}
    return np.asarray(X_bibliovid,dtype=object),np.asarray(YCat_bibliovid),np.asarray(YSpe_bibliovid),label_YCat_dict,label_YSpe_dict

def ProcessSplitBibliovidData(X,YCat,Yspe=None,catDict=None,speDict=None):
    TrainBib=np.asarray(X[0:277])
    Cat_TrainBib=np.asarray(YCat[0:277])
    
    ValBib=np.asarray(X[277:327])
    Cat_ValBib=np.asarray(YCat[277:327])
    TestBib=np.asarray(X[327:377])
    Cat_TestBib=np.asarray(YCat[327:377])
    if catDict is not None :
        Cat_TrainBib=np.asarray([catDict.get(y) for y in Cat_TrainBib])
        Cat_ValBib=np.asarray([catDict.get(y) for y in Cat_ValBib])
        Cat_TestBib=np.asarray([catDict.get(y) for y in Cat_TestBib])
        
    return TrainBib,Cat_TrainBib,ValBib,Cat_ValBib,TestBib,Cat_TestBib