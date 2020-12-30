#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien
"""

import argparse

import DataLoad
import weightComputation as WC
import pytorchEnv as pe
MAXLEN=500

parser=argparse.ArgumentParser()
parser.add_argument("--bert",help='if selected, model used will be bert')
parser.add_argument("--RNN",help='if selected, model used will be RNN')
args = parser.parse_args()


biblioVidJson=DataLoad.importDataFromJson("bibliovid.json")
Xbib,Ycat,Yspe,label_YCat_dict,label_YSpe_dict=DataLoad.GetDataWithAbstractBibliovid(biblioVidJson)


TrainBib,Cat_TrainBib,ValBib,Cat_ValBib,TestBib,Cat_TestBib=DataLoad.ProcessSplitBibliovidData(Xbib,Ycat,catDict=label_YCat_dict)

YCatWeight=WC.getWeight(Cat_TrainBib)

if args.bert :
    X_trainBib, YCat_trainBib = pe.prepare_textsBert(TrainBib, Cat_TrainBib,MAXLEN)
    X_validBib, YCat_validBib = pe.prepare_textsBert(ValBib, Cat_ValBib,MAXLEN)
    X_testBib, YCat_testBib = pe.prepare_textsBert(TestBib, Cat_TestBib,MAXLEN)
    
    train_loader,valid_loader,test_loader=pe.getLoader(X_trainBib,YCat_trainBib,X_validBib,YCat_validBib,X_testBib,YCat_testBib)
    
    bertModel=pe.BertClassifierSequence()
    
    pe.fit(bertModel,epochs=5,train_loader=train_loader,valid_loader=valid_loader,lr=2e-5,class_weights=YCatWeight)
if args.RNN :
    