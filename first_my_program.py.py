from sklearn.neighbors 
import KNeighborsClassifier 
from sklearn.metrics
import roc_curve
from sklearn.svm 
import SVC
import requests

import sys
import os
import glob
from subprocess 
import run,PIPE
import pandas as pd
import json

#import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing 
import LabelEncoder
from sklearn.model_selection 
import train_test_split
from sklearn 
import svm
from sklearn.metrics 
import roc_auc_score
from sklearn.linear_model 
import LogisticRegression
from sklearn.metrics 
import roc_auc_score

#from ubidots 
import ApiClient
import json
import urllib.request
import urllib.parse
from flask 
import jsonify
from flask_cors 
import CORS, cross_origin
import requests
from flask 
import Flask,
app,requestapp = Flask( name )
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-
Type'face_flag = 0

@app.route('/heart', methods=['GET','POST'])
@cross_origin()
def login():
if request.method == 'POST':

# print("called")
print(request.json['data_set'])
result =
input(request.json['data_set']
return result
def
data_analysis():
    global regressor
dataset=pd.read_csv('diabetes3.csv')
dataset.head()

columnlist=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Ag
e','Outcome']
for i in columnlist:
labelencoder_X=LabelEncoder()
dataset[i]=labelencoder_X.fit_transform(dataset[i])
print(dataset)
X=dataset.iloc[:,0:-1].values
X.shape
Y=dataset.iloc[:,7:].values
Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
regressor = LogisticRegression()
H=regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
classifier = SVC(kernel = 'linear', C = 1, random_state = 0, probability = True)
classifier.fit(X_train, Y_train)
probs = classifier.predict_proba(X_test)
#print(probs)
# keep probabilities for the positive outcome
onlyprobs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(Y_test,probs)
print('AUC: %.3f' % auc)
# clf = svm.SVC(kernel='linear',


C=1.0)Y_pred =
regressor.predict(X_test)
Y_pred
print("ACC:", roc_auc_score(Y_test, Y_pred))

return
regressordef
input(s):
    data_analysis()
patient_condition='Not
found'x = list(map(float,
s))
print(x)
# prediction
# 102 0 0 160000 0 1 0 0 --0(Normal)
# 102 0 0 60000 0 1 0 0 --1(Critical)
Y_pred = regressor.predict([x])
# Y_pred
print(Y_pred)
if Y_pred[0]== 1:
    a = 1
# patient_condition='Patient is critical'
patient_condition='Critical'

print("Patient iscritical")
else:
b = 0


# patient_condition='Patient is normal'

patient_condition='Normal'
print(patient_condition)
     return ({"data" :patient_condition})
if name == ' main ':
app.run(host='0.0.0.0', port=5008) 
WEB APPLICATION CODE

import React, { useState, useEffect } from"react";

import Grid from '@material-ui/core/Grid';

import { makeStyles } from '@material-ui/core/styles';

import Typography from '@material-ui/core/Typography';

import TextField from '@material-ui/core/TextField';

import FormControlLabel from '@material-ui/core/FormControlLabel';

import Checkbox from '@material-ui/core/Checkbox';

import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';

import FormControl from '@material-ui/core/FormControl';
import Box from "@material-ui/core/Box";

import Button from '@material-ui/core/Button';

import FormGroup from '@material-ui/core/FormGroup';
import axios from 'axios';

import Snackbar from '@material-ui/core/Snackbar';

import MuiAlert from '@material-ui/lab/Alert';
import sha256 from "sha256"
import initJWTService from 'jwt-service';
import { CircularProgress } from '@material-ui/core';
import SavedPatientDetails from '../SavedPatientDetails/SavedPatientDetails'
function Alert(props) {

           return <MuiAlert elevation={6} variant="filled" {...props} />;
}
const useStyles = makeStyles((theme) => ({
formControl: 
{
margin:
theme.spacing(1),
minWidth: 600,
},