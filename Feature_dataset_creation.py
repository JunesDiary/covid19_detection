#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa as lr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from PIL import Image
import pathlib
from glob import glob
import csv 

from openpyxl import load_workbook

#taking only the audio files i.e. ogg and webm
data_dir = "E:\Seminar Conferences\Cough Based Disease Detection\Cough analysis using ML\publicdataset"
audiofiles = glob(data_dir + '/*.ogg') 
#+ glob(data_dir + '/*.webm')

#counting audio files
len(audiofiles)


# In[ ]:


file = open("E:\Seminar Conferences\Cough Based Disease Detection\Cough analysis using ML\Spectral_Feature.csv", 'a', newline='')

with file:
    to_appendhead = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    writer = csv.writer(file)
    writer.writerow(to_appendhead.split())



for i in range(0, len(audiofiles)):
    y, sr = lr.load(audiofiles[i])
    time = np.arange(0, len(y))/ sr
    
    filename = os.path.basename(f'E:\Seminar Conferences\Cough Based Disease Detection\Cough analysis using ML\publicdataset\{audiofiles[i]}')
    
    #finding the root mean squared Energy
    rmse = lr.feature.rms(y=y)
    #finding the chroma feature 
    chroma_stft = lr.feature.chroma_stft(y=y, sr=sr)
    #finding the spectral centroid
    spec_cent = lr.feature.spectral_centroid(y=y, sr=sr)
    #finding the spectral bandwidth
    spec_bw = lr.feature.spectral_bandwidth(y=y, sr=sr)
    #finding the spectral rolloff
    rolloff = lr.feature.spectral_rolloff(y=y, sr=sr)
    #finding the zero crossing rate
    zcr = lr.feature.zero_crossing_rate(y)
    #finding the mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    
    file = open("E:\Seminar Conferences\Cough Based Disease Detection\Cough analysis using ML\Spectral_Feature.csv", 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
            
Spectral_Feature.close()


# In[4]:





# In[ ]:





# In[ ]:





# In[ ]:




