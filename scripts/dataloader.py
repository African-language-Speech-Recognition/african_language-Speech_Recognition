import imp
import librosa
import numpy as np
import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
from os.path import exists
import seaborn as sns
from functools import reduce
import os,sys
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from logger import logger
import torch
import torchaudio
import random

import IPython.display as ipd
import warnings
import wave, array
warnings.filterwarnings("ignore")
    
# function to load audiofile 

class DataLoader

    def load_audio(self,language,wav_type='train',start=0,stop=None,files=None):
        amharic_train_audio_path = f'../data_test/amharic_{wav_type}_wav/'
        amharic_wav_folders = os.listdir(path=amharic_train_audio_path)
        swahili_wavs = []
        transformed_files=[]
        if files:
            transformed_files =  [x+'.wav' for x in files]
            
        for wav_file in amharic_wav_folders[start:len(amharic_wav_folders) if not stop else stop]:
            try:
                if len(transformed_files) > 1:
                    if wav_file in transformed_files:
                        loaded_files.append(librosa.load(amharic_train_audio_path+wav_file, sr=44100))
                    else:
                        loaded_files.append(librosa.load(amharic_train_audio_path+wav_file, sr=44100))

                except Exception as e:
                    logger.error(e)
        result = []
        for file in loaded_files:
            audio,rate = file
            result.append((audio,rate,self.get_duration(audio,rate)))
        return result
    