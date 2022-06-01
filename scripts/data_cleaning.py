"""import your liberaries here"""
import imp
import librosa
import numpy as np
import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
from os.path import exists
import seaborn as sns
import os,sys
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import torch
import torchaudio
import random



class Clean:
    
    
    def __init__(self,df = None):
        """initialize the cleaning class"""
        self.df = df
        print("Successfully initialized clean class")
    
    # function to openfile   
    def openfile(self, audio_file):
        """
        - Gezahegne
        - to open audio file and return the signal and sampling rate
        """
        sig, sr = torchaudio.load(audio_file)
        #print(sig, sr)
        return (sig, sr)
    
    
    # function to convert signal to a fixed length 
    # function to get features from the audio 
    # function to load audiofile 
    # function to get lebel of the data 
    # function to read_text 
    # function to  data from training and testing file paths 
    # function to get duration of a file
    # function to convert_channels to stereo
    
    
         
