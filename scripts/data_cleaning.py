"""import your liberaries here"""
import importlib
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
import wave, array
warnings.filterwarnings("ignore")



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
    
    def convert_channels(self,file1, output):
       
        print("monon to stereo started")

        ifile = wave.open(file1)
        print(ifile.getparams())
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
        assert comptype == 'NONE'  # Compressed not supported yet
        array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
        left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
        ifile.close()
        stereo = 2 * left_channel
        stereo[0::2] = stereo[1::2] = left_channel
        ofile = wave.open(output, 'w')
        ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
        try:
            ofile.writeframes(stereo)
            print("succesffully converted to stereo")
        except Exception as e:
            logger.error(e)
        ofile.close()


         
