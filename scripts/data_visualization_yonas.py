import librosa
import pandas as pd
import os
import soundfile as sf
import sys

class ResizeAudio:

    def __init__(self) :
        pass

    def transcription_loader(filename):
        name_to_text = {}
        with open (filename, encoding="utf-8")as f:
            f.readline()
            for line in f:
                name=line.split("</s>")[1]
                name=name.replace('(', '')
                name=name.replace(')', '')
                name=name.replace('\n','')
                name=name.replace(' ','')
                text=line.split("</s>")[0]
                text=text.replace("<s>","")
                name_to_text[name]=text
            return name_to_text

    def meta_data(trans, path ):
        target=[]
        features=[]
        filenames=[]
        duration_of_recordings=[]
        for k in trans:
            filename=path+k +".wav"
            filenames.append(filename)
            audio, fs = librosa.load(filename, sr=None)
            duration_of_recordings.append(float(len(audio)/fs))
        
            lable = trans[k]
            target.append(lable)
        return filenames, target,duration_of_recordings

    def generate_dataframe(self):
        transcription=self.transcription_loader("drive/MyDrive/Week_4_10Acad/data/AMHARIC/data/train/trsTrain.txt")

        filenames, target,duration_of_recordings= self.meta_data(transcription,'valid/')
        data=pd.DataFrame({'key': filenames,'text': target, 'duration':duration_of_recordings})
        return data

    def resize_pad_trunc(self,max_ms=4000):
        # aud, max_ms
        df = self.generate_dataframe()

        for i in range(df.shape[0]):
            data = df.loc[i, 'Output']
            
            # print(str(data).split("../")[-1])
            path=os.path.abspath(os.path.join(os.pardir, str(data).split("../")[-1]))
            
            sig, sr =sf.read(path)
            num_rows, sig_len = sig.shape
            max_len = sr // 1000 * max_ms
            if sig_len > max_len:
                # Truncate the signal to the given length
                # sig = sig[:, :max_len]
                trimmed=librosa.util.fix_length(sig, size=max_len)
                
            elif sig_len < max_len:
                # Length of padding to add at the beginning and end of the signal
                
                trimmed=librosa.util.fix_length(sig, size=max_len)

        return trimmed, sr

    

