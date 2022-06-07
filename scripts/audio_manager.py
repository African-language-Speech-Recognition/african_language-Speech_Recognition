   
#from log_help import App_Logger
import os
import sys

import IPython.display as ipd
import librosa  # for audio processing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from librosa.core import audio
from numpy.lib.stride_tricks import as_strided
from scipy.io import wavfile  # for audio processing
from os.path import exists
import wave
import array
#sys.path.insert(0, "../logs/")
sys.path.append(os.path.abspath(os.path.join("..")))



class AugmentAudio:
    
    def __init__(self) -> None:
        print("class initialized succesfully")
        pass

    def add_noise(self, data: np.array, noise_factor: float) -> np.array:
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        augmented_data = augmented_data.astype(type(data[0]))

        return augmented_data

    def add_time_shift(self,
                       data: np.array,
                       sampling_rate: int,
                       shift_max: float,
                       shift_direction: str) -> np.array:
        shift = np.random.randint(sampling_rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0

        return augmented_data
    
    def change_pitch(self, data, sampling_rate, pitch_factor):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def change_speed(self, data, speed_factor):
        return librosa.effects.time_stretch(data, speed_factor)
    
    # transcription loader
    def tran_loader(filename):
        name_to_text = {}
        with open(filename, encoding="utf-8")as f:
            f.readline()
            for line in f:
                name = line.split("</s>")[1]
                name = name.replace('(', '')
                name = name.replace(')', '')
                name = name.replace('\n', '')
                name = name.replace(' ', '')
                text = line.split("</s>")[0]
                text = text.replace("<s>", "")
                name_to_text[name] = text
            return name_to_text
    # mono to stereo converter    
    #import wave, array
    def change_channel_to_stereo(file1, output):
        try:
            ifile = wave.open(file1)
            print(ifile.getparams())
            # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
            (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
            assert comptype == 'NONE'  # Compressed not supported yet
            array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
            left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
            ifile.close()

            stereo = 2 * left_channel
            stereo[0::2] = stereo[1::2] = left_channel

            ofile = wave.open(output, 'w')
            ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
            print(ofile.getnchannels())
            ofile.writeframes(stereo.tobytes())
            ofile.close()
            return ofile.getnchannels()
        except Exception as e:
            print(e)
            
    #Resize audio sig
    def resize_audio(audio: np.array, size: int) -> np.array:
            
            resized = librosa.util.fix_length(audio, size, axis=1)
            print(f"Audio resized to {size} samples")
            return resized
        
        
    #FFt Builder and ploter

    def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
        """
        - FFT
        
        """
        assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

        window = np.hanning(fft_length)[:, None]
        window_norm = np.sum(window**2)

        
        scale = window_norm * sample_rate

        trunc = (len(samples) - fft_length) % hop_length
        x = samples[:len(samples) - trunc]

        # "stride trick" reshape to include overlap
        nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
        nstrides = (x.strides[0], x.strides[0] * hop_length)
        x = as_strided(x, shape=nshape, strides=nstrides)

        # window stride sanity check
        assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

        # broadcast window, compute fft over columns and square mod
        x = np.fft.rfft(x * window, axis=0)
        x = np.absolute(x)**2

        # scale, 2.0 for everything except dc and fft_length/2
        x[1:-1, :] *= (2.0 / scale)
        x[(0, -1), :] /= scale

        freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

        return x, freqs

    spe_samples, frequency = spectrogram(samples)
    print(frequency)
    print(spe_samples)


    def plot_spectrogram_feature(vis_spectrogram_feature):
        # plot the normalized spectrogram
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(vis_spectrogram_feature, cmap=plt.cm.jet, aspect='auto')
        plt.title('Spectrogram')
        plt.ylabel('Time')
        plt.xlabel('Frequency')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        # plt.savefig('spectogramfeature.png')
        
    #Audio augmentation    
        
    def plot_spec(data: np.array, sr: int) -> None:
        '''
        Function for plotting spectrogram along with amplitude wave graph
        '''

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].title.set_text(f'Shfiting the wave by Times {sr/10}')
        ax[0].specgram(data, Fs=2)
        ax[1].set_ylabel('Amplitude')
        ax[1].plot(np.linspace(0, 1, len(data)), data)
        # fig.savefig('spectogramamplitude.png')
    wav_roll = np.roll(samples, int(sample_rate/10))
    plot_spec(data=wav_roll, sr=sample_rate)
    ipd.Audio(wav_roll,rate=sample_rate)


    # Generate Meta Data
    #path = train_changed_wav_location
    def meta_data(trans, path):
        target = []
        features = []
        mode=[]
        rmse=[]
        spec_cent=[]
        spec_bw=[]
        rolloff=[]
        zcr=[]
        mfcc=[]
        rate=[]
        filenames = []
        duration_of_recordings = []
        for index, k in enumerate(trans):
            if index < 10:
                filename = path + k + ".wav"
                next_file_name = path + k + "changed.wav"
                if exists(filename):
                    # stereo = make_stereo(filename, next_file_name)
                    filenames.append(filename)
                    audio, fs = librosa.load(filename, sr=44100)
                    chroma_stft = librosa.feature.chroma_stft(y = audio, sr = fs)
                    rmse.append(np.mean(librosa.feature.rms(y = audio)))
                    spec_cent.append(np.mean(librosa.feature.spectral_centroid(y = audio, sr = fs)))
                    spec_bw.append(np.mean(librosa.feature.spectral_bandwidth(y = audio, sr = fs)))
                    rolloff.append(np.mean(librosa.feature.spectral_rolloff(y = audio, sr = fs)))
                    zcr.append(np.mean(librosa.feature.zero_crossing_rate(audio)))
                    mfcc.append(np.mean(librosa.feature.mfcc(y = audio, sr = fs)))
                    duration_of_recordings.append(float(len(audio)/fs))
                    rate.append(fs)
                    mode.append('stereo') # if stereo == 1 else 'stereo')
                    lable = trans[k]
                    target.append(lable)
        return filenames, target, duration_of_recordings,mode,rmse,spec_cent,spec_bw,rolloff,zcr,mfcc,rate
        


if __name__ == "__main__":
    audio_augmenter = AugmentAudio()
    


