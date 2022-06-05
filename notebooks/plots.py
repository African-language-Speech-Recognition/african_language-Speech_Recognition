import matplotlib.pyplot as plt
import pandas as pd
import wave
import numpy as np

class Plots:
    def __init__(self) -> None:
        pass

    def visualize(self, path: str):
    
        # reading the audio file
        raw = wave.open(path)
        signal = raw.readframes(-1)
        signal = np.frombuffer(signal, dtype ="int16")
        
        # gets the frame rate
        f_rate = raw.getframerate()
        time = np.linspace(
            0, # start
            len(signal) / f_rate,
            num = len(signal)
        )
        plt.figure(figsize=(15,5))
        plt.title("Sound")
        plt.xlabel("Time")
        plt.plot(time, signal)
        plt.show()