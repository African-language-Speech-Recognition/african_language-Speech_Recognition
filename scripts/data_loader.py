import math, random
import torch
import torchaudio
from torchaudio import transforms

class DataLoader:

    def __init__(self) -> None:
        pass

    def load(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)