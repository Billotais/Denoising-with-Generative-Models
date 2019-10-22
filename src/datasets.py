
import os

import torchaudio
from torch.utils.data import Dataset, TensorDataset

from utils import sliding_window


# The identity dataset, loads one file 
class AudioIDDataset(Dataset):
    def __init__(self, filename, window, stride, samples, start=0):
        waveform, sample_rate = torchaudio.load(filename)
        input = waveform[0]
        inputs = sliding_window(input, window, stride)
        self.x = inputs[start:start+samples, None, :]
        
    def __getitem__(self, index):
        return (self.x[index], self.x[index])

    def __len__(self):
        return len(self.x)


class AudioUpScalingDataset(Dataset):
    def __init__(self, filename, window, stride, compressed_rate, target_rate, size=-1, start=0):

        
        os.system('cp '+ filename + ' /tmp/vita/original.wav')
        # Get the compressed data = input
        # Compress it and then upsample at the same rate as the target so the network works
        os.system('sox /tmp/vita/original.wav -r ' + str(compressed_rate) + ' /tmp/vita/compressed.wav')
        os.system('sox  /tmp/vita/compressed.wav -r ' + str(target_rate) + ' /tmp/vita/source.wav')

        waveform_compressed, _ = torchaudio.load('/tmp/vita/source.wav')

        self.x = waveform_compressed[0]
        self.x = sliding_window(self.x, window, stride)
        self.x = self.x[start:start+size, None, :]
        #self.x = self.x[start:start+samples, None, :]
        
        # Get the target data

        os.system('sox '+ filename + ' -r ' + str(target_rate) + ' /tmp/vita/target.wav')

        waveform_target, _ = torchaudio.load('/tmp/vita/target.wav')

        self.y = waveform_target[0]
        self.y = sliding_window(self.y, window, stride)
        self.y = self.y[start:start+size, None, :]
        #self.y = self.y[start:start+samples, None, :]

        #os.system('rm -rf /tmp/vita')
 
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


class AudioWhiteNoiseDataset(Dataset):
    def __init__(self, filename, window, stride, rate,  size=-1):

        
        os.system('cp '+ filename + ' /tmp/vita/original.wav')

        # Get the compressed data = input
        # Compress it and then add some white noise to the audio
        os.system('sox /tmp/vita/original.wav -r ' + str(rate) + ' /tmp/vita/compressed.wav')
        os.system('sox /tmp/vita/compressed.wav -p synth whitenoise vol 0.01 | sox -m /tmp/vita/compressed.wav - /tmp/vita/source.wav')

        waveform_noisy, _ = torchaudio.load('/tmp/vita/source.wav')

        self.x = waveform_noisy[0]
        self.x = sliding_window(self.x, window, stride)
        self.x = self.x[:size, None, :]
        #self.x = self.x[start:start+samples, None, :]
        
        # Get the target data

        waveform_target, _ = torchaudio.load('/tmp/vita/compressed.wav')

        self.y = waveform_target[0]
        self.y = sliding_window(self.y, window, stride)
        self.y = self.y[:, None, :]
        #self.y = self.y[start:start+samples, None, :]

        #os.system('rm -rf /tmp/vita')
 
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)
