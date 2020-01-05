
import os

import torchaudio
from torch.utils.data import Dataset, TensorDataset

from utils import sliding_window

from preprocessing import preprocess

# Not used anymore, kept for "archiving"
class AudioDataset(Dataset):
    def __init__(self, run_name, filename, window, stride, arguments, size=-1, start=0):

        # Create the two files using the preprocessing pipeline
        in_file, out_file = preprocess(run_name, filename, arguments.split(','))

        # Load the input
        waveform_in, _ = torchaudio.load(in_file)
        self.x = waveform_in[0]
        # Split it using the sliding window
        self.x = sliding_window(self.x, window, stride)
        # Only keep necessary samples
        self.x = self.x[start:start+size, None, :]

        # Load the output
        waveform_out, _ =  torchaudio.load(out_file)
        self.y = waveform_out[0]
         # Split it using the sliding window
        self.y = sliding_window(self.y, window, stride)
        # Only keep necessary samples
        self.y = self.y[start:start+size, None, :]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


        
""" # The identity dataset, loads one file 
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
        
        in_file = sample(filename, compressed_rate, target_rate)
        out_file = sample(filename, target_rate, target_rate)

        waveform_in, _ = torchaudio.load('tmp/' + in_file)

        self.x = waveform_in[0]
        self.x = sliding_window(self.x, window, stride)
        self.x = self.x[start:start+size, None, :]

        waveform_out, _ =  torchaudio.load('tmp/' + out_file)

        self.y = waveform_out[0]
        self.y = sliding_window(self.y, window, stride)
        self.y = self.y[start:start+size, None, :]
 
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


class AudioWhiteNoiseDataset(Dataset):
    def __init__(self, filename, window, stride, rate,  size=-1):

        
        os.system('cp '+ filename + ' tmp/original.wav')

        # Get the compressed data = input
        # Compress it and then add some white noise to the audio
        os.system('sox tmp/original.wav -r ' + str(rate) + ' tmp/compressed.wav')
        os.system('sox tmp/compressed.wav -p synth whitenoise vol 0.01 | sox -m tmp/compressed.wav - tmp/source.wav')

        waveform_noisy, _ = torchaudio.load('tmp/source.wav')

        self.x = waveform_noisy[0]
        self.x = sliding_window(self.x, window, stride)
        self.x = self.x[:size, None, :]
        
        # Get the target data

        waveform_target, _ = torchaudio.load('tmp/compressed.wav')

        self.y = waveform_target[0]
        self.y = sliding_window(self.y, window, stride)
        self.y = self.y[:, None, :]
        #self.y = self.y[start:start+samples, None, :]

        #os.system('rm -rf /tmp/vita')
 
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x) """


