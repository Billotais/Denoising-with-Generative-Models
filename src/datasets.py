
from torch.utils.data import Dataset, TensorDataset
import torchaudio
from utils import sliding_window
import os


# The identity dataset, loads one file 
class AudioIDDataset(Dataset):
    def __init__(self, filename, window, stride, samples):
        waveform, sample_rate = torchaudio.load(filename)
        input = waveform[0]
        inputs = sliding_window(input, window, stride)
        self.x = inputs[:samples, None, :]
        
    def __getitem__(self, index):
        return (self.x[index], self.x[index])

    def __len__(self):
        return len(self.x)


class AudioUpScalingDataset(Dataset):
    def __init__(self, filename, window, stride, samples, compressed_rate, target_rate):

        
        
        # Get the compressed data = input
        # Compress it and then upsample at the same rate as the target so the network works
        os.system('sox '+ filename + ' -r ' + str(compressed_rate) + ' /tmp/vita/compressed.wav')
        os.system('sox  /tmp/vita/compressed.wav -r ' + str(target_rate) + ' /tmp/vita/source.wav')

        waveform_compressed, _ = torchaudio.load('/tmp/vita/source.wav')

        self.x = waveform_compressed[0]
        self.x = sliding_window(self.x, window, stride)
        self.x = self.x[:samples, None, :]
        
        # Get the target data

        os.system('sox '+ filename + ' -r ' + str(target_rate) + ' /tmp/vita/target.wav')

        waveform_target, _ = torchaudio.load('/tmp/vita/target.wav')

        self.y = waveform_target[0]
        self.y = sliding_window(self.y, window, stride)
        self.y = self.y[:samples, None, :]
 
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)