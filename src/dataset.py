
from torch.utils.data import Dataset, TensorDataset
import torchaudio
from helpers import pytorch_rolling_window

class AudioDataset(Dataset):
    def __init__(self, filename, window, stride, samples):
        waveform, sample_rate = torchaudio.load(filename)
        input = waveform[0]
        inputs = pytorch_rolling_window(input, window, stride)
        self.x = inputs[:samples, None, :]
        
    def __getitem__(self, index):
        return (self.x[index], self.x[index])

    def __len__(self):
        return len(self.x)