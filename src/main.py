#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

from network import Net
from helpers import pytorch_rolling_window, make_train_step
from dataset import AudioDataset

from progress.bar import Bar
import matplotlib.pyplot as plt

#%%
net = Net(8, verbose = 0)

#%%
# Open data


filename = "/mnt/Data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"

train_data = AudioDataset(filename, window=1024, stride=128, samples=10000)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
train_step = make_train_step(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=0.0001))

#%%
n_epochs = 1
losses = []
for epoch in range(n_epochs):
    for x_batch, y_batch in Bar('Processing', suffix='%(percent)d%%').iter(train_loader):
        # the dataset "lives" in the CPU, so do our mini-batches
        # therefore, we need to send those mini-batches to the
        # device where the model "lives"
        # x_batch = x_batch.to(device)
        # y_batch = y_batch.to(device)

        
        loss = train_step(x_batch, y_batch)
        losses.append(loss)
        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('loss.png')
        
#%%
x = next(iter(train_loader))[0]
print(x)
net.eval()
y1 = net(x)
net.eval()
y2 = net(x)

print(y1 == y2)