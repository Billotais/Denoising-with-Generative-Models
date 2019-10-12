#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

from network import Net
from utils import make_train_step, make_test_step, concat_list_tensors
from datasets import AudioIDDataset, AudioUpScalingDataset

from progress.bar import Bar
import matplotlib.pyplot as plt

#%%
net = Net(8, verbose = 0)

#%%
# Open data


filename = "/mnt/Data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
#filename = "../MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
train_data = AudioUpScalingDataset(filename, window=1024, stride=128, samples=10000, compressed_rate=5000, target_rate=10000, start=10000)
#train_data = AudioIDDataset(filename, window=1024, stride=128, samples=10000, start=10000)
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
        # print(x_batch)
        # print(y_batch)
        
        loss = train_step(x_batch, y_batch)
        losses.append(loss)
        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('loss.png')
        
#%%

test_data = AudioUpScalingDataset(filename, window=1024, stride=1024, samples=1000, compressed_rate=5000, target_rate=10000)
#test_data = AudioIDDataset(filename, window=1024, stride=1024, samples=1000)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
test_step = make_test_step(net, nn.MSELoss())

#%%
losses = []
outputs = []
for x_test, y_test in test_loader:
    
    loss, y_test_hat = test_step(x_test, y_test)
    losses.append(loss)
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig('loss_test.png')
    outputs.append(y_test_hat)

#%%

print(outputs)

out = concat_list_tensors(outputs)

out_formated = out.reshape((1, out.size()[2]))
print(out_formated.size())
torchaudio.save("out.wav", out_formated, 10000, precision=16, channels_first=True)

#%%


# Essayer d'overfit avec un petit bout de musique
# Ca devrait donner un bon résultat
# Essayer de faire l'identité sans les skip connections
# ssh nice = 19 pour pas prendre trop de cpu sur le server
# pytorch seulement choisir 1 seul gpu
# mettre le code pour qu'il puisse passer de gpu a cpu automatiquement en fonction d'ou on est

# Trouver la source du clic-clic-clic
# Faire un entrainnement long
# voir si y'a pas de décalage entre in et out.


