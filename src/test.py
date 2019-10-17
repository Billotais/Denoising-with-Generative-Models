import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt

input, _ = torchaudio.load("/tmp/vita/source.wav")
output, _ = torchaudio.load("/tmp/vita/target.wav")

# plt.figure()
# plt.plot(input.t()[:100].numpy())
# plt.figure()
# plt.plot(output.t()[:100].numpy())
# plt.show()


print(input.t()[:20].numpy())
print(output.t()[:20].numpy())
