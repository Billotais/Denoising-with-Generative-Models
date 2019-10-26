import torchaudio
import torch
import math
def metric(file1, file2, size, metric):


    audio1, _ = torchaudio.load(file1)
    audio2, _ = torchaudio.load(file2)
    if metric == 'snr':
        return snr(audio1[:,:size], audio2[:,:size])
    elif metric == 'lsd':
        return lsd(audio1[:,:size], audio2[:,:size])


def snr(x, y):

    return 10*math.log10(torch.norm(y, p=2) / torch.dist(x, y, p=2))
    

def lsd(x, y, channel=0):

    spectogram = torchaudio.transforms.Spectrogram(n_fft=1024) # value of paper
    X = torch.log(spectogram(x))
    X_hat = torch.log(spectogram(y))

    #X = (channels, k freq, l frames)
    K = X.size()[1] # number of frequencies
    L = X.size()[2] # number of frames

    #print(str(L) + " frames, " +  str(K) + " frequencies")
    sum_l = 0
    for l in range(L):
        sum_k = 0
        for k in range(K):
            sum_k += pow(X[channel, k, l] - X_hat[channel, k, l],2)
        sum_l += math.sqrt(sum_k / K)
    return sum_l / L
        

    
    


# y = reference signal
# x = approximation signal
x = "/tmp/vita/source.wav"
y = "/tmp/vita/target.wav"
# x = "/home/lois/Documents/EPFL/MA3/VITA/src/out/overfit_sr_32_epochs_of_500/target.wav"
# y = "/home/lois/Documents/EPFL/MA3/VITA/src/out/overfit_sr_32_epochs_of_500/overfit_sr.wav"
print(metric(x, y, 30000, 'lsd'))