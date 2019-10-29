import torchaudio
import torch
import math
import argparse
def metric(file1, file2, size, metric):
    audio1, _ = torchaudio.load(file1)
    audio2, _ = torchaudio.load(file2)
    
    if (size == -1): size = min(audio1.size()[1], audio2.size()[1])
    #print(audio1.size())
    if metric == 'snr':
        return snr(audio1[:,:size], audio2[:,:size])
    elif metric == 'lsd':
        return lsd(audio1[:,:size], audio2[:,:size])

# Look at the noise of the signal
def snr(x, y):
     
    return 10*math.log10(pow(torch.norm(y, p=2),2) /pow(torch.dist(x, y, p=2), 2))
    #return 10*math.log10(torch.norm(y, p=2) /torch.dist(x, y, p=2))
    
# look at the presence of specific frequencies
def lsd(x, y, channel=0):

    spectogram = torchaudio.transforms.Spectrogram(n_fft=1024) # value of paper

    X = torch.log10(spectogram(x))
    X_hat = torch.log10(spectogram(y))
    # the spectrogram is the magnitude squared of the stft
    # the paper used log |stft|^2
    # => we just need to use the log
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="low quality signal", type=str)
    ap.add_argument("--target", help="reference signal",type=str)
    ap.add_argument("--generated", help="genereated signal",type=str)
    ap.add_argument("-c", "--count", type=int,default=-1)

    args = ap.parse_args()
    variables = vars(args)
    count = variables['count']
    source = variables['source']
    target = variables['target']
    generated = variables['generated']
    print("SNR (higher is better)")
    print("Original " + str(metric(source, target, count, 'snr')))
    print("Improved " + str(metric(generated, target, count, 'snr')))
    print("LSD (lower is better)")
    print("Original " + str(metric(source, target, count, 'lsd')))
    print("Improved " + str(metric(generated, target, count, 'lsd')))


if __name__ == "__main__":
    main()

# overfit 512 epochs of 100 : 

# SNR (higher is better)
# Original 11.82354842681325
# Improved 2.6116020354635543
# LSD (lower is better)
# Original 5.747375236221171
# Improved 3.534085585130723

# overfit 32 epochs of 500 : 
# SNR (higher is better)
# Original 11.82354842681325
# Improved 2.576323071715057
# LSD (lower is better)
# Original 5.747375236221171
# Improved 4.158397121584476



        

    
    


# y = reference signal
# x = approximation signal
# x = "/tmp/vita/source.wav"
# y = "/tmp/vita/target.wav"
x = "/home/lois/Documents/EPFL/MA3/VITA/src/out/overfit_sr_32_epochs_of_500/target.wav"
y = "/home/lois/Documents/EPFL/MA3/VITA/src/out/overfit_sr_32_epochs_of_500/overfit_sr.wav"
#print(metric(x, y, 30000, 'lsd'))
