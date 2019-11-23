import os

import torchaudio
from torch.utils.data import Dataset, TensorDataset
from pysndfx import AudioEffectsChain

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
noises = ["whitenoise", "pinknoise", "brownnoise", "tpdfnoise"]

from random import gauss, seed
seed(111)

def sample(filename_x, filename_y, audio_rate, file_rate):
    name_x = ".".join(filename_x.split('.')[:-1]) + "-rate_" + str(audio_rate) + "_x." + filename_x.split('.')[-1]
    # Get the compressed data = input
    # Compress it and then upsample at the same rate as the target so the network works
    os.system('sox ' + filename_x + ' -r ' + str(audio_rate) + ' tmp/compressed.wav')
    os.system('sox tmp/compressed.wav -r ' + str(file_rate) + " " + name_x)
    name_y = filename_y.split('.')[0] + "-rate_" + str(file_rate) + "_y." + filename_y.split('.')[1]
    # Compress it at the target rate directly
    os.system('sox ' + filename_y + ' -r ' + str(file_rate) + " " + name_y)

    os.system("rm tmp/compressed.wav -f")

    return name_x, name_y

def noise(filename_x, filename_y, variance, noise_type, intensity):
    if noise_type not in noises: 
        print(noise_type + " is not a valid noise type !")
        return filename_x, filename_y

    intensity += gauss(0, variance)
    name = ".".join(filename_x.split('.')[:-1]) + "-" + noise_type + "_" + str(intensity) + "." + filename_x.split('.')[-1]
    os.system("sox " + filename_x + " tmp/noise.wav synth " + noise_type + " vol " + str(intensity) + " && sox -m " + filename_x + " tmp/noise.wav " + name + "")

    os.system("rm tmp/noise.wav -f")
    return name, filename_y


def reverb(filename_x, filename_y, variance=0,reverberance=80, hf_damping=100, room_scale=100, stereo_depth=100, pre_delay=40, wet_gain=0, wet_only=False):

    reverberance += gauss(0, variance)
    hf_damping += gauss(0, variance)
    room_scale += gauss(0, variance)
    stereo_depth += gauss(0, variance)
    pre_delay += gauss(0, variance)
    wet_gain += gauss(0, variance)
    
    fx = (
        AudioEffectsChain()
        # .highshelf()
        .reverb(reverberance=80, hf_damping=100, room_scale=100, stereo_depth=100, pre_delay=40, wet_gain=0, wet_only=False)
        # .phaser()
        # .delay()
        # .lowshelf()
    )
    name = ".".join(filename_x.split('.')[:-1]) + "-reverb." + filename_x.split('.')[-1]
    fx(filename_x, name)
    return name, filename_y

<<<<<<< HEAD
def reverb_room(filename_x, filename_y):
    corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]
=======


>>>>>>> ce4001ad402a4ac5102db57fc3d738ad58cfd8b9

def preprocess(run_name, filename, arguments):
    # now = datetime.now()
    # date_time = now.strftime("%m_%d_%Y-%H:%M:%S")

    folder = "out/" + run_name + "/tmp"
    
    os.system('cp '+ filename + ' ' + folder + '/' + filename.split('/')[-1])

    file_x = folder + "/" + filename.split('/')[-1]
    file_y = folder + "/" + filename.split('/')[-1]

    for command in arguments:
        
        args = command.strip().split(' ')
        #print(args)
        if args[0] == "sample":
            file_x, file_y = sample(file_x, file_y, *args[1:])
        if args[0] in noises:
            file_x, file_y = noise(file_x, file_y, *args)
        if args[0] == "reverb": # "reverb sample_rate *reverb_args"
            file_x, file_y = reverb(file_x, file_y, *args[2:])
            file_x, file_y = sample(file_x, file_y, args[1], args[1])
    
    # file_x_wav = file_x.split(".")[-2] + ".wav"
    # file_y_wav = file_y.split(".")[-2] + ".wav"
    # os.system("sox " + file_x + " " + file_x_wav)
    # os.system("sox " + file_y + " " + file_y_wav)

    # return file_x_wav, file_y_wav
    return file_x, file_y



# Apply phaser and reverb directly to an audio file.

# print(sample("in.wav", 5000, 10000))
# print(whitenoise("in.wav", 0.005))

# print(preprocess("in.wav", ["sample 5000 10000", "whitenoise 0.005"]))
#print(preprocess("in.wav", ["reverb"]))



