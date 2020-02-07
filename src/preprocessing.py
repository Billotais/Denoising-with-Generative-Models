import os
from datetime import datetime
from random import gauss, seed

import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra
import torchaudio
from pysndfx import AudioEffectsChain
from scipy.io import wavfile
from scipy.signal import fftconvolve
from torch.utils.data import Dataset, TensorDataset

noises = ["whitenoise", "pinknoise", "brownnoise", "tpdfnoise"]


# Change the sampling rates of both the input and output file
def sample(filename_x, filename_y, audio_rate, file_rate):
    
    name_x = filename_x.split('.')[0] + "-rate_" + str(int(audio_rate)) + "_x." + filename_x.split('.')[-1]
    # Get the compressed data = input
    # Compress it and then upsample at the same rate as the target so the network works
    os.system('sox ' + filename_x + ' -r ' + str(audio_rate) + ' tmp/compressed.wav')
    os.system('sox tmp/compressed.wav -r ' + str(file_rate) + " " + name_x)

    name_y = filename_y.split('.')[0] + "-rate_" + str(int(file_rate)) + "_y." + filename_y.split('.')[1]
    # Compress it at the target rate directly
    os.system('sox ' + filename_y + ' -r ' + str(file_rate) + " " + name_y)

    os.system("rm tmp/compressed.wav -f")

    return name_x, name_y

# Add noise of type "noise_type", with a given intensity and maybe some variance
def noise(filename_x, filename_y, noise_type, variance, intensity):
    if noise_type not in noises: 
        print(noise_type + " is not a valid noise type !")
        return filename_x, filename_y

    intensity += gauss(0, float(variance))
    name = ".".join(filename_x.split('.')[:-1]) + "-" + noise_type + "_" + str(intensity) + "." + filename_x.split('.')[-1]
    os.system("sox " + filename_x + " tmp/noise.wav synth " + noise_type + " vol " + str(intensity) + " && sox -m " + filename_x + " tmp/noise.wav " + name + "")

    os.system("rm tmp/noise.wav -f")
    return name, filename_y

# Add some reverberation using pysndfx
def reverb(filename_x, filename_y, variance=0,reverberance=80, hf_damping=100, room_scale=100, stereo_depth=100, pre_delay=40, wet_gain=0, wet_only=False):

    # add some variance
    reverberance += gauss(0, float(variance))
    hf_damping += gauss(0, float(variance))
    room_scale += gauss(0, float(variance))
    stereo_depth += gauss(0, float(variance))
    pre_delay += gauss(0, float(variance))
    wet_gain += gauss(0, float(variance))
    

    name = ".".join(filename_x.split('.')[:-1]) + "-reverb." + filename_x.split('.')[-1]

    fx = (
        AudioEffectsChain().reverb(reverberance=80, hf_damping=100, room_scale=100, stereo_depth=100, pre_delay=40, wet_gain=0, wet_only=False)
    )
    

    fx(filename_x, name)
    return name, filename_y


def scale(file, rate):
   
    print(file)
    name = file.split('.')[0] + "-rate_" + str(int(rate)) + "." + file.split('.')[1]
    # Compress it at the target rate directly
    os.system('sox ' + file + ' -r ' + str(rate) + " " + name)


    return name, name

def preprocess(run_name, filename, arguments, test):


    folder = "out/" + run_name + "/tmp"
    os.system('cp '+ filename + ' ' + folder + '/' + filename.split('/')[-1])
    # print('cp '+ filename + ' ' + folder + '/' + filename.split('/')[-1])
    
    # print(os.system('ls ' + folder))
    # print(folder)
    file_x = folder + "/" + filename.split('/')[-1]
    file_y = folder + "/" + filename.split('/')[-1]

    for command in arguments:
        
        args = command.strip().split(' ')
        
        if args[0] == "sample": # "sample input_rate target_rate"
            file_x, file_y = sample(file_x, file_y, *[float(i) for i in args[1:]])
        if args[0] in noises: #" noisetype variance intensity"
            file_x, file_y = noise(file_x, file_y, args[0], *[float(i) for i in args[1:]])
        if args[0] == "reverb": # "reverb sample_rate *reverb_args"
            file_x, file_y = reverb(file_x, file_y, *[float(i) for i in args[2:]])
            file_x, file_y = sample(file_x, file_y, args[1], args[1])
        if args[0] == "scale":
            file_x, file_y = scale(file_x, args[1])
    os.system('rm '+ folder + '/' + filename.split('/')[-1])


    if test: # Copy the test files for easier metrics evaluation
        # os.system('cp ' + file_x + " " + "out/" + run_name + "/in.wav")
        # os.system('cp ' + file_y + " " +  "out/" + run_name + "/target.wav")
        os.system('sox ' + file_x + " --channels 1 " + "out/" + run_name + "/in.wav")
        os.system('sox ' + file_y + " --channels 1 " +  "out/" + run_name + "/target.wav")

    return file_x, file_y
