import os

import torchaudio
from torch.utils.data import Dataset, TensorDataset
from pysndfx import AudioEffectsChain

from datetime import datetime

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


def whitenoise(filename_x, filename_y, intensity):
   
    name = ".".join(filename_x.split('.')[:-1]) + "-white_noise_" + str(intensity) + "." + filename_x.split('.')[-1]
    print(name)
    os.system("sox " + filename_x + " tmp/noise.wav synth whitenoise vol " + str(intensity) + " && sox -m " + filename_x + " tmp/noise.wav " + name + "")



    os.system("rm tmp/noise.wav -f")
    return name, filename_y

def reverb(filename_x, filename_y, reverberance=80, hf_damping=100, room_scale=100, stereo_depth=100, pre_delay=40, wet_gain=0, wet_only=False):


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


def preprocess(run_name, filename, arguments):
    # now = datetime.now()
    # date_time = now.strftime("%m_%d_%Y-%H:%M:%S")

    folder = "tmp/" + run_name
    os.system("mkdir " + folder)
    os.system('cp '+ filename + ' ' + folder + '/' + filename.split('/')[-1])

    file_x = folder + "/" + filename.split('/')[-1]
    file_y = folder + "/" + filename.split('/')[-1]

    for command in arguments:
        
        args = command.strip().split(' ')
        print(args)
        if args[0] == "sample":
            file_x, file_y = sample(file_x, file_y, *args[1:])
        if args[0] == "whitenoise":
            file_x, file_y = whitenoise(file_x, file_y, *args[1:])
        if args[0] == "reverb": # "reverb sample_rate *reverb_args"
            file_x, file_y = reverb(file_x, file_y, *args[2:])
            file_x, file_y = sample(file_x, file_y, args[1], args[1])



    return file_x, file_y



# Apply phaser and reverb directly to an audio file.

# print(sample("in.wav", 5000, 10000))
# print(whitenoise("in.wav", 0.005))

# print(preprocess("in.wav", ["sample 5000 10000", "whitenoise 0.005"]))
#print(preprocess("in.wav", ["reverb"]))



