#%%
import pyroomacoustics as pra
room = pra.ShoeBox([9, 7.5, 3.5], fs=44100, absorption=0.35, max_order=17)
#%%
# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
from scipy.io import wavfile
_, audio = wavfile.read('in.wav')


my_source = pra.SoundSource([2.5, 3.73, 1.76], signal=audio, delay=1.3)

# place the source in the room
room.add_source(my_source)

#%%
# # define the location of the array
import numpy as np
R = np.c_[
    [6.3, 4.87, 1.2],  # mic 1
    [6.3, 4.93, 1.2],  # mic 2
    ]

# the fs of the microphones is the same as the room
mic_array = pra.MicrophoneArray(R, room.fs)

# finally place the array in the room
room.add_microphone_array(mic_array)

#%%
room.compute_rir()

# plot the RIR between mic 1 and source 0
import matplotlib.pyplot as plt
plt.plot(room.rir[1][0])
plt.show()

#%%
room.simulate()

# plot signal at microphone 1
plt.plot(room.mic_array.signals[1,:])