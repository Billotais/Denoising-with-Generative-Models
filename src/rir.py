#%%
import pyroomacoustics as pra
import numpy as np
#%%
# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
from scipy.io import wavfile
fs, signal = wavfile.read("input.wav")
#room = pra.ShoeBox([9, 7.5, 3.5], fs=10000, absorption=0.35, max_order=17)

corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]
room = pra.Room.from_corners(corners, fs=fs, max_order=1, absorption=0.2)
room.extrude(2.)
room.add_source([1., 1., 0.5], signal=signal)

# add two-microphone array
R = np.array([[3.5, 3.6], [2., 2.], [0.5,  0.5]])  # [[x], [y], [z]]
room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
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

wavfile.write("out_reverb.wav", fs, room.mic_array.signals[0,:])

# %%
