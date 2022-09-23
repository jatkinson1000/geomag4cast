from maginput import MagInput
from w_indices import calculate_G, calculate_W, TSParams
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy import signal
from scipy.io import wavfile


mag_input, _, _, _ = MagInput.from_QD_file('../data/QDInput_10day.dat')

dt = 5*60 # seconds
fs = 1 / dt

day = 144 #288 #60*24/5
overlap = 12

freq, time, coeff = signal.spectrogram(mag_input.dens, fs, nperseg=day,noverlap=overlap)

freq = freq / 5 # Hz

plt.pcolor(time, freq, np.log10(coeff), label='dens')

plt.show()



# frequencies = np.logspace(-3,-1,100)
# scale = pywt.frequency2scale('cmor1.5-1.0', frequencies)
#
# coef, freqs_unscaled=pywt.cwt(mag_input.dens,scale,'gaus1')
#
# dt = 5*60 # seconds
# freqs = pywt.scale2frequency('cmor1.5-1.0', freqs_unscaled) / dt



# fig, ax = plt.subplots(2, 1)
#
# ax[0].plot(mag_input.timestamp, mag_input.W1, label='W1')
# ax[0].plot(mag_input.timestamp, mag_input.W2, label='W2')
# ax[0].plot(mag_input.timestamp, mag_input.W3, label='W3')
# ax[0].plot(mag_input.timestamp, mag_input.W4, label='W4')
# ax[0].plot(mag_input.timestamp, mag_input.W5, label='W5')
# ax[0].plot(mag_input.timestamp, mag_input.W6, label='W6')
# ax[0].set_ylim([0, 10])

# TF Plots of each...
#coef, freqs_unscaled=pywt.cwt(mag_input.dens,scale,'gaus1')
#ax[1].pcolor(mag_input.timestamp, np.log10(freqs), coef, label='dens')








# coef, freqs_unscaled=pywt.cwt(mag_input.velo,scale,'gaus1')
# ax[2].plot(freqs, coef, label='velo')
#
# coef, freqs_unscaled=pywt.cwt(mag_input.BzIMF,scale,'gaus1')
# ax[3].plot(freqs, coef, label='BzIMF')

# for ax_i in ax:
#     ax_i.legend()

#ax[-1].set_xlim([datetime(2000, 7, 1), datetime(2000, 7, 31)])

plt.show()
