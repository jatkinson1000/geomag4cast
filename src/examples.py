from maginput import MagInput
from w_indices import calculate_G, calculate_W, TSParams
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pathlib import Path

from scipy.signal import find_peaks

p2f = Path(__file__)


if __name__ == "__main__":
    mag_input, _, _, _ = MagInput.from_QD_file(str(p2f.parent.parent.joinpath('data/QDInput_5year.dat')))

    # calculate_G(mag_input)
    # calculate_W(mag_input, TSParams.load_json_params('ts05_params_5min.json'))

    fig, ax = plt.subplots(4, 1, sharex='col')

    # ax[0].plot(mag_input.time, mag_input.G1, label='G1')
    # ax[0].plot(mag_input.time, mag_input.G2, label='G2')
    # ax[0].plot(mag_input.time, mag_input.G3, label='G3')
    # ax[0].set_ylim([0, 75])

    ax[0].plot(mag_input.time, mag_input.W1, label='W1')
    ax[0].plot(mag_input.time, mag_input.W2, label='W2')
    ax[0].plot(mag_input.time, mag_input.W3, label='W3')
    ax[0].plot(mag_input.time, mag_input.W4, label='W4')
    ax[0].plot(mag_input.time, mag_input.W5, label='W5')
    ax[0].plot(mag_input.time, mag_input.W6, label='W6')
    ax[0].set_ylim([0, 10])

    ax[1].plot(mag_input.time, mag_input.dens, label='dens')

    ax[2].plot(mag_input.time, mag_input.velo, label='velo')

    ax[3].plot(mag_input.time, abs(np.minimum(mag_input.BzIMF, 0.0)), label='BzIMF')

    for ax_i in ax:
        ax_i.legend()

    # ax[-1].set_xlim([datetime(2000, 7, 1), datetime(2000, 7, 31)])

    plt.show()
