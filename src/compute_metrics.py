from maginput import MagInput
from w_indices import calculate_G, calculate_W, TSParams
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pickle

from scipy.signal import find_peaks


def preprocess_data(data, ind_end, max_window):
    """function preprocess_data
    processes time window preceding etime to return input metrics for prediction code

    Parameters
    ----------
    data : MagInput
        input data
    ind_end : int
        end index of window
    max_window : int
        number of indices in window

    Returns
    -------
    input, output : ndarray
        arrays of input and output variables for neural net

    """
    means = np.array([])
    vars = np.array([])

    window_size = 1

    while window_size < max_window:

        if ind_end < max_window:
            raise Exception("End Index too small.")

        means = np.append(means, np.mean(data.Kp[ind_end - window_size : ind_end]))
        means = np.append(means, np.mean(data.Dst[ind_end - window_size : ind_end]))
        means = np.append(means, np.mean(data.dens[ind_end - window_size : ind_end]))
        means = np.append(means, np.mean(data.velo[ind_end - window_size : ind_end]))
        means = np.append(means, np.mean(data.Pdyn[ind_end - window_size : ind_end]))
        means = np.append(means, np.mean(data.ByIMF[ind_end - window_size : ind_end]))
        means = np.append(means, np.mean(data.BzIMF[ind_end - window_size : ind_end]))

        vars = np.append(vars, np.var(data.Kp[ind_end - window_size : ind_end]))
        vars = np.append(vars, np.var(data.Dst[ind_end - window_size : ind_end]))
        vars = np.append(vars, np.var(data.dens[ind_end - window_size : ind_end]))
        vars = np.append(vars, np.var(data.velo[ind_end - window_size : ind_end]))
        vars = np.append(vars, np.var(data.Pdyn[ind_end - window_size : ind_end]))
        vars = np.append(vars, np.var(data.ByIMF[ind_end - window_size : ind_end]))
        vars = np.append(vars, np.var(data.BzIMF[ind_end - window_size : ind_end]))

        window_size = int(window_size * 2)

    max_amp = np.max(data.W3[ind_end - int(24 * 60 / 5) : ind_end])
    max_amp_ind = np.argmax(data.W3[ind_end - int(24 * 60 / 5) : ind_end])
    t_since_max = ind_end - max_amp_ind

    peaks, _ = find_peaks(data.W3[ind_end - int(24 * 60 / 5) : ind_end], distance=24)
    try:
        last_peak_amp = data.W3[peaks[-1]]
        ind_since_last_peak = ind_end - peaks[-1]
    except IndexError:
        last_peak_amp = 0.0
        ind_since_last_peak = 1000000000

    input = np.append(means, vars)
    input = np.append(input, [max_amp, max_amp_ind, last_peak_amp, ind_since_last_peak])

    output_window = int(6 * 60 / 5)

    mean6h = np.mean(data.W3[ind_end : ind_end + output_window])
    var6h = np.var(data.W3[ind_end : ind_end + output_window])
    mean24h = np.mean(data.W3[ind_end : ind_end + 4 * output_window])
    var24h = np.var(data.W3[ind_end : ind_end + 4 * output_window])

    output = np.array([mean6h, var6h, mean24h, var24h])

    return input, output


if __name__ == "__main__":

    mag_input, _, _, _ = MagInput.from_QD_file("../data/QDInput_test.dat")

    Ts = 5  # minutes
    hour_window = int(60 / Ts)
    day_window = int(24 * hour_window)
    back_window = int(10 * day_window)

    N = np.size(mag_input.dens)
    N_reduced = int((N - back_window) / hour_window - 1)

    input = None
    output = None
    for i in range(N_reduced):

        next_window = int((i + 1) * hour_window + back_window)
        input_i, output_i = preprocess_data(mag_input, next_window, back_window)

        if input is not None:
            input.concatenate(input_i, axis=1)
            output.concatenate(output_i, axis=1)
        else:
            input = input_i
            output = output_i

    pickle.dump([input, output], open("data_1year_reduced.p", "wb"))
