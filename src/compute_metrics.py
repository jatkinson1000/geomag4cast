from maginput import MagInput
from w_indices import calculate_G, calculate_W, TSParams
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pickle


def preprocess_data(data, ind_end):

    means = np.array([])
    vars  = np.array([])

    window_size = 2
    max_window_size = int(10*24*60/5)


    #ind_end = (np.abs(data.timestamp - tstamp)).argmin()

    while window_size < max_window_size:

        if ind_end < max_window_size:
            raise Exception('End Index too small.')

        means = np.append(means, np.mean(data.Kp[ind_end-window_size:ind_end]))
        means = np.append(means, np.mean(data.Dst[ind_end-window_size:ind_end]))
        means = np.append(means, np.mean(data.dens[ind_end-window_size:ind_end]))
        means = np.append(means, np.mean(data.velo[ind_end-window_size:ind_end]))
        means = np.append(means, np.mean(data.Pdyn[ind_end-window_size:ind_end]))
        means = np.append(means, np.mean(data.ByIMF[ind_end-window_size:ind_end]))
        means = np.append(means, np.mean(data.BzIMF[ind_end-window_size:ind_end]))

        vars = np.append(vars, np.var(data.Kp[ind_end-window_size:ind_end]))
        vars = np.append(vars, np.var(data.Dst[ind_end-window_size:ind_end]))
        vars = np.append(vars, np.var(data.dens[ind_end-window_size:ind_end]))
        vars = np.append(vars, np.var(data.velo[ind_end-window_size:ind_end]))
        vars = np.append(vars, np.var(data.Pdyn[ind_end-window_size:ind_end]))
        vars = np.append(vars, np.var(data.ByIMF[ind_end-window_size:ind_end]))
        vars = np.append(vars, np.var(data.BzIMF[ind_end-window_size:ind_end]))

        window_size = int(window_size * 2)

    output_mean6h = np.mean(data.W3[ind_end:ind_end+int(6*60/5)])
    output_var6h  = np.var(data.W3[ind_end:ind_end+int(6*60/5)])
    output_mean24h = np.mean(data.W3[ind_end:ind_end+int(24*60/5)])
    output_var24h  = np.var(data.W3[ind_end:ind_end+int(24*60/5)])

    return means, vars, output_mean6h, output_var6h, output_mean24h, output_var24h



if __name__ == "__main__":

    mag_input, _, _, _ = MagInput.from_QD_file('../data/QDInput_1year.dat')

    day_window_size = int(24*60/5)

    N = np.size(mag_input.dens)
    N_reduced = int(N / day_window_size) - 10

    for i in range(N_reduced):
        next_window = int((i+1)*day_window_size + 10*day_window_size)
        preprocess_data(mag_input, next_window)
