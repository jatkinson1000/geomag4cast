import itertools
import numpy as np
from dataclasses import dataclass, field
import json
import warnings

@dataclass
class RunParams:
    """ dataclass to read and hold input variables from a
        json file to customise a run of the code.
    """
    # start and end dates
    start_date: str = "2020-01-01 00:00:00"
    end_date: str = "2020-01-01 23:59:59"
    mag_ind_file: str = "test_data/omni_5min_2015_test.lst"
    mag_ind_type: str = "OMNI"
    Kp_file: str = ""
    w_paramfile: str = "./ts05_params_5min.json"
    OutFile: str = "/"

    @classmethod
    def load_json_params(cls, jsonpath):
        params = cls()
        with open(jsonpath, "r") as read_file:
            paramsdict = json.load(read_file)
        params.start_date = paramsdict['start_date']
        params.end_date = paramsdict['end_date']
        params.mag_ind_file = paramsdict['mag_ind_file']
        params.mag_ind_type = paramsdict['mag_ind_type']
        params.Kp_file = paramsdict['Kp_file']
        params.w_paramfile = paramsdict["w_paramfile"]
        params.OutFile = paramsdict['outfile']
        return params


@dataclass
class TSParams:
    """ dataclass to read and hold parameters for
        TS05 parameter calculations from a json file
    """
    dt: float = 5.0  # Time interval of data in minutes
    w_st: list = field(default_factory=lambda: [0.44, 0.42, 0.66, 0.48, 0.49, 0.91])  # Average W1-6 from QD as start
    gamma: list = field(default_factory=lambda: [0.87, 0.67, 1.32, 1.29, 0.69, 0.53])  # W1-6
    beta: list = field(default_factory=lambda: [0.8, 0.18, 2.32, 1.25, 1.6, 2.4])  # W1-6
    lam: list = field(default_factory=lambda: [0.39, 0.46, 0.39, 0.42, 0.41, 1.29])  # W1-6
    r: list = field(default_factory=lambda: [0.39, 0.7, .031, 0.58, 1.15, 0.88])  # W1-6
    f: list = field(default_factory=lambda: [1.049, 1.065, 1.051, 1.061, 1.050, 1.251])  # W1-6
    # corrtime_1m: list = field(default_factory=lambda: [10.59, 7.91, 64.87, 35.95, 29.41])  # By, Bz, V, Den, P
    # corrtime_5m: list = field(default_factory=lambda:  [2.76, 2.17, 14.55,  8.37,  6.94])  # By, Bz, V, Den, P [hr]
    # corrtime_1h: list = field(default_factory=lambda:  [1.46, 0.73, 13.50,  2.18,  1.68])  # By, Bz, V, Den, P [hr]
    corrtime: list = field(default_factory=lambda:  [2.76, 2.17, 14.55,  8.37,  6.94])  # By, Bz, V, Den, P [hr]

    @classmethod
    def load_json_params(cls, jsonpath):
        TSParameters = cls()
        with open(jsonpath, "r") as read_file:
            paramsdict = json.load(read_file)
        TSParameters.dt = paramsdict['dt']
        TSParameters.w_st = paramsdict['w_st']
        TSParameters.gamma = paramsdict['gamma']
        TSParameters.beta = paramsdict['beta']
        TSParameters.lam = paramsdict['lam']
        TSParameters.r = paramsdict['r']
        TSParameters.f = paramsdict['f']
        TSParameters.corrtime = paramsdict['corrtime']
        return TSParameters


def bsgamma(Bz, gamma):
    """ function bsgamma
    Checks magnetic field for neccessary conditions as required by TS04 algorithms.
    Returns southward component raise to power of gamma_i

    Parameters
    ----------
    Bz : ndarray
        1D array of length nt containing Bz values in the timeseries
    gamma : ndarray
        1D array of length 6 containing gamma for each W

    Returns
    -------
    Bs : ndarray
        2D array of size [nt, 6] containing Bs = abs(Bz)**gamma as defined by Tsyganenko, using
        southward component only, and not bad data

    """

    Bs = np.empty([len(Bz), len(gamma)])

    for i, bz in enumerate(Bz):
        if bz == 9999.99:
            Bs[i, :] = 9999.99
        elif bz >= 0.0:
            Bs[i, :] = 0.0
        else:
            Bs[i, :] = abs(bz)**gamma

    return Bs


def get_intervals(arr_dat, bad_val):
    """ function get_intervals
    Locates gaps in data and provides a list of intervals

    Parameters
    ----------
    arr_dat : ndarray
        array of daya in which to locate gaps
    bad_val : float
        value of bad data in arr_dat

    Returns
    -------
    intervals : list
        list of len(2) lists of indices covering intervals of bad data

    """
    arr_gaps = np.nonzero(arr_dat == bad_val)[0]

    # generate intervals from indices
    # - generate a group for each interval based on whenever (value-index) increases
    intervals = []
    for key, group, in itertools.groupby(enumerate(arr_gaps), lambda t: t[1]-t[0]):
        interval_idx = list(group)
        intervals.append([interval_idx[0][1], interval_idx[-1][1]])

    return intervals


def interp_QD(time, data, timestamp, intervals, corrtime):
    """ function interp_QD
    Interpolates across gaps in data according to QD guidelines

    Parameters
    ----------
    time : ndarray
        1D array of size [nt] containing time data
    data : ndarray
        1D array of size [nt] containing variable data
    timestamp : ndarray
        1D array of size [nt] containing time data as a timestamp
    intervals : list
        list of len(2) lists of indices covering intervals of bad data
    corrtime : float
        correlation time in hrs for this variable

    Returns
    -------
    data : ndarray
        1D array of size [nt] containing start and end of data gaps
    qflag : ndarray
        1D array of size [nt] containing quality flag of data 2=measured, 1=correlation, 0=average

    """

    qflag = 2.0*np.ones(len(time))

    for i, interval in enumerate(intervals):
        sint = interval[0]
        eint = interval[1]
        dt = (time[eint+1]-time[sint-1]).total_seconds()/3600.0
        dtau = dt/corrtime
        if dtau <= 2.0:
            # less than 2 tau => linear interpolation
            data[sint:eint+1] = np.interp(timestamp[sint:eint+1],
                                          [timestamp[sint-1], timestamp[eint+1]], [data[sint-1], data[eint+1]])
            qflag[sint:eint+1] = 1.0
        elif dtau > 4.0:
            # more than 4 tau => correlation with average values between
            inttime = np.array([val.total_seconds() / 3600.0 for val in (time[sint:eint + 1] - time[sint - 1])])
            # number of indices in from LHS/RHS/over centre to use correlation times for
            try:
                l_idx = np.where(inttime < corrtime)[0][-1]
            except IndexError:
                # When all internal time indices covered by interval 2*tau then correlate for one index
                l_idx = 0
            try:
                r_idx = len(inttime) - 1 - np.where(inttime > (dt-corrtime))[0][0]
            except IndexError:
                r_idx = 0

            try:
                lm_idx = np.where(inttime < 2.0*corrtime)[0][-1]
            except IndexError:
                lm_idx = 0
            try:
                rm_idx = len(inttime)-1 - np.where(inttime > (dt - 2.0*corrtime))[0][0]
            except IndexError:
                rm_idx = 0

            # Value to set at centre of interval
            midval = (data[sint-1]+data[eint+1])/2.0
            # LHS - correlation
            data[sint:sint + l_idx + 1] = data[sint - 1]
            # RHS - correlation
            data[eint - r_idx:eint + 1] = data[eint + 1]
            # centre
            data[sint+lm_idx+1:eint-rm_idx] = midval
            # LHS Interp
            data[sint+ l_idx + 1:sint+lm_idx+1] = np.interp(timestamp[sint + l_idx + 1:sint + lm_idx + 1],
                                                            [timestamp[sint + l_idx], timestamp[sint + lm_idx + 1]],
                                                            [data[sint + l_idx], data[sint + lm_idx + 1]])
            # RHS Interp
            data[eint-rm_idx:eint - r_idx] = np.interp(timestamp[eint - rm_idx:eint - r_idx],
                                                       [timestamp[eint - rm_idx - 1], timestamp[eint - r_idx]],
                                                       [data[eint - rm_idx - 1], data[eint - r_idx]])
            qflag[sint:sint + lm_idx + 1] = 1.0
            qflag[sint + lm_idx + 1:eint - rm_idx] = 0.0
            qflag[eint - rm_idx:eint + 1] = 1.0

        else:
            # between 2 and 4 tau => correlation with linear interpolation between over interval of 2 tau
            # print('medium: ', dt, corrtime)
            inttime = np.array([val.total_seconds()/3600.0 for val in (time[sint:eint+1] - time[sint-1])])
            # number of indices in from LHS/RHS to use correlation times
            try:
                l_idx = np.where(inttime < (dt-2.0*corrtime)/2.0)[0][-1]
            except IndexError:
                # When all internal time indices covered by interval 2*tau then correlate for one index
                l_idx = 0
            try:
                r_idx = len(inttime)-1 - np.where(inttime > (dt+2.0*corrtime)/2.0)[0][0]
            except IndexError:
                # When all internal time indices covered by interval 2*tau then correlate for one index
                r_idx = 0
            # LHS - correlation
            data[sint:sint+l_idx+1] = data[sint-1]
            # RHS - correlation
            data[eint-r_idx:eint+1] = data[eint+1]
            # Centre - interpolate
            data[sint+l_idx+1:eint-r_idx] = np.interp(timestamp[sint+l_idx+1:eint-r_idx],
                                                      [timestamp[sint+l_idx], timestamp[eint-r_idx]],
                                                      [data[sint-1], data[eint+1]])

    return data, qflag


def interp_gaps(magdata, wparams):
    """ function interp_gaps
    Locates gaps in data and interpolates across according to QD guidelines

    Parameters
    ----------
    magdata : maginput.MagInput
        MagInput class containing SW index data
    wparams : TSParams
        dataclass holding values to be used in the TS/QD algorithms

    Returns
    -------
    None : None
        Changes are made to the maginput.MagInput class which is passed by reference under python.

    """

    # Locate indices for missing data for all relevant variables
    Dst_intervals = get_intervals(magdata.Dst, magdata.bad_data)
    if len(Dst_intervals) > 0:
        warnings.warn(f'There are gaps in the dst index at the following index ranges: {Dst_intervals}')
    dens_intervals = get_intervals(magdata.dens, magdata.bad_data)
    velo_intervals = get_intervals(magdata.velo, magdata.bad_data)
    Pdyn_intervals = get_intervals(magdata.Pdyn, magdata.bad_data)
    ByIMF_intervals = get_intervals(magdata.ByIMF, magdata.bad_data)
    BzIMF_intervals = get_intervals(magdata.BzIMF, magdata.bad_data)

    # Interpolate each variable across the gaps based on correlation times
    magdata.dens, dens_qflag = interp_QD(magdata.time, magdata.dens, magdata.timestamp, dens_intervals, wparams.corrtime[3])
    magdata.velo, velo_qflag = interp_QD(magdata.time, magdata.velo, magdata.timestamp, velo_intervals, wparams.corrtime[2])
    magdata.Pdyn, Pdyn_qflag = interp_QD(magdata.time, magdata.Pdyn, magdata.timestamp, Pdyn_intervals, wparams.corrtime[4])
    magdata.ByIMF, ByIMF_qflag = interp_QD(magdata.time, magdata.ByIMF, magdata.timestamp, ByIMF_intervals, wparams.corrtime[0])
    magdata.BzIMF, BzIMF_qflag = interp_QD(magdata.time, magdata.BzIMF, magdata.timestamp, BzIMF_intervals, wparams.corrtime[1])

    # TODO: Nothing done with qflags at present, ought to write out or store in database.
    #  Could add them as parameters to maginput class, as also provided for QDData and TS files

    return None


def calculate_W(magdata, wparams):
    """ function calculate_W
    Calculates the TS04/05 W input parameters following Tsyganenko and Qin-Denton methodology

    Parameters
    ----------
    magdata : maginput.MagInput
        MagInput class containing SW index data
    wparams : TSParams
        dataclass holding values to be used in the TS/QD algorithms

    Returns
    -------
    None : None
        Changes are made to the maginput.MagInput class which is passed by reference under python.

    """

    # Calculate Bz^gamma for gamma[0:6]
    # Bs = bsgamma(magdata.BzIMF, wparams.gamma)

    # Calculate S values
    s = np.empty([len(magdata.time), 6])
    for i in range(magdata.nt):
        # n in #/cm^3, velo in km/s, B in nT
        # Use only magnitude of southward component of Bz
        s[i, :] = ((magdata.dens[i]/5.0)**np.asarray(wparams.lam)
                   * (magdata.velo[i]/400.0)**np.asarray(wparams.beta)
                   * (abs(min(magdata.BzIMF[i], 0))/5.0)**np.asarray(wparams.gamma))
        # Set time interval between data points in [hr]
        dt = wparams.dt/60.0
        if i == 0:
            # Starting values taken as averages.
            # TODO: When we move to reading from a database in operational mode these lines will
            #  need updating to read last known W into wparams.w_st at start time
            magdata.W1[i] = wparams.r[0]*s[i, 0]*dt + wparams.w_st[0]*np.exp(-wparams.r[0]*dt)
            magdata.W2[i] = wparams.r[1]*s[i, 1]*dt + wparams.w_st[1]*np.exp(-wparams.r[1]*dt)
            magdata.W3[i] = wparams.r[2]*s[i, 2]*dt + wparams.w_st[2]*np.exp(-wparams.r[2]*dt)
            magdata.W4[i] = wparams.r[3]*s[i, 3]*dt + wparams.w_st[3]*np.exp(-wparams.r[3]*dt)
            magdata.W5[i] = wparams.r[4]*s[i, 4]*dt + wparams.w_st[4]*np.exp(-wparams.r[4]*dt)
            magdata.W6[i] = wparams.r[5]*s[i, 5]*dt + wparams.w_st[5]*np.exp(-wparams.r[5]*dt)
        else:
            magdata.W1[i] = wparams.r[0]*s[i, 0]*dt + magdata.W1[i-1]*np.exp(-wparams.r[0]*dt)
            magdata.W2[i] = wparams.r[1]*s[i, 1]*dt + magdata.W2[i-1]*np.exp(-wparams.r[1]*dt)
            magdata.W3[i] = wparams.r[2]*s[i, 2]*dt + magdata.W3[i-1]*np.exp(-wparams.r[2]*dt)
            magdata.W4[i] = wparams.r[3]*s[i, 3]*dt + magdata.W4[i-1]*np.exp(-wparams.r[3]*dt)
            magdata.W5[i] = wparams.r[4]*s[i, 4]*dt + magdata.W5[i-1]*np.exp(-wparams.r[4]*dt)
            magdata.W6[i] = wparams.r[5]*s[i, 5]*dt + magdata.W6[i-1]*np.exp(-wparams.r[5]*dt)

    return None


def calculate_G(magdata):
    """ function calculate_G
    Calculates the TS02/03s G input parameters following Tsyganenko and Qin-Denton methodology

    Parameters
    ----------
    magdata : maginput.MagInput
        MagInput class containing SW index data

    Returns
    -------
    None : None
        Changes are made to the maginput.MagInput class which is passed by reference under python.

    """

    # Calculate Bz^gamma for gamma[0:6]
    # Bs = bsgamma(magdata.BzIMF, wparams.gamma)

    # Calculate S values
    Bs = np.absolute(np.minimum(magdata.BzIMF, 0))
    Bperp40 = np.sqrt(np.power(magdata.ByIMF, 2.0) + np.power(magdata.BzIMF, 2.0)) / 40.0
    hB = np.power(Bperp40, 2.0) / (1.0 + Bperp40)
    theta = np.arctan2(magdata.ByIMF, magdata.BzIMF)
    theta = np.where(theta < 0, theta+2.0*np.pi, theta)
    sin3theta = np.power(np.sin(theta/2.0), 3.0)

    for i in range(magdata.nt):
        if i < 11:
            # Starting values taken as averages.
            # TODO: When database is active methods need updating to read previous data in at start time
            magdata.G1[i] = (1.0/(i+1)) * np.sum(magdata.velo[:i+1]*hB[:i+1]*sin3theta[:i+1])
            magdata.G2[i] = (0.005/(i+1)) * np.sum(magdata.velo[:i+1]*Bs[:i+1])
            magdata.G3[i] = (0.0005/(i+1)) * np.sum(magdata.dens[:i+1]*magdata.velo[:i+1]*Bs[:i+1])
        else:
            magdata.G1[i] = (1.0/12.0) * np.sum(magdata.velo[i-11:i+1]*hB[i-11:i+1]*sin3theta[i-11:i+1])
            magdata.G2[i] = (0.005/12.0) * np.sum(magdata.velo[i-11:i+1]*Bs[i-11:i+1])
            magdata.G3[i] = (0.0005/12.0) * np.sum(magdata.dens[i-11:i+1]*magdata.velo[i-11:i+1]*Bs[i-11:i+1])

    return None
