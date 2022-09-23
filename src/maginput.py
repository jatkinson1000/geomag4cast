import warnings

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys

# import ***INSERT API HERE IF YOU HAVE ACCESS*** as swdata_api


def dt64_2_ts(dt64):
    """
    Routine to convert an array of np.datetime64 to timestamp in s

    Parameters
    ----------
    dt64: ndarray
        np.ndarray of dtype np.datetime64

    Returns
    ----------
    list
        list of timestamps corresponding to input datetime64 times

    """
    return (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')


def datetime_range(start, end, delta):
    """
    Routine to generate an array of datetimes

    Parameters
    ----------
    start, end: datetime
        Start and end times for data
    delta: timedelta
        intervals to retrieve data at

    Returns
    ----------
    yield statement as generator
    """
    current = start
    while current <= end:
        yield current
        current += delta


@dataclass
class MagInput:
    """dataclass for magnetic data required as IRBEM inputs
    """
    nt: int = None
    time: np.ndarray = None  # array of datetimes corresponding to each index
    timestamp: np.ndarray = None  # timestamp corresponding to self.time (useful in various applications)
    Kp: np.ndarray = None  # Kp*10 as per OMNI2, double, 0-90
    Dst: np.ndarray = None  # [nT]
    dens: np.ndarray = None  # Solar wind density [cm-3]
    velo: np.ndarray = None  # Solar wind velocity [km s-1]
    Pdyn: np.ndarray = None  # Solar wind dynamic pressure [nPa]
    ByIMF: np.ndarray = None  # GSM ya component of IMF mag. field [nT]
    BzIMF: np.ndarray = None  # GSM z component of IMF mag. field [nT]
    G1: np.ndarray = None  # G1 Index for TS04/05 field. See online documentation.
    G2: np.ndarray = None  # G2 Index for TS04/05 field. See online documentation.
    G3: np.ndarray = None  # G3 Index for TS04/05 field. See online documentation.
    W1: np.ndarray = None  # W2 Index for TS04/05 field. See online documentation.
    W2: np.ndarray = None  # W3 Index for TS04/05 field. See online documentation.
    W3: np.ndarray = None  # W4 Index for TS04/05 field. See online documentation.
    W4: np.ndarray = None  # W5 Index for TS04/05 field. See online documentation.
    W5: np.ndarray = None  # W6 Index for TS04/05 field. See online documentation.
    W6: np.ndarray = None  # W7 Index for TS04/05 field. See online documentation.
    AL: np.ndarray = None  # Auroral Index
    bad_data: float = -9999.0  # Bad data value for IRBEM

    def from_db(self, sdate, edate):
        """
        Reads in data from UM Met Office SW database via API

        Parameters
        ----------
        sdate, edate : datetime.datetime
            datetime for the start and end date for which to pull data

        Returns
        -------
        self : MagInput Class
            Instance of MagInput populated using OMNI HighRes data file data

        References
        -------
        Private documentation for UK Met Office API developed as part of SWIMMR Project
        """

        # TS04 NEEDS:
            # Dst
            # dens
            # velo
            # Pdyn
            # ByIMF
            # BzIMF

        # We can also get
            # Kp

        # Add extra day to start and end to ensure we are covered for period we want.
        date1 = sdate.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        date2 = edate.replace(hour=23, minute=59, second=59, microsecond=999999) + timedelta(days=1)

        # Use API to pull data from database
        # Keep times as np.datetime64 until we extract our final timeseries at the end

        # Dst - Pull data from database, extract, and generate as series keeping only 1 record
        Dst_dc = swdata_api.fetch_dst(timestamp_from=date1, timestamp_to=date2)
        # TODO: There is a dodgy workaround here due to the fact that two values of Dst are returned with
        #  1min separation for each hour. Need UKMO to fix their API.
        # dt_Dst = Dst_dc['timestamp'].astype('datetime64[s]')
        # Dst = Dst_dc['dst'].astype(float)
        dt_Dst = []
        Dst = []
        for i in Dst_dc:
            if not dt_Dst or i['timestamp'].astype('datetime64[s]') > dt_Dst[-1]+np.timedelta64(1, 'm'):
                dt_Dst.append(i['timestamp'].astype('datetime64[s]'))
                Dst.append(i['dst'])
        dt_Dst = np.asarray(dt_Dst, dtype='datetime64[s]')
        Dst = np.asarray(Dst)

        # Kp - Pull data from database, extract, and generate as series with accompanying 3hr datetimes
        Kp_dc = swdata_api.fetch_kp_index_potsdam(timestamp_from=date1, timestamp_to=date2)
        dt_Kp = []
        Kp = []
        try:
            for ti, ts in enumerate(Kp_dc):
                for i in range(8):
                    # dt_Kp.append(datetime.utcfromtimestamp((ts['timestamp'] - np.datetime64('1970-01-01T00:00:00'))
                    #                                        / np.timedelta64(1, 's')) + i*timedelta(minutes=180))
                    dt_Kp.append(ts['timestamp'].astype('datetime64[s]') + i*np.timedelta64(3, 'h'))
                    Kp.append(ts[i+1])
        except TypeError as e:
            print(f'Issues with Kp data processing. Error report:\n\t{e}')
        dt_Kp = np.asarray(dt_Kp, dtype='datetime64[s]')
        Kp = np.asarray(Kp, dtype=float)

        # L1 solar wind data - Pull data from database, extract, and generate as series for each var
        # B
        B_dc = swdata_api.fetch_rtsw_magnetometer(timestamp_from=date1, timestamp_to=date2)
        dt_B = B_dc['timestamp'].astype('datetime64[s]')
        Bt = B_dc['bt'].astype(float)
        Bx_GSM = B_dc['bx_gsm'].astype(float)
        By_GSM = B_dc['by_gsm'].astype(float)
        Bz_GSM = B_dc['bz_gsm'].astype(float)
        # V
        V_dc = swdata_api.fetch_rtsw_wind(timestamp_from=date1, timestamp_to=date2)
        dt_V = V_dc['timestamp'].astype('datetime64[s]')
        Vt = V_dc['proton_speed'].astype(float)
        Vx_GSM = V_dc['proton_vx_gsm'].astype(float)
        Vy_GSM = V_dc['proton_vy_gsm'].astype(float)
        Vz_GSM = V_dc['proton_vz_gsm'].astype(float)
        dens = V_dc['proton_density'].astype(float)
        P_dyn = 1.6726 * 1.0E-6 * dens * np.power(Vt, 2.0)
        # Position data for L1 satellite - Provided hourly (sort of... hr and hr+1s)
        L1_pos_dc = swdata_api.fetch_rtsw_ephemerides(timestamp_from=date1, timestamp_to=date2)
        # TODO: Do we need a check to make sure that B/V data and position data are from same source (DSCOVR/ACE etc)?
        dt_L1 = []
        L1_x_GSE = []
        L1_y_GSE = []
        for i, posi in enumerate(L1_pos_dc):
            if not dt_L1 or posi['timestamp'] > dt_L1[-1]:
                dt_L1.append(posi['timestamp'])
                L1_x_GSE.append(posi['x_gse'])
                L1_y_GSE.append(posi['y_gse'])

        # Deal with the fact that B, V, and Pos timestamps do not match - Interpolate position points to B/V timeseries
        # Following unneccessary as result is dt_B
        L1_x_GSE_B = np.interp(dt64_2_ts(dt_B), dt64_2_ts(dt_L1), L1_x_GSE)
        L1_y_GSE_B = np.interp(dt64_2_ts(dt_B), dt64_2_ts(dt_L1), L1_y_GSE)
        L1_x_GSE_V = np.interp(dt64_2_ts(dt_V), dt64_2_ts(dt_L1), L1_x_GSE)
        L1_y_GSE_V = np.interp(dt64_2_ts(dt_V), dt64_2_ts(dt_L1), L1_y_GSE)

        # Propogate B, Bi, V, Vi, n forward to dayside of magnetosphere
        #  by adding dt factor to the ACE time series (Or DSCOVR if swapped?)
        # https://omniweb.gsfc.nasa.gov/html/ow_data.html - Section 12
        V_E = 30  # Orbital velocity of Earth in [km/s]
        # B
        # TODO: Use Vt or Vx, as OMNI equation states 'solar wind speed, assumed radial'?
        Vt_B = np.interp(dt64_2_ts(dt_B), dt64_2_ts(dt_V), Vt)
        W_B = np.tan(0.5*np.arctan(Vt_B/428.0))  # Angle between convection and co-rotation
        delta_t_B = (L1_x_GSE_B + L1_y_GSE_B*W_B)/(Vt_B + V_E*W_B)
        dt_B = dt_B + np.around(delta_t_B).astype('timedelta64[s]')
        # V
        W_V = np.tan(0.5*np.arctan(Vt/428.0))  # Angle between convection and co-rotation
        delta_t_V = (L1_x_GSE_V + L1_y_GSE_V*W_V)/(Vt + V_E*W_V)
        dt_V = dt_V + np.around(delta_t_V).astype('timedelta64[s]')

        # Define timestamps for final data
        out_start = sdate - (sdate - datetime.min) % timedelta(minutes=5)
        out_end = edate + (datetime.min - edate) % timedelta(minutes=5)
        sdate64 = np.datetime64(out_start, 's')
        edate64 = np.datetime64(out_end, 's')

        dt_out = np.arange(sdate64, edate64 + np.timedelta64(5*60, 's'),
                           np.timedelta64(5*60, 's'), dtype='datetime64[s]')

        # Map/average data to required timestamps for output
        # Map Dst from hourly to finer measurement using interpolation
        Dst_out = np.interp(dt64_2_ts(dt_out), dt64_2_ts(dt_Dst), Dst)

        # Map Kp using propagation, not interpolation
        kp_out = []
        # for ti, t in enumerate(dt_out):
        idx = np.searchsorted(dt_Kp, dt_out, side='right')
        Kp_out = Kp[idx-1]

        # Map V, B, n, Pdyn
        # Data from shorter intervals than 5min output - interpolate or perform averaging?
        # Do basic interpolation for now.
        By_out = np.interp(dt64_2_ts(dt_out), dt64_2_ts(dt_B), By_GSM)
        Bz_out = np.interp(dt64_2_ts(dt_out), dt64_2_ts(dt_B), Bz_GSM)
        V_out = np.interp(dt64_2_ts(dt_out), dt64_2_ts(dt_V), Vt)
        dens_out = np.interp(dt64_2_ts(dt_out), dt64_2_ts(dt_V), dens)
        P_dyn_out = np.interp(dt64_2_ts(dt_out), dt64_2_ts(dt_V), P_dyn)

        # Convert to timestamp for interpolation use later
        timestamp_out = dt64_2_ts(dt_out)

        nt = len(dt_out)
        self.nt = nt
        self.time = dt_out
        self.timestamp = np.array(timestamp_out)
        self.Kp = Kp_out
        self.Dst = Dst_out
        self.dens = dens_out
        self.velo = V_out
        self.Pdyn = P_dyn_out
        self.ByIMF = By_out
        self.BzIMF = Bz_out
        self.G1 = self.bad_data*np.ones(nt)
        self.G2 = self.bad_data*np.ones(nt)
        self.G3 = self.bad_data*np.ones(nt)
        self.W1 = self.bad_data*np.ones(nt)
        self.W2 = self.bad_data*np.ones(nt)
        self.W3 = self.bad_data*np.ones(nt)
        self.W4 = self.bad_data*np.ones(nt)
        self.W5 = self.bad_data*np.ones(nt)
        self.W6 = self.bad_data*np.ones(nt)
        self.AL = self.bad_data*np.ones(nt)

        return self

    def from_OMNI_HR_file(self, OMNI_HR_file, reset_bad=False):
        """
        Reads in data from OMNI High Resolution (1 or 5 min averaged) data file taken from OMNI website (see reference)
        or prepared by user to populate class with relevant variables.

        NB: This uses SYM-H index from files as a proxy for DST as an input.  DST is provided hourly, whilst
            SYM-H is on 1min intervals.

        Parameters
        ----------
        OMNI_HR_file : string
            Path to relevant OMNI file in appropriate format covering relevant dates
            Assumes data appears in headerless, whitespace-delineated text file in the following format:
                Year, DoY, Hr, Min, IMF Mag, Bx, By, Bz, velo, dens, Pdyn, AL, SYM/H
                Note that B is required in GSM components
        reset_bad : bool
            if true then all bad data values will be reset from their omni defaults to that of self.bad_data

        Returns
        -------
        self : MagInput Class
            Instance of MagInput populated using OMNI HighRes data file data

        References
        -------
        https://omniweb.gsfc.nasa.gov/form/omni_min.html web interface for downloading high resolution OMNI data
        """

        # Read in OMNI data
        OMNI_data = np.loadtxt(OMNI_HR_file)

        year_OMNI = np.array(OMNI_data[:, 0], dtype=int)
        nt = len(year_OMNI)
        DoY_OMNI = np.array(OMNI_data[:, 1], dtype=int)
        hour_OMNI = np.array(OMNI_data[:, 2], dtype=int)
        minute_OMNI = np.array(OMNI_data[:, 3], dtype=int)
        # Bmag = np.array(OMNI_data[:, 4], dtype=float)
        # Bx = np.array(OMNI_data[:, 5], dtype=float)
        ByIMF = np.array(OMNI_data[:, 6], dtype=float)
        BzIMF = np.array(OMNI_data[:, 7], dtype=float)
        velo = np.array(OMNI_data[:, 8], dtype=float)
        dens_pro = np.array(OMNI_data[:, 9], dtype=float)
        Pdyn = np.array(OMNI_data[:, 10], dtype=float)
        AL = np.array(OMNI_data[:, 11], dtype=int)
        SYMH = np.array(OMNI_data[:, 12], dtype=int)

        if reset_bad:
            # Bmag = np.where(Bmag == 9999.99, self.bad_data, Bmag)
            # Bx = np.where(Bx == 9999.99, self.bad_data, Bx)
            ByIMF = np.where(ByIMF == 9999.99, self.bad_data, ByIMF)
            BzIMF = np.where(BzIMF == 9999.99, self.bad_data, BzIMF)
            velo = np.where(velo == 99999.9, self.bad_data, velo)
            dens_pro = np.where(dens_pro == 999.99, self.bad_data, dens_pro)
            Pdyn = np.where(Pdyn == 99.99, self.bad_data, Pdyn)
            AL = np.where(AL == 99999, self.bad_data, AL)
            SYMH = np.where(SYMH == 99999, self.bad_data, SYMH)

        datetime_OMNI = []
        for i in range(nt):
            datetime_OMNI.append(datetime(year_OMNI[i], 1, 1)
                                 + timedelta(days=int(DoY_OMNI[i]-1),
                                             hours=int(hour_OMNI[i]),
                                             minutes=int(minute_OMNI[i])))
        # Convert to timestamp for interpolation use later
        timestamp_OMNI = np.array([datetime.timestamp(t) for t in datetime_OMNI])

        self.nt = nt
        self.time = np.array(datetime_OMNI)
        self.timestamp = timestamp_OMNI
        self.Kp = self.bad_data*np.ones(nt)
        self.Dst = SYMH
        self.dens = dens_pro
        self.velo = velo
        self.Pdyn = Pdyn
        self.ByIMF = ByIMF
        self.BzIMF = BzIMF
        self.G1 = self.bad_data*np.ones(nt)
        self.G2 = self.bad_data*np.ones(nt)
        self.G3 = self.bad_data*np.ones(nt)
        self.W1 = self.bad_data*np.ones(nt)
        self.W2 = self.bad_data*np.ones(nt)
        self.W3 = self.bad_data*np.ones(nt)
        self.W4 = self.bad_data*np.ones(nt)
        self.W5 = self.bad_data*np.ones(nt)
        self.W6 = self.bad_data*np.ones(nt)
        self.AL = AL

        return self

    def from_TS0405_file(self, TS05_file):
        """
        Reads in data from TS05 data file taken from Tsyganenko's website (see reference) or prepared by user
        to populate class with relevant variables.

        NB: This uses SYM-H index from files as a proxy for DST as an input.  DST is provided hourly, whilst
            SYM-H is on 1min intervals.

        Parameters
        ----------
        TS05_file : string
            Path to relevant yearly TS05 file downloaded from Tsyganenko's website or produced from source code

        Returns
        -------
        self : MagInput Class
            Instance of MagInput populated using TS04/05 data file data
        data_flag :  numpy array
            Array containing data quality flags for IMF and ISW at each time point

        References
        -------
        http://geo.phys.spbu.ru/~tsyganenko/TS05_data_and_stuff/ Contains information on preparing TS05 input files
        and a database of downloadable yearly files.
        Note - TS05 files are provided in yearly intervals.  If you need an interval spanning multiple years
        you will need to combine these into a single file or pre-process yourself.
        """

        # Read in TS data (all data in case user wants to use flags etc in future)
        try:
            TS05_data = np.loadtxt(TS05_file)
        except NameError as error:
            print(error)
            print('Your requested Tsyganenko input file appears not to exist.\nPlease check and try again.')
            sys.exit(1)

        year_TS = np.array(TS05_data[:, 0], dtype=int)
        nt = len(year_TS)
        DoY_TS = np.array(TS05_data[:, 1], dtype=int)
        hr_TS = np.array(TS05_data[:, 2], dtype=int)
        min_TS = np.array(TS05_data[:, 3], dtype=int)
        BxIMF = np.array(TS05_data[:, 4], dtype=float)
        ByIMF = np.array(TS05_data[:, 5], dtype=float)
        BzIMF = np.array(TS05_data[:, 6], dtype=float)
        Vx = np.array(TS05_data[:, 7], dtype=float)
        Vy = np.array(TS05_data[:, 8], dtype=float)
        Vz = np.array(TS05_data[:, 9], dtype=float)
        dens_pro = np.array(TS05_data[:, 10], dtype=float)
        temp = np.array(TS05_data[:, 11], dtype=float)
        SYMH = np.array(TS05_data[:, 12], dtype=float)
        IMFFLAG = np.array(TS05_data[:, 13], dtype=int)
        ISWFLAG = np.array(TS05_data[:, 14], dtype=int)
        Tilt = np.array(TS05_data[:, 15], dtype=float)
        Pdyn = np.array(TS05_data[:, 16], dtype=float)
        W1 = np.array(TS05_data[:, 17], dtype=float)
        W2 = np.array(TS05_data[:, 18], dtype=float)
        W3 = np.array(TS05_data[:, 19], dtype=float)
        W4 = np.array(TS05_data[:, 20], dtype=float)
        W5 = np.array(TS05_data[:, 21], dtype=float)
        W6 = np.array(TS05_data[:, 22], dtype=float)

        datetime_TS = []
        for i in range(nt):
            # TODO: Need to force data to be int in following line, despite already specifying above.
            #  Not sure why, examine in depth at a later point.
            datetime_TS.append(datetime(year_TS[i], 1, 1)
                               + timedelta(days=int(DoY_TS[i]-1),
                                           hours=int(hr_TS[i]),
                                           minutes=int(min_TS[i])))
        # Convert to timestamp for interpolation use later
        timestamp_TS = np.array([datetime.timestamp(t) for t in datetime_TS])

        self.nt = nt
        self.time = np.array(datetime_TS)
        self.timestamp = timestamp_TS
        self.Kp = self.bad_data*np.ones(nt)
        self.Dst = SYMH
        self.dens = dens_pro
        self.velo = self.bad_data*np.ones(nt)
        self.Pdyn = Pdyn
        self.ByIMF = ByIMF
        self.BzIMF = BzIMF
        self.G1 = self.bad_data*np.ones(nt)
        self.G2 = self.bad_data*np.ones(nt)
        self.G3 = self.bad_data*np.ones(nt)
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.W4 = W4
        self.W5 = W5
        self.W6 = W6
        self.AL = self.bad_data*np.ones(nt)

        data_flag = np.column_stack((IMFFLAG, ISWFLAG))

        return self, data_flag

    @classmethod
    def from_QD_file(cls, QD_file):
        """
        Reads in data from Qin-Denton data file taken from the RBSP data portal (see reference) or prepared by user
        to populate class with relevant variables.

        NB: This uses SYM-H index from files as a proxy for DST as an input.  DST is provided hourly, whilst
            SYM-H is on 1min intervals.

        Parameters
        ----------
        QD_file : string
            Path to relevant yearly QD file downloaded from rbsp database or produced from source code

        Returns
        -------
        self : MagInput Class
            Instance of MagInput populated using QD data file data
        SW_flag, G_flag, W_flag : numpy array
            Numpy arrays containing quality flags for SW indices, G parameters, and W parameters at each time
            2 is good direct from parameters,
            1 is interpolated from nearby parameters,
            0 is based on 20/long term average
            G based on quality of inputs, W based on a weighted average of quality of inputs

        References
        -------
        https://rdenton.host.dartmouth.edu/swpar.pdf
        Description of the Qin-Denton interpolation procedure for SW indices and G/W field model parameters

        https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/
        Database of downloadable Qin-Denton files.
        Note - You may need to combine data into a single file or pre-process yourself if modelling an interval that crosses
        multiple file date ranges.

        NB there are 1hr, 5min, and 1min data files available.
        It would appear that certain indices are linearly interpolated to higher resolution, e.g. Kp
        Need to be aware of this in application.
        """

        # Read in QD data
        try:
            QD_data = np.genfromtxt(QD_file, comments="#", usecols=np.arange(1, 44))
        except NameError as error:
            print(error)
            print('Your requested Qin-Denton input file appears not to exist.\nPlease check and try again.')
            sys.exit(1)

        year_QD = np.array(QD_data[:, 0], dtype=int)
        nt = len(year_QD)
        month_QD = np.array(QD_data[:, 1], dtype=int)
        day_of_month_QD = np.array(QD_data[:, 2], dtype=int)
        hr_QD = np.array(QD_data[:, 3], dtype=int)
        min_QD = np.array(QD_data[:, 4], dtype=int)
        sec_QD = np.array(QD_data[:, 5], dtype=int)
        ByIMF = np.array(QD_data[:, 6], dtype=float)
        BzIMF = np.array(QD_data[:, 7], dtype=float)
        velo = np.array(QD_data[:, 8], dtype=float)
        dens_pro = np.array(QD_data[:, 9], dtype=float)
        Pdyn = np.array(QD_data[:, 10], dtype=float)

        G1 = np.array(QD_data[:, 11], dtype=float)
        G2 = np.array(QD_data[:, 12], dtype=float)
        G3 = np.array(QD_data[:, 13], dtype=float)

        ByIMF_flag = np.array(QD_data[:, 14], dtype=int)
        BzIMF_flag = np.array(QD_data[:, 15], dtype=int)
        velo_flag = np.array(QD_data[:, 16], dtype=int)
        dens_pro_flag = np.array(QD_data[:, 17], dtype=int)
        Pdyn_flag = np.array(QD_data[:, 18], dtype=int)

        G1_flag = np.array(QD_data[:, 19], dtype=int)
        G2_flag = np.array(QD_data[:, 20], dtype=int)
        G3_flag = np.array(QD_data[:, 21], dtype=int)

        KP = np.array(QD_data[:, 22], dtype=float)
        AKP3 = np.array(QD_data[:, 23], dtype=float)
        DST = np.array(QD_data[:, 24], dtype=int)
        # Bz = np.array(QD_data[:, 25], dtype=float)

        W1 = np.array(QD_data[:, 31], dtype=float)
        W2 = np.array(QD_data[:, 32], dtype=float)
        W3 = np.array(QD_data[:, 33], dtype=float)
        W4 = np.array(QD_data[:, 34], dtype=float)
        W5 = np.array(QD_data[:, 35], dtype=float)
        W6 = np.array(QD_data[:, 36], dtype=float)

        W1_flag = np.array(QD_data[:, 37], dtype=int)
        W2_flag = np.array(QD_data[:, 38], dtype=int)
        W3_flag = np.array(QD_data[:, 39], dtype=int)
        W4_flag = np.array(QD_data[:, 40], dtype=int)
        W5_flag = np.array(QD_data[:, 41], dtype=int)
        W6_flag = np.array(QD_data[:, 42], dtype=int)

        datetime_QD = []
        for i in range(nt):
            datetime_QD.append(datetime(year_QD[i], month_QD[i], day_of_month_QD[i],
                                        hr_QD[i], min_QD[i], sec_QD[i]))
        # Convert to timestamp for interpolation use later
        timestamp_QD = np.array([datetime.timestamp(t) for t in datetime_QD])

        dat_ret = cls()
        dat_ret.time = np.array(datetime_QD)
        dat_ret.nt = nt
        dat_ret.timestamp = timestamp_QD
        dat_ret.Kp = KP
        dat_ret.Dst = DST
        dat_ret.dens = dens_pro
        dat_ret.velo = velo
        dat_ret.Pdyn = Pdyn
        dat_ret.ByIMF = ByIMF
        dat_ret.BzIMF = BzIMF
        dat_ret.G1 = G1
        dat_ret.G2 = G2
        dat_ret.G3 = G3
        dat_ret.W1 = W1
        dat_ret.W2 = W2
        dat_ret.W3 = W3
        dat_ret.W4 = W4
        dat_ret.W5 = W5
        dat_ret.W6 = W6
        dat_ret.AL = dat_ret.bad_data*np.ones(nt)

        SW_flag = np.column_stack((ByIMF_flag, BzIMF_flag, velo_flag, dens_pro_flag, Pdyn_flag))
        G_flag = np.column_stack((G1_flag, G2_flag, G3_flag))
        W_flag = np.column_stack((W1_flag, W2_flag, W3_flag, W4_flag, W5_flag, W6_flag))

        return dat_ret, SW_flag, G_flag, W_flag

    def prepare_maginput(self, timeseries):
        """
        Use the data contained in an instance of mag_data to generate input in the
        format required using IRBEM.
        input timeseries is a list of datetimes at which the input is required.
        Data already read in will be interpolated to these values.

        Parameters
        ----------
        timeseries : array_like
            list or array of datetimes that data contained within self will be interpolated to

        Returns
        -------
        mag_in : dict
            Dictionary of the format required by IRBEM.py as an input to MagFields class

        References
        -------
        """

        ts_timestamp = [datetime.timestamp(t) for t in timeseries]

        if ts_timestamp[0] < self.timestamp[0]:
            raise ValueError('The first time in the input data lies outside the MagInput times read from file.')
        if ts_timestamp[-1] > self.timestamp[-1]:
            raise ValueError('The final time in the input data lies outside the MagInput times read from file.')

        # Code to return as a dictionary as required by IRBEM.py
        mag_in = {
            "Kp": np.interp(ts_timestamp, self.timestamp, self.Kp),  # Kp
            "Dst": np.interp(ts_timestamp, self.timestamp, self.Dst),  # Dst
            "dens": np.interp(ts_timestamp, self.timestamp, self.dens),  # Dens
            "velo": np.interp(ts_timestamp, self.timestamp, self.velo),  # Velo
            "Pdyn": np.interp(ts_timestamp, self.timestamp, self.Pdyn),  # Pdyn
            "ByIMF": np.interp(ts_timestamp, self.timestamp, self.ByIMF),  # ByIMF
            "BzIMF": np.interp(ts_timestamp, self.timestamp, self.BzIMF),  # BzIMF
            "G1": np.interp(ts_timestamp, self.timestamp, self.G1),  # G1
            "G2": np.interp(ts_timestamp, self.timestamp, self.G2),  # G2
            "G3": np.interp(ts_timestamp, self.timestamp, self.G3),  # G3
            "W1": np.interp(ts_timestamp, self.timestamp, self.W1),  # W1
            "W2": np.interp(ts_timestamp, self.timestamp, self.W2),  # W2
            "W3": np.interp(ts_timestamp, self.timestamp, self.W3),  # W3
            "W4": np.interp(ts_timestamp, self.timestamp, self.W4),  # W4
            "W5": np.interp(ts_timestamp, self.timestamp, self.W5),  # W5
            "W6": np.interp(ts_timestamp, self.timestamp, self.W6),  # W6
            "AL": np.interp(ts_timestamp, self.timestamp, self.AL),  # AL
        }

        return mag_in


def get_maginput_i(magin_arr_dict, i):
    """
    Extract a single element from a maginput dictionary composed of arrays

    !!!
    NB: Later found that this can be replaced by a single line operating on a dictionay which is mpre pythonic:
        maginput_i = {k: v[i] for k, v in maginput.items()}
    Use this in future and endeavour to replace in older code
    !!!

    Parameters
    ----------
    magin_arr_dict : dict
        dictionary of magin data prepared for input to IRBEM where each value is an array
    i : int
        integer to extract from arrays

    Returns
    -------
    mag_in_i : dict
        Dictionary of the format required by IRBEM.py as an input to MagFields class for element i

    References
    -------
    """

    # Code to return ith element from each as a dictionary as required by IRBEM.py

    warnings.warn("This function is deprecated."
                  "Consider using 'maginput_i = {k: v[i] for k, v in maginput.items()}' intead.",
                  DeprecationWarning)

    mag_in_i = {
        "Kp": magin_arr_dict["Kp"][i],  # Kp
        "Dst": magin_arr_dict["Dst"][i],  # Dst
        "dens": magin_arr_dict["dens"][i],  # Dens
        "velo": magin_arr_dict["velo"][i],  # Velo
        "Pdyn": magin_arr_dict["Pdyn"][i],  # Pdyn
        "ByIMF": magin_arr_dict["ByIMF"][i],  # ByIMF
        "BzIMF": magin_arr_dict["BzIMF"][i],  # BzIMF
        "G1": magin_arr_dict["G1"][i],  # G1
        "G2": magin_arr_dict["G2"][i],  # G2
        "G3": magin_arr_dict["G3"][i],  # G3
        "W1": magin_arr_dict["W1"][i],  # W1
        "W2": magin_arr_dict["W2"][i],  # W2
        "W3": magin_arr_dict["W3"][i],  # W3
        "W4": magin_arr_dict["W4"][i],  # W4
        "W5": magin_arr_dict["W5"][i],  # W5
        "W6": magin_arr_dict["W6"][i],  # W6
        "AL": magin_arr_dict["AL"][i],  # AL
    }

    return mag_in_i
