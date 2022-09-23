import warnings

import numpy as np
from datetime import datetime, timedelta

import w_indices as w_indices
import maginput as maginput


if __name__ == "__main__":
    # Load default parameters including sdate, edate, and filepaths to indices
    runparams = w_indices.RunParams.load_json_params('default_params.json')
    sdate = datetime.strptime(runparams.start_date, "%Y-%m-%d %H:%M:%S")
    edate = datetime.strptime(runparams.end_date, "%Y-%m-%d %H:%M:%S")

    # Generate a MagInput object to store data in
    mag_input = maginput.MagInput()

    Wparams = w_indices.TSParams.load_json_params(runparams.w_paramfile)

    # Extract the data from the file into the MagInput structure
    if runparams.mag_ind_type in ("QD", "QinDenton", "Qin-Denton"):
        # From Qin-Denton file from virbo or rbsp database
        print("Reading W-parameters from Qin-Denton data...", end="")
        mag_input, _, _, _ = mag_input.from_QD_file(runparams.mag_ind_file)
        print("success.")

    elif runparams.mag_ind_type in ("TS", "Tsyganenko", "TS04", "TS05", "T04", "T05"):
        # From Tsyganenko file from personal website
        print("Reading W-parameters from Tsyganenko data...", end="")
        mag_input, _ = mag_input.from_TS0405_file(runparams.mag_ind_file)
        print("success.")

    elif runparams.mag_ind_type in ("OMNI", "OMNI_HR", "OMNI_5m", "OMNIHR", "OMNI5m"):
        # Calculate W and G from OMNI data. Also collects Kp data if desired.
        print("Generating W-parameters from OMNI data...", end="")
        mag_input = mag_input.from_OMNI_HR_file(runparams.mag_ind_file)
        w_indices.interp_gaps(mag_input, Wparams)
        w_indices.calculate_W(mag_input, Wparams)

        # Read Kp Data in from file
        try:
            OMNI_data = np.loadtxt(runparams.Kp_file)
            year_OMNI = np.array(OMNI_data[:, 0], dtype=int)
            nt = len(year_OMNI)
            DoY_OMNI = np.array(OMNI_data[:, 1], dtype=int)
            hour_OMNI = np.array(OMNI_data[:, 2], dtype=int)
            Kp = np.array(OMNI_data[:, 3], dtype=int)
            datetime_OMNI = []
            for i in range(nt):
                datetime_OMNI.append(
                    datetime(year_OMNI[i], 1, 1)
                    + timedelta(days=int(DoY_OMNI[i] - 1), hours=int(hour_OMNI[i]))
                )
            timestamp_OMNI = np.array([datetime.timestamp(t) for t in datetime_OMNI])
            mag_input.Kp = Kp[np.searchsorted(timestamp_OMNI, mag_input.timestamp)]
        except OSError as e:
            warnings.warn("Kp data not read.\nIf you were expecting Kp data please check the filename provided.",
                          stacklevel=2)

        print("success.")

    elif runparams.mag_ind_type in ("swdb", "SWDB"):
        # TODO Read from BAS SARIF/swdb API
        #  Method should be same as ukmo below but with the UKMO API swapped out for equivalent BAS swdb API
        #  To be completed by Peter Kirsch in the Polar Data Centre
        mag_input = None

    elif runparams.mag_ind_type in ("ukmo", "UKMO"):
        # Read from UKMO SWIMMR API
        print("Generating W-parameters from UKMO data...", end="")
        mag_input = mag_input.from_db(sdate, edate)
        w_indices.interp_gaps(mag_input, Wparams)
        w_indices.calculate_W(mag_input, Wparams)
        print("success.")

    else:
        print(
            "Inappropriate specifier for magnetic input data type!\n"
            "Use 'QD', 'TS', 'OMNI', 'swdb', 'ukmo'"
        )

    # maginput is a dictionary containing a timeseries of inputs

    # Interpolate maginput to a desired timeseries e.g. to pass to IRBEM with satellite track data
    t_out = np.array([sdate + timedelta(minutes=10*i) for i in range(6)])
    magin = mag_input.prepare_maginput(t_out)
