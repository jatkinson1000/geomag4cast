import os
from datetime import datetime, timedelta
import urllib.request
import urllib.error

import requests, zipfile, io


def check_dirpaths(sdate, edate, out_dir):
    """ function check_dirpaths
    checks directory paths exist for all years required according to
    naming convention, and creates them in preparation if not.

    Parameters
    ----------
    sdate, edate : datetime object
        start and end dates of time range to be processed
    out_dir : string
        directory path to where any results from this work will be stored

    Returns
    -------
    None

    """
    for year in range(sdate.year, edate.year+1):
        dirpath = f'{out_dir}{year:04d}/'
        if not os.path.exists(dirpath):
            print('Directory {} does not exist. Creating...'.format(dirpath))
            os.makedirs(dirpath)

    return None


def retrieve_TS_data_year(prcdate, out_dir, try_download=True):
    """ function retrieve_TS_data_day
    Function to check for the existence of and retrieve TS data files for a single year

    Parameters
    ----------
    prcdate : datetime
        date to be processed
    out_dir : string
        directory to store TS files
    try_download : bool
        If a file is missing should we try and download (True)
        or skip this satellite&day (False)

    Returns
    -------
    proceed : bool
        file present. OK to proceed with processing

    """
    proceed = True

    datapath = f'{prcdate.year:04d}_OMNI_5m_with_TS05_variables.dat'
    filepath_prc = f'{out_dir}{datapath}'

    if not os.path.exists(filepath_prc):
        print('\t\tCannot locate processed TS datafile for ', prcdate.date())
        url_zip = (f'http://geo.phys.spbu.ru/~tsyganenko/TS05_data_and_stuff/'
                   f'{prcdate.year:04d}_OMNI_5m_with_TS05_variables.zip')
        url_dat = (f'http://geo.phys.spbu.ru/~tsyganenko/TS05_data_and_stuff/'
                   f'{prcdate.year:04d}_OMNI_5m_with_TS05_variables.dat')
        if try_download:
            print('\t\tAttempting to download from online source: ', url_zip)
            try:
                r = requests.get(url_zip)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(out_dir)
            except requests.exceptions.SSLError as e:
                print(f'\t\tCannot download zip file for {prcdate.date()} so trying .dat...')
                print(e)
                try:
                    urllib.request.urlretrieve(url_dat, filepath_prc)
                except urllib.error.URLError as e:
                    print('\t\tCannot download file for ', prcdate.date())
                    print(e)
                    proceed = False
                    # log_missing_day(prcdate, out_dir, res)
        else:
            proceed = False
            # log_missing_day(prcdate, out_dir, res)

    else:
        print('\t\tUsing existing TS file')

    return proceed


def retrieve_Tsyganenko_Tsyganenko(sdate, edate, datadir, outfile):
    """
    Reads in data from Qin-Denton data file taken from the RBSP data portal (see reference) or prepared by user
    to populate class with relevant variables.

    Parameters
    ----------
    sdate, edate : datetime
        times between which you wish to save data
    datadir : str
        directory to put QD files
    outfile : str
        filepath for saving the combined data

    Returns
    -------
    None : None
        No returns, data is saved into a file as specified at input

    References
    -------
    https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/
    Database of downloadable Tsyganenko files.
    Note - You may need to combine data into a single file or pre-process yourself if modelling an interval that crosses
    multiple file date ranges.
    """

    prcdate = datetime(sdate.year, sdate.month, sdate.day, hour=0, minute=0, second=0)
    firstfile=True

    while prcdate.year <= edate.year:
        success = retrieve_TS_data_year(prcdate, datadir, try_download=True)
        if success:
            datapath = f'{prcdate.year:04d}_OMNI_5m_with_TS05_variables.dat'
            filepath_prc = f'{datadir}{datapath}'
            with open(filepath_prc, 'r') as rf:
                lines = rf.readlines()
                if firstfile:
                    with open(f'{datadir}/{outfile}', 'w') as wf:
                        wf.writelines(lines)
                    firstfile = False
                else:
                    with open(f'{datadir}/{outfile}', 'a') as wf:
                        lines = [l for l in lines if not l.startswith("#")]
                        wf.writelines(lines)

            # extract data and append
            # Write note to screen/file to say OK
        prcdate = prcdate.replace(year=prcdate.year+1)

    return None


def retrieve_QD_data_day(prcdate, out_dir, res, try_download=True):
    """ function retrieve_QD_data_day
    Function to check for the existence of and retrieve QD data files for a single day

    Parameters
    ----------
    prcdate : datetime
        date to be processed
    out_dir : string
        directory to store QD files
    res : str
        identifier for the resolution of file to download
    try_download : bool
        If a file is missing should we try and download (True)
        or skip this satellite&day (False)

    Returns
    -------
    proceed : bool
        file present. OK to proceed with processing

    """
    proceed = True

    datapath = f'{prcdate.year:04d}/QinDenton_{prcdate.year:04d}{prcdate.month:02d}{prcdate.day:02d}_{res}.txt'
    filepath_prc = f'{out_dir}{datapath}'

    if not os.path.exists(filepath_prc):
        print('\t\tCannot locate processed QD datafile for ', prcdate.date())
        url_prc = f'https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/{datapath}'

        if try_download:
            print(f'\t\tAttempting to download from online source: {url_prc}...', end='')
            try:
                urllib.request.urlretrieve(url_prc, filepath_prc)
                print('success')
            except urllib.error.URLError as e:
                # print('\t\tCannot download file for ', prcdate.date())
                print('download failed')
                print(e)
                proceed = False
                # log_missing_day(prcdate, out_dir, res)

        else:
            proceed = False
            # log_missing_day(prcdate, out_dir, res)

    else:
        print('\t\tUsing existing QD file')

    return proceed


def retrieve_QinDenton_RBSP(sdate, edate, datadir, outfile, freq='5m'):
    """
    Reads in data from Qin-Denton data file taken from the RBSP data portal (see reference) or prepared by user
    to populate class with relevant variables.

    Parameters
    ----------
    sdate, edate : datetime
        times between which you wish to save data
    datadir : str
        directory to put QD files
    outfile : str
        filepath for saving the combined data
    freq : str
        identifier for data frequency to download

    Returns
    -------
    None : None
        No returns, data is saved into a file as specified at input

    References
    -------
    https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/
    Database of downloadable Qin-Denton files.
    Note - You may need to combine data into a single file or pre-process yourself if modelling an interval that crosses
    multiple file date ranges.

    NB there are 1hr, 5min, and 1min data files available.
    It would appear that certain indices are linearly interpolated to higher resolution, e.g. Kp
    Need to be aware of this in application.g
    """

    if freq == '5m':
        res = '5min'
    elif freq == '1m':
        res = '1min'
    elif (freq == '60m') or (freq == '1h') or (freq == '1hr'):
        res = 'hour'
    else:
        raise ValueError(f'Frequency {freq} for data to download not recognised.\nPlease review and try again')

    prcdate = datetime(sdate.year, sdate.month, sdate.day, hour=0, minute=0, second=0)
    firstfile = True

    while prcdate <= edate:
        success = retrieve_QD_data_day(prcdate, datadir, res, try_download=True)
        if success:
            datapath = f'{prcdate.year:04d}/QinDenton_{prcdate.year:04d}{prcdate.month:02d}{prcdate.day:02d}_{res}.txt'
            filepath_prc = f'{datadir}{datapath}'
            with open(filepath_prc, 'r') as rf:
                lines = rf.readlines()
                if firstfile:
                    with open(f'{datadir}/{outfile}', 'w') as wf:
                        wf.writelines(lines)
                    firstfile = False
                else:
                    with open(f'{datadir}/{outfile}', 'a') as wf:
                        lines = [l for l in lines if not l.startswith("#")]
                        wf.writelines(lines)

            # extract data and append
            # Write note to screen/file to say OK
        prcdate += timedelta(days=1)

    return None


if __name__ == '__main__':
    # Retrieve QD data for a specified time period
    # Store individual files in datadir
    # Generate overall file in outfile
    # NB need to go one day beyond to capture midnight on final day
    sdate = datetime(2000, 7, 1)
    edate = datetime(2000, 7, 31)
    datadir = '../data/'
    outfile = 'QDInput_Jul_2000.dat'
    check_dirpaths(sdate, edate, datadir)
    retrieve_QinDenton_RBSP(sdate, edate, datadir, outfile, freq='5m')

    # # Retrieve TS data for a specified time period
    # # Store individual files in datadir
    # # Generate overall file in outfile
    # sdate = datetime(2015, 1, 1)
    # edate = datetime(2015, 12, 31)
    # datadir = '../data/'
    # outfile = 'TSInput_2015.dat'
    # check_dirpaths(sdate, edate, datadir)
    # retrieve_Tsyganenko_Tsyganenko(sdate, edate, datadir, outfile)
