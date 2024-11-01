# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:10:14 2024

@author: mathieu.difranco
"""

#%% Imports
import os
import sys
import struct
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import numpy as np
import numpy.ma as ma
import random
import re
import shutil

run_nb = 90
# directories control at line 386

#%% Set up save directory function
def make_analysis_dir(save_path, run_nb):
    """
    Create a folder for analysis files.

    Parameters
    ----------
    save_path : str
        Location for the analysis folder.
    run_nb : int
        ID of the measurement session.

    Returns
    -------
    save_dir : str
        Path of the created analysis folder, named 'run{run_nb}_analysis'.
        
    Raises
    ------
    FileExistsError
        If the analysis folder already exists.
    """
    save_dir = os.path.join(save_path,"run{}".format(run_nb))
    try: 
        os.mkdir(save_dir)

    except FileExistsError:
        # print("Analysis already done")
        # sys.exit()
        pass
    
    return save_dir
 
       
#%% Read input file function
def read_input(param_filename, save_dir):
    """
    Read calibration analysis parameters from a text file and return the calibration data along with other analysis parameters.

    Parameters
    ----------
    param_filename : str
        Name of the text file containing the calibration analysis parameters.
    save_dir : str
        Path of the analysis directory

    Returns
    -------
    chan_calib : DataFrame
        DataFrame containing the calibration data, including channel numbers ('chan'), 'a', 'b', 'c' calibration values, and computed energy limits ('ener_lim').
    ener_binning : int
        Value indicating the energy binning parameter used for analysis.
    time_split : int
        Value indicating the time binning parameter used for analysis.
    random_seed : int
        Value indicating the random seed parameter used for analysis.
    """
    
    print('Working on run {}...\n'.format(save_dir[-2:]))
    print('Reading calibration file... \t', end='')
    param_filename_copy = os.path.join(save_dir, "input.txt")
    shutil.copyfile(param_filename, param_filename_copy)
    # Initialize lists to store the data
    a_values = []
    b_values = []
    c_values = []
    chan_values = []
    with open(param_filename, 'r') as f:
        # Read each line of the file
        for line in f:
            # Read lines containing calibration data of each channel
            if line.find("Cal det") != -1:
                # Split line in parts separated by " "
                parts = line.split()
                
                # Read a, b, c, calibration values
                c_values.append(float(parts[0]))
                b_values.append(float(parts[1]))
                a_values.append(float(parts[2]))
                # Read channel number
                channel_str = next((part for part in parts if part.startswith("CH")), None)
                if channel_str:
                    # Extract channel number by removing the "CH" part 
                    channel_num = int(channel_str[2:])
                    chan_values.append(channel_num)
            # Read lines containing analysis parameters
            elif line.find("energy binnning") != -1:
                # Extract energy binning value
                ener_binning = int(line.split()[0])
            elif line.find("time split") != -1:
                # Extract time split value
                time_split = int(line.split()[0])
            elif line.find("seed") != -1:
                # Extract random seed value
                random_seed = int(line.split()[0])

    # Compute energy limit for each channel with calib 
    energy_limit_values = [a * 4095**2 + b * 4095 + c for a, b, c in zip(a_values, b_values, c_values)]
    
    # Create a data frame with the read values
    chan_calib = pd.DataFrame({'chan': chan_values,
                               'a': a_values, 
                               'b': b_values,
                               'c': c_values, 
                               'ener_lim': energy_limit_values})
    print('Done')
    return chan_calib, ener_binning, time_split, random_seed

#%% Read BIN file functions

# Function for reading bin file : 
def read_uint16(bin_file):
    return struct.unpack('<H', bin_file.read(2))[0]

def read_uint32(bin_file):
    return struct.unpack('<L', bin_file.read(4))[0]

def read_uint64(bin_file):
    return struct.unpack('<Q', bin_file.read(8))[0]

def read_event(bin_file):
    """
    Read an event from a binary file.

    Parameters
    ----------
    bin_file : file
        Binary file object to read from.

    Returns
    -------
    tuple
        A tuple containing the board number, channel number, timestamp, energy, and flag of the event.
    """
    try:
        board = read_uint16(bin_file)
        channel = read_uint16(bin_file)
        timestamp = read_uint64(bin_file)
        energy = read_uint16(bin_file)
        flag = read_uint32(bin_file)
        return (board, channel, timestamp, energy, flag)
    except struct.error:
        return None

def apply_calibration(ener, a, b, c, ener_lim):
    """
    Apply calibration to the energy value.

    Parameters
    ----------
    ener : float
        The original energy value.
    a, b, c : float, float, float
        Coefficients of the calibration.
    ener_lim : float
        Energy limit for calibration.

    Returns
    -------
    float
        Calibrated energy value if it lies within the energy limit, otherwise returns NaN.
    """
    calibrated_energy = a*ener**2 + b*ener + c
    if calibrated_energy>0 and calibrated_energy<ener_lim :
        return calibrated_energy
    else:
        return np.nan
    

def read_bin_file(bin_file_path, random_seed):
    """
    Read data from a binary file and apply calibration.

    Parameters
    ----------
    bin_file_path : str
        Path to the binary file.

    Returns
    -------
    channels_data : dict
        Dictionary containing calibrated data for each channel.
        The dictionary has channel numbers as keys and corresponding calibrated data as values. 
        Each value is a numpy array of shape (N, 2) where N is the number of events for that channel.
        The first column represents timestamps, and the second column represents calibrated energies.
    """
    # # Création du dictionnaire channels_data avec les clés définies par les valeurs de la colonne chan
    channels_data = {channel_name: [] for channel_name in chan_calib["chan"].tolist()}
    random.seed(random_seed)
    # Read bin file in a dict
    with open(bin_file_path, 'rb') as file:
        header = read_uint16(file)
        #print("Header : {}".format(hex(header)))
        
        print('Reading BIN file... \t', end='')
        while True:
            event = read_event(file)
            if event is None:
                break
            board, channel, timestamp, energy_raw, flag = event
            energy_raw += random.random() -0.5
            channels_data[channel].append((timestamp, energy_raw))
        print('Done')
    
    print('Applying calibration parameters : ')
    for channel in list(channels_data.keys()):
        print("Channel {} ... \t".format(channel), end = "")
        # Reshape dict : from dict = {"channel": [(time_1, ener_1), (time_2, ener_2)]}
        # to {"channel": [(timestamps), (energies))]}
        channels_data[channel] = np.reshape(channels_data[channel], (len(channels_data[channel]), 2))
        
        print('Done')    
    return channels_data

#%%

import os
import numpy as np
import matplotlib.pyplot as plt

def make_2d_hist(channels_data, chan_calib, save_dir, time_binning=1):
    """
    Generate and save 2D histograms for each channel in the provided data.
    
    This function processes the time and energy data for each channel,
    generates 2D histograms, and saves the results as text files.
    It also calibrates the energy bins and saves the calibrated
    values.

    Parameters
    ----------
    channels_data : dict
        Dictionary where keys are channel identifiers and values are numpy
        arrays with two columns: time and energy data.
    chan_calib : pandas.DataFrame
        DataFrame containing calibration coefficients 'a', 'b', 'c', and
        energy limits 'ener_lim' for each channel. Must have columns: 
        ['chan', 'a', 'b', 'c', 'ener_lim'].
    save_dir : str
        Directory where the output files will be saved. A subdirectory
        named '2d_histograms' will be created to store the results.

    Returns
    -------
    None
    """
    histogram_2d_dir = os.path.join(save_dir, "2d_histograms")
    histogram_2d_graph_dir = os.path.join(save_dir, "2d_histograms_graphs")
    try:
        os.mkdir(histogram_2d_dir)
    except FileExistsError:
        pass
    
    try:
        os.mkdir(histogram_2d_graph_dir)
    except FileExistsError:
        pass

    for channel in list(channels_data.keys()):
        print("Channel {} ... \t".format(channel), end="")
        try:

            channel_data = channels_data[channel].copy()
            time = channel_data[:, 0]
            time *= 1e-12
            energy = channel_data[:, 1]
            
            time_max = np.ceil(max(channel_data[:, 0]))
            time_bins = np.arange(start=0, stop=time_max, step=time_binning)
            # print(time_bins, len(time_bins))

            energy_bins = np.arange(start=0, stop=4095, step=1)
            a, b, c, ener_lim = chan_calib.loc[chan_calib['chan'] == channel, ['a', 'b', 'c', 'ener_lim']].values[0]
            energy_bins_calib = a * energy_bins**2 + b * energy_bins + c
            
            H, xedges, yedges = np.histogram2d(time, energy, bins=(time_bins, energy_bins))
            H = H.T
            H_plot = ma.masked_where(H == 0, H)

            yRange = [50, 300]
            yIndices = np.where((yedges[:-1] >= yRange[0]) & (yedges[:-1] < yRange[1]))[0]

            slicedH = H[:, yIndices]
            maxVal = np.max(slicedH)/10
            print(maxVal, np.max(H_plot))

            plt.figure(dpi=200)
            # Affichage de l'histogramme 2D avec les ticks calibrés en y arrondis aux entiers
            extent = [xedges[0]+time_binning/2, xedges[-1]-time_binning/2, energy_bins_calib[0], energy_bins_calib[-1]]
            plt.imshow(H_plot, aspect='auto', origin='lower', extent=extent, cmap='rainbow', norm=LogNorm())
            plt.axhline(y=yRange[0], color='red', linestyle='--', label=f'y={yRange[0]}')
            plt.axhline(y=yRange[1], color='red', linestyle='--', label=f'y={yRange[1]}')
            plt.title("run{} ch{} - 2D Histogram".format(run_nb, channel))
            plt.xlabel("Time (s)")
            plt.ylabel("Energy (keV)")
            plt.colorbar(label='Counts')

            # Sauvegarde de l'histogramme 2D
            histogram_graph_filename = "ch{}_2d_hist.png".format(channel)
            hist_graph_save_path = os.path.join(histogram_2d_graph_dir, histogram_graph_filename)
            plt.savefig(hist_graph_save_path)

            plt.figure(dpi=200)
            plt.imshow(H_plot, aspect='auto', origin='lower', extent=extent, cmap='rainbow', vmax=maxVal)
            plt.ylim(50, 300)
            plt.title("run{} ch{} - 2D histogram".format(run_nb, channel))
            plt.xlabel("Time (s)")
            plt.ylabel("Energy (keV)")
            plt.colorbar(label='Counts')
            histogram_graph_filename2 = "ch{}_2d_hist2.png".format(channel)
            hist_graph_save_path2 = os.path.join(histogram_2d_graph_dir, histogram_graph_filename2)
            plt.savefig(hist_graph_save_path2)
            
            plt.figure(dpi=200)
            energy_histogram = np.sum(H, axis=1)
            plt.step(energy_bins_calib[0:-1], energy_histogram)
            plt.title("run{} ch{} - Energy histogram".format(run_nb, channel)) 
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            spectrum_graph_filename = "ch{}_spectrum.png".format(channel)
            spectrum_graph_save_path = os.path.join(histogram_2d_graph_dir, spectrum_graph_filename)
            plt.savefig(spectrum_graph_save_path)
            
            plt.figure(dpi=200)
            rate_histogram = np.sum(H, axis=0)
            plt.step(time_bins[0:-1], rate_histogram)
            plt.title("run{} ch{} - Rate histogram".format(run_nb, channel))
            plt.xlabel("Time (s)")
            plt.ylabel("Counts")
            rates_graph_filename = "ch{}_rates.png".format(channel)
            rates_graph_save_path = os.path.join(histogram_2d_graph_dir, rates_graph_filename)
            plt.savefig(rates_graph_save_path)
            # plt.show()
            
            # Sauvegarde de l'histogramme 2D
            histogram_filename = "ch{}_2d_hist.txt".format(channel)
            hist_save_path = os.path.join(histogram_2d_dir, histogram_filename)
            np.savetxt(hist_save_path, H, delimiter="\t", fmt="%d")
            
            # Sauvegarde de l'histogramme 2D
            histogram_filename2 = "ch{}_energy_proj_calib.txt".format(channel)
            hist_save_path2 = os.path.join(histogram_2d_dir, histogram_filename2)
            np.savetxt(hist_save_path2, np.sum(H, axis=1), delimiter="\t", fmt="%d")

            # Sauvegarde des energy_bins_calib
            energy_bins_filename = "ch{}_energy_bins_calib.txt".format(channel)
            energy_bins_save_path = os.path.join(histogram_2d_dir, energy_bins_filename)
            np.savetxt(energy_bins_save_path, energy_bins_calib, delimiter="\t", fmt="%.2f")
            
            # Sauvegarde des energy_bins_calib
            time_bins_filename = "ch{}_time_bins_calib.txt".format(channel)
            time_bins_save_path = os.path.join(histogram_2d_dir, time_bins_filename)
            np.savetxt(time_bins_save_path, time_bins, delimiter="\t", fmt="%i")
            
            print("Done")
        except ValueError:
            print("No data in this channel")

#%% Main

# main_dir
# |
# |__result_path
# |__data_dir
#    |__input.txt
#    |__run

# This is the main directory. Inside there are two folders according to the structure scheme above.
# In the folder 'data_dir', you depose the compass folders with the raw data and the 'input.txt' file.
# In the folder 'result_dir', the script will generate all the result files.
# If the 'result_dir' folder doesn't exist, the script will create a new folder named 'data_folder' followed by '_data_analysis'
# All the subfolders in 'result_dir' will be created atutomatically
main_dir = '/Users/akanellako/Documents/GAMMA-MRI_data'  
data_folder = 'july2024-compass'
os.chdir(main_dir)

result_path = os.path.join(main_dir, "{}_data_analysis".format(data_folder))
try: os.mkdir(result_path)
except FileExistsError: pass
result_dir = make_analysis_dir(result_path, run_nb)

param_filename = os.path.join(data_folder, "input.txt")

data_dir = os.path.join(main_dir, data_folder, "run{}/RAW".format(run_nb))
data_filename = "DataR_run{}.BIN".format(run_nb)
data_path = os.path.join(data_dir, data_filename)

## -----Read text file containing the ananlysis parameters and calibration values -----##
chan_calib, ener_binning, time_split, random_seed = read_input(param_filename, result_dir)


## ------Read BIN file ---------##
channels_data = read_bin_file(data_path, random_seed)

make_2d_hist(channels_data, chan_calib, result_dir)

# dirFile = open("main_dir.txt", "r")
# main_dir = dirFile.read()
# os.chdir(main_dir)


