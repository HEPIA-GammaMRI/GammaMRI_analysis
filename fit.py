# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:13:55 2024

@author: anastasi.kanellak
"""

#%% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import satlas as sat
import math

runNb = 68
timeBinSize = 30

def linearModel(x, par):

    x0 = par[0]
    a0 = par[1]
    a1 = par[2]

    linear = (a1*(x-x0))+a0

    return linear

def gaussianModel(x, par):

    A = par[0]
    mu = par[1]
    sigma = par[2]

    gaussian = A * np.exp(-0.5*((x - mu)/sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    return gaussian

def gammaPeakModel(x, par):

    A = par[0]
    mu = par[1]
    sigma = par[2]
    x0 = par[3]
    a0 = par[4]
    a1 = par[5]

    linear = (a1*(x-x0))+a0

    gaussian = A * np.exp(-0.5*((x - mu)/sigma)**2) / (sigma * np.sqrt(2 * np.pi))


    return linear + gaussian

def fitGammaPeak(xdata, ydata, init=False, showInit=False):
    """
    Fit a histogram of energy data with a Gaussian plus background function.

    Parameters
    ----------
    xdata : array-like
        Array containing the energy values.
    ydata : array-like
        Array containing the corresponding counts.
    show_graph : bool, optional
        Whether to display the fitted histogram graph. Default is True.
    title : str, optional
        Title of the histogram graph. Default is None.

    Returns
    -------
    A : float
        Amplitude of the Gaussian component.
    mu : float
        Standard deviation of the Gaussian component.
    """

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    if  not init:
        # Initialisation of model parameters
        init_A = ydata.max()*20
        init_mu = 196
        init_sigma = 50/2.355
        init_a1 = (np.mean(ydata[-5:])-np.mean(ydata[:5]))/(xdata[-1]-xdata[0])    
        init_a0 = np.mean(ydata[:5])+5
        init_x0 = xdata[0]
        init = [init_A, init_mu, init_sigma, init_x0, init_a0, init_a1]
    else:
        init = init        

    model = sat.MiscModel(gammaPeakModel, init, ['A', 'mu', 'sigma', 'x0', 'a0', 'a1'])
    model.set_boundaries({'mu': {'min': 180, 'max':220},
                            'sigma': {'min': 8, 'max': 100},
                            'a0': {'min': 0, 'max': ydata.max()},
                            'a1': {'max': 0},
                            'x0': {'min': 0},
                            'A': {'min': 0}})

    if showInit:
        plt.figure(dpi = 200)
        plt.errorbar(xdata, ydata, yerr=ydata**0.5, label = "Measure", fmt='.', zorder=1)
        plt.plot(xdata, model(xdata), label="Fit", linestyle="--", zorder=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
        plt.xlabel("Energy [keV]")
        plt.ylabel("Counts")
        plt.title('init')
        plt.show()

    success, message = sat.chisquare_fit(model, xdata, ydata, yerr=ydata**0.5)
    print('Fit completed |', success, '|', message)
    
    modelDF = model.get_result_frame()

    return model, modelDF

def rebinning(anArray, binSize):
    _, bins = anArray.shape
    splitBins = []
    binsList = []
    for j in range(0, bins, binSize):
        aBin = anArray[:, j:j+binSize]
        splitBins.append(aBin)
        binsList.append(j)
    return binsList, splitBins

def main(timeBinDir, saveDir):
    try: os.mkdir(saveDir)
    except FileExistsError: pass
    os.chdir(dataDir)

    lowerEnergy = 140
    upperEnergy = 250

    for file in os.listdir(dataDir):
        if file.endswith('_2d_hist.txt'):
            outDir = os.path.join(saveDir,"fit")
            try: os.mkdir(outDir)
            except FileExistsError: pass
            channel= file[:-12]
            channelNb = channel[2:]
            array = np.loadtxt(file)
            energyBins = np.loadtxt('{}_energy_bins_calib.txt'.format(channel))
            timeBins = np.loadtxt('{}_time_bins_calib.txt'.format(channel))
            outFile = os.path.join(outDir,"run{}_ch{}.csv".format(runNb, channelNb))

            resultDF = pd.DataFrame()

            timeBins, rebinnedArray = rebinning(array, timeBinSize)

            boo = True

            for i in range(len(rebinnedArray)):
            # for anArray in rebinnedArray:

                anArray = rebinnedArray[i]
                timeBin = timeBins[i]

                if timeBin == timeBins[-1]: timeBinEnd = array.shape[1]
                else : timeBinEnd = timeBin+timeBinSize


                print('\n run{} | channel{} | time bin {}->{}'.format(runNb, channelNb, timeBin, timeBinEnd))

                aSpectrum = np.sum(anArray, axis=1)

                energySeries = pd.Series(energyBins.flatten(), name='energy')
                countSeries = pd.Series(aSpectrum.flatten(), name='counts')

                histDF = pd.concat([energySeries, countSeries], axis=1)
                histOrigDF = histDF
                histDF = histDF.loc[(histDF["energy"] > lowerEnergy) & (histDF["energy"] < upperEnergy)].reset_index(drop = True)
                histDF = histDF[(histDF.T != 0).all()]
                histDF = histDF.dropna()

                if timeBinEnd-timeBin < timeBinSize/2:
                    aDF = pd.DataFrame([[timeBin, timeBinEnd, np.nan, np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan,
                        np.nan, np.nan, np.nan]],
                        columns=['time', 'timeEnd', 'A3s', 'dA3s',
                                'A', 'dA', 'mu', 'dmu', 'sigma', 'dsigma',
                                'x0', 'dx0', 'a0', 'da0', 'a1', 'da1',
                                'Chi2', 'NDoF', 'Red. Chi2'])
                else:
                    if boo:
                        model, modelDF = fitGammaPeak(xdata=histDF.energy.values, ydata=histDF.counts.values, showInit=False)
                        boo = False
                    else:
                        model, modelDF = fitGammaPeak(xdata=histDF.energy, ydata=histDF.counts,
                            init=[modelDF.A.Value.values[0], modelDF.mu.Value.values[0], modelDF.sigma.Value.values[0], modelDF.x0.Value.values[0], modelDF.a0.Value.values[0], modelDF.a1.Value.values[0]])
                    
                    lower3Sigma = modelDF.mu.Value.values[0]-3*modelDF.sigma.Value.values[0]
                    upper3Sigma = modelDF.mu.Value.values[0]+3*modelDF.sigma.Value.values[0]
                    hist3SigmaDF = histOrigDF.loc[(histOrigDF["energy"] > lower3Sigma) & (histOrigDF["energy"] < upper3Sigma)].reset_index(drop = True)
                    A3Sigma = hist3SigmaDF.counts.sum()

                    binPerKeV = len(histDF)/(upperEnergy-lowerEnergy)

                    try: modelDF.A.Uncertainty.values[0]*binPerKeV
                    except TypeError: anUnc = modelDF.A.Uncertainty.values[0] 
                    else: anUnc = modelDF.A.Uncertainty.values[0]*binPerKeV

                    aDF = pd.DataFrame([[timeBin, timeBinEnd, A3Sigma, A3Sigma**0.5,
                        modelDF.A.Value.values[0]*binPerKeV, anUnc,
                        modelDF.mu.Value.values[0], modelDF.mu.Uncertainty.values[0],
                        modelDF.sigma.Value.values[0], modelDF.sigma.Uncertainty.values[0],
                        modelDF.x0.Value.values[0], modelDF.x0.Uncertainty.values[0],
                        modelDF.a0.Value.values[0], modelDF.a0.Uncertainty.values[0],
                        modelDF.a1.Value.values[0], modelDF.a1.Uncertainty.values[0],
                        modelDF.Chisquare.values[0], modelDF.NDoF.values[0], float(modelDF.Chisquare.values[0]/modelDF.NDoF.values[0])]],
                        columns=['time', 'timeEnd', 'A3s', 'dA3s',
                                'A', 'dA', 'mu', 'dmu', 'sigma', 'dsigma',
                                'x0', 'dx0', 'a0', 'da0', 'a1', 'da1',
                                'Chi2', 'NDoF', 'Red. Chi2'])

                resultDF = pd.concat([resultDF, aDF])

                outGraph = os.path.join(saveDir,'fit_graphs')
                try: os.mkdir(outGraph)
                except FileExistsError: pass
                outGraph = os.path.join(outGraph,'ch{}'.format(channelNb))
                try: os.mkdir(outGraph)
                except FileExistsError: pass
                outGraph = os.path.join(outGraph,"run{}_ch{}_tbin{}.png".format(runNb, channelNb, timeBin))

                lowerEnergyPlot = lowerEnergy - 20
                upperEnergyPlot = upperEnergy + 20
                histOrigDF = histOrigDF.loc[(histOrigDF["energy"] > lowerEnergyPlot) & (histOrigDF["energy"] < upperEnergyPlot)].reset_index(drop = True)

                if timeBinEnd-timeBin < timeBinSize/2: pass
                else:
                    background = sat.MiscModel(linearModel, [modelDF.x0.Value.values[0], modelDF.a0.Value.values[0], modelDF.a1.Value.values[0]])
                    gaussianPeak = sat.MiscModel(gaussianModel, [modelDF.A.Value.values[0], modelDF.mu.Value.values[0], modelDF.sigma.Value.values[0]])

                plt.figure(dpi = 200)
                plt.errorbar(histOrigDF.energy, histOrigDF.counts, yerr=(histOrigDF.counts)**0.5, label = "Measure", fmt='.', zorder=1)
                if timeBinEnd-timeBin < timeBinSize/2: pass
                else:
                    plt.plot(histDF.energy, model(histDF.energy), label = "Fit", linestyle = "--", zorder=2)
                    plt.plot(histDF.energy, background(histDF.energy), label = "bkgrn", linestyle = "--", color='tab:green',zorder=3)
                    plt.plot([modelDF.mu.Value.values[0], modelDF.mu.Value.values[0]],[histDF.counts.min(), histDF.counts.max()], color='k', linestyle=':')
                    plt.plot([lower3Sigma, lower3Sigma],[histDF.counts.min(), histDF.counts.max()], color='k', linestyle=':')
                    plt.plot([upper3Sigma, upper3Sigma],[histDF.counts.min(), histDF.counts.max()], color='k', linestyle=':')
                    plt.fill_between(histDF.energy, background(histDF.energy), model(histDF.energy), label = "peak", alpha=0.33, color='tab:orange')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
                plt.xlabel("Energy [keV]")
                plt.ylabel("Counts")
                plt.title('run{} | ch{} | tbin{}-{}'.format(runNb, channelNb, timeBin, timeBinEnd))         
                plt.savefig(outGraph)

            resultDF = resultDF.sort_values(by = "time")
            resultDF = resultDF.reset_index(drop = True)
            resultDF.to_csv(outFile, index = False)

#%% Main

# main_dir
# |
# |__result_path
#    |__run
# |__data_dir
#    |__input.txt
#    |__run

main_dir = '/Users/akanellako/Documents/GAMMA-MRI_data'  
data_folder = 'july2024-compass'
os.chdir(main_dir)
result_dir = os.path.join(main_dir, "{}_data_analysis".format(data_folder))
runDir = os.path.join(result_dir, "run{}".format(runNb))
saveDir = os.path.join(runDir, "time_bin_{}_fit".format(timeBinSize))
dataDir = os.path.join(runDir, "2d_histograms")

main(dataDir, saveDir)


