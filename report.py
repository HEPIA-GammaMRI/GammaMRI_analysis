# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:10:14 2024

@author: anastasi.kanellak
"""

#%% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy

runNb = 70
timeBinSize = 30
quantity2plot = 'A'
timeMarkers = True
plotMean = True

def plotQuantity(aDF, quantity, aDir, markersDF=None):

    if quantity == 'FWHM' or quantity == 'fwhm': quantity = 'sigma'

    aDF['dA'] = aDF['dA'].fillna(aDF['A']**0.5)
    aDF['duration'] = aDF.timeEnd-aDF.time
    aDF['timeBinMid'] = aDF.time + 0.5*aDF.duration
    aDF = aDF[aDF['duration'] == timeBinSize]

    plt.figure(dpi = 200)
    if quantity == 'sigma': 
        plt.errorbar(aDF.timeBinMid, aDF['{}'.format(quantity)]*np.sqrt(2 * np.log(2))*2, yerr=aDF['d{}'.format(quantity)]*np.sqrt(2 * np.log(2))*2, xerr=0.5*timeBinSize, label = "Measure", fmt='.', zorder=1)
    else:
        plt.errorbar(aDF.timeBinMid, aDF['{}'.format(quantity)], yerr=aDF['d{}'.format(quantity)], xerr=0.5*timeBinSize, label = "Measure", fmt='.', zorder=1)
    
    if markersDF is None:
        reportDF = pd.DataFrame()
        mean = ufloat(aDF['{}'.format(quantity)].mean(), aDF['{}'.format(quantity)].std())
        if plotMean:
            plt.plot([aDF.time.min(), aDF.timeEnd.max()], [mean.nominal_value, mean.nominal_value], color='tab:orange', linestyle='-')
            plt.fill_between([aDF.time.min(), aDF.timeEnd.max()], [mean.nominal_value-mean.std_dev, mean.nominal_value-mean.std_dev], [mean.nominal_value+mean.std_dev, mean.nominal_value+mean.std_dev], color='tab:orange', alpha=0.33)
        reportDF['time'] = [0]
        reportDF['mean_{}'.format(quantity)] = [mean.nominal_value]
        reportDF['std_{}'.format(quantity)] = [mean.std_dev]
    else:
        meanList = np.array([])
        yLim = plt.gca().get_ylim()
        for index, row in markersDF.iterrows():
            minVal = markersDF.loc[index]['Time']
            try: maxVal = markersDF.loc[index+1]['Time']
            except KeyError: maxVal = aDF.timeEnd.max()
            if quantity == 'sigma': meanDF = aDF[(aDF['timeBinMid'] > minVal) & (aDF['timeBinMid'] < maxVal)] * np.sqrt(2 * np.log(2))*2
            else: meanDF = aDF[(aDF['timeBinMid'] > minVal) & (aDF['timeBinMid'] < maxVal)]
            mean = ufloat(meanDF['{}'.format(quantity)].mean(), meanDF['{}'.format(quantity)].std())
            somme = ufloat(meanDF['{}'.format(quantity)].sum(), meanDF['{}'.format(quantity)].sum()**0.5)
            if plotMean:
                plt.plot([minVal, maxVal], [mean.nominal_value, mean.nominal_value], color='tab:orange', linestyle='-')
                plt.fill_between([minVal, maxVal], [mean.nominal_value-mean.std_dev, mean.nominal_value-mean.std_dev], [mean.nominal_value+mean.std_dev, mean.nominal_value+mean.std_dev], color='tab:orange', alpha=0.33)

            if row.Time == 0:
                boo = True
                textLabel1 = row.Label
            else:
                plt.axvline(x=row.Time, color='r', linestyle='--')

                textLabel = row.Label
                textObj = plt.text(row.Time, yLim[1]-1, textLabel, color='r', verticalalignment='bottom', horizontalalignment='right', rotation=90, visible=False)

                plt.gcf().canvas.draw()
                renderer = plt.gcf().canvas.get_renderer()
                bbox = textObj.get_window_extent(renderer=renderer)
                transform = plt.gca().transData.inverted()
                bbox_data = transform.transform(bbox)
                text_height = bbox_data[1][1] - bbox_data[0][1]
                yPos = yLim[1] - text_height - 10**(np.floor(np.log10(np.abs(yLim[1])))-3)
                plt.text(row.Time+timeBinSize, yPos, textLabel, color='r', verticalalignment='top', horizontalalignment='left', rotation=90)

                if boo:
                    boo=False
                    textObj1 = plt.text(row.Time, yLim[1]-1, textLabel1, color='r', verticalalignment='bottom', horizontalalignment='right', rotation=90, visible=False)

                    plt.gcf().canvas.draw()
                    renderer1 = plt.gcf().canvas.get_renderer()
                    bbox1 = textObj1.get_window_extent(renderer=renderer1)
                    transform1 = plt.gca().transData.inverted()
                    bbox_data1 = transform1.transform(bbox)
                    text_height1 = bbox_data1[1][1] - bbox_data1[0][1]
                    yPos = yLim[1] - text_height1 - 10**(np.floor(np.log10(np.abs(yLim[1])))-3)

                    plt.text(row.Time-timeBinSize, yPos, textLabel1, color='r', verticalalignment='top', horizontalalignment='right', rotation=90)

            meanList = np.append(meanList, mean)

        reportDF = markersDF
        reportDF['mean_{}'.format(quantity)] = unumpy.nominal_values(meanList)
        reportDF['std_{}'.format(quantity)] = unumpy.std_devs(meanList)

    if quantity == 'sigma': quantity = 'FWHM'

    if quantity == 'A': yLabeltext = 'Total counts under the peak per {}s-bin '.format(timeBinSize)
    elif quantity == 'A3s' : yLabeltext = '$3\sigma$ total counts integral under the peak per {}s-bin'.format(timeBinSize)
    elif quantity == 'mu' : yLabeltext = '$^{{129m}}Xe$ photopeak position per {}s-bin (kev)'.format(timeBinSize)
    elif quantity == 'FWHM' : yLabeltext = '$^{{129m}}Xe$ photopeak FWHM per {}s-bin (kev)'.format(timeBinSize)
    elif quantity == 'a0' : yLabeltext = 'Compton background intercept per {}s-bin'.format(timeBinSize)
    elif quantity == 'a1' : yLabeltext = 'Compton background slope per {}s-bin (1/kev)'.format(timeBinSize)
    else : yLabeltext = '{} per {}s-bin'.format(timeBinSize)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    plt.xlabel('Time [s]')
    plt.ylabel(yLabeltext)
    plt.title('run{} | ch{} | {}'.format(runNb, channelNb, quantity))
    plt.xlim(left=0)
    if plotMean: outGraph = os.path.join(aDir + '_graphs',"run{}_ch{}_{}_avg.png".format(runNb, channelNb, quantity)) 
    else: outGraph = os.path.join(aDir + '_graphs',"run{}_ch{}_{}.png".format(runNb, channelNb, quantity)) 
    plt.savefig(outGraph)

    reportDF = reportDF.reset_index(drop = True)
    outFile = os.path.join(outDir, 'run{}_ch{}_{}.csv'.format(runNb, channelNb, quantity))
    reportDF.to_csv(outFile, index = False)

    return

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

data_dir = os.path.join(main_dir, "{}_data_analysis".format(data_folder))
runDir = os.path.join(data_dir, "run{}".format(runNb))
dataDir = os.path.join(runDir, "time_bin_{}_fit".format(timeBinSize))
fitDir = os.path.join(dataDir, "fit".format(timeBinSize))
outDir = os.path.join(dataDir, "report")
outGraphDir = os.path.join(dataDir, "report_graphs")

os.chdir(dataDir)
try: os.mkdir(outDir)
except FileExistsError: pass
try: os.mkdir(outGraphDir)
except FileExistsError: pass
os.chdir(fitDir)

for file in os.listdir(fitDir):
    if file.endswith('.csv'):
        channelNb = file.split('ch')[1].split('.')[0]
        print('run {} | ch {}'.format(runNb, channelNb))
        inDF = pd.read_csv(file)

        if quantity2plot == 'asymmetry':
            print('To calculate the asymmetry, use the asymmetry.py file')
        else:
            try: markersDF = pd.read_csv(os.path.join(runDir, 'time_markers.csv'))
            except FileNotFoundError: plotQuantity(inDF, quantity2plot, outDir)
            else: 
                if timeMarkers: plotQuantity(inDF, quantity2plot, outDir, markersDF)
                else: plotQuantity(inDF, quantity2plot, outDir)


