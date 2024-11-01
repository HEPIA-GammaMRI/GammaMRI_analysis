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
timeBinSize = 60
timeMarkers = True
plotMean = False

detLch = 18

def calculateAsymmetry(aDF, det1, det2):

    det1A = unumpy.uarray(aDF['ch{}_A'.format(det1)].tolist(), aDF['ch{}_dA'.format(det1)].tolist())
    det2A = unumpy.uarray(aDF['ch{}_A'.format(det2)].tolist(), aDF['ch{}_dA'.format(det2)].tolist())

    det1A3s = unumpy.uarray(aDF['ch{}_A3s'.format(det1)].tolist(), aDF['ch{}_dA3s'.format(det1)].tolist())
    det2A3s = unumpy.uarray(aDF['ch{}_A3s'.format(det2)].tolist(), aDF['ch{}_dA3s'.format(det2)].tolist())

    asymmetryA = ((det2A - det1A) / (det2A + det1A))*100
    asymmetryA3s = ((det2A3s - det1A3s) / (det2A3s + det1A3s))*100

    aDF['{}v{}_asymmetry_A'.format(i, j)] = unumpy.nominal_values(asymmetryA)
    aDF['{}v{}_dasymmetry_A'.format(i, j)] = unumpy.std_devs(asymmetryA)
    aDF['{}v{}_asymmetry_A3s'.format(i, j)] = unumpy.nominal_values(asymmetryA3s)
    aDF['{}v{}_dasymmetry_A3s'.format(i, j)] = unumpy.std_devs(asymmetryA3s)

    return aDF

def plotAsymmetry(aDF, det1, det2, aDir, markersDF=None):

    if markersDF is None: reportDF = pd.DataFrame()

    for quantity in ['A', 'A3s']:
        plt.figure(dpi = 200)
        plt.errorbar(aDF.timeBinMid, aDF['{}v{}_asymmetry_{}'.format(det1, det2, quantity)], yerr=aDF['{}v{}_dasymmetry_{}'.format(det1, det2, quantity)], xerr=0.5*timeBinSize, label = "Measure", fmt='.', zorder=1)

        if markersDF is None:
            mean = ufloat(aDF['{}v{}_asymmetry_{}'.format(det1, det2, quantity)].mean(), aDF['{}v{}_asymmetry_{}'.format(det1, det2, quantity)].std())
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
                mean = ufloat(meanDF['{}v{}_asymmetry_{}'.format(det1, det2, quantity)].mean(), meanDF['{}v{}_asymmetry_{}'.format(det1, det2, quantity)].std())
                if plotMean:
                    plt.plot([minVal, maxVal], [mean.nominal_value, mean.nominal_value], color='tab:orange', linestyle='-')
                    plt.fill_between([minVal, maxVal], [mean.nominal_value-mean.std_dev, mean.nominal_value-mean.std_dev], [mean.nominal_value+mean.std_dev, mean.nominal_value+mean.std_dev], color='tab:orange', alpha=0.33)

                if row.Time == 0:
                    lcBoo = True
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

                    if lcBoo:
                        lcBoo=False
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
            reportDF['mean_asym{}'.format(quantity)] = unumpy.nominal_values(meanList)
            reportDF['std_asym{}'.format(quantity)] = unumpy.std_devs(meanList)
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
        plt.xlabel('Time [s]')
        plt.ylabel('Asymmetry ch{} vs. ch{}'.format(det1, det2))
        plt.title('run{} | Asymmetry {} | ch{} vs. ch{}'.format(runNb, quantity, det1, det2))
        plt.xlim(left=0)
        outGraph = os.path.join(aDir + '_graphs',"run{}_ch{}vs{}_asym{}.png".format(runNb, det1, det2, quantity))
        plt.savefig(outGraph)

    reportDF = reportDF.reset_index(drop = True)
    outFile = os.path.join(aDir,"run{}_ch{}vs{}.csv".format(runNb, det1, det2)) 
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

result_dir = os.path.join(main_dir, "{}_data_analysis".format(data_folder))
runDir = os.path.join(result_dir, "run{}".format(runNb))
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

asymmetryDF = pd.DataFrame()
channelList = []
processedList = set()

boo = True

for file in os.listdir(fitDir):
    if file.endswith('.csv'):
        channelNb = file.split('ch')[1].split('.')[0]
        print('run {} | ch {}'.format(runNb, channelNb))
        inDF = pd.read_csv(file)

        channelList.append(channelNb)

        if boo:
            asymmetryDF['time'] = inDF.time
            asymmetryDF['timeEnd'] = inDF.timeEnd
            asymmetryDF['duration'] = asymmetryDF.timeEnd-asymmetryDF.time
            asymmetryDF['timeBinMid'] = asymmetryDF.time + 0.5*asymmetryDF.duration
            asymmetryDF = asymmetryDF[asymmetryDF['duration'] == timeBinSize]
            boo = False
        else: pass

        asymmetryDF['ch{}_A3s'.format(channelNb)] = inDF.A3s
        asymmetryDF['ch{}_dA3s'.format(channelNb)] = inDF.dA3s
        asymmetryDF['ch{}_A'.format(channelNb)] = inDF.A
        asymmetryDF['ch{}_dA'.format(channelNb)] = inDF.dA

for i in channelList:
    for j in channelList:
        if i != j:
            combo = tuple(sorted((i,j)))
            if combo not in processedList:
                asymmetryDF = calculateAsymmetry(asymmetryDF, i, j)

                try: markersDF = pd.read_csv(os.path.join(runDir, 'time_markers.csv'))
                except FileNotFoundError: plotAsymmetry(asymmetryDF, i, j, outDir)
                else: 
                    if timeMarkers: plotAsymmetry(asymmetryDF, i, j, outDir, markersDF)
                    else: plotAsymmetry(asymmetryDF, i, j, outDir)

                processedList.add(combo)

outFile = os.path.join(outDir, 'run{}_asymmetry.csv'.format(runNb))
asymmetryDF = asymmetryDF.reset_index(drop = True)
asymmetryDF.to_csv(outFile, index = False)


