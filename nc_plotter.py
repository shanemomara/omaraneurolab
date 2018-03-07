# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:51:18 2017

@author: Raju
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:39:19 2017

@author: Raju
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from nc_ext import find, CircStat
from enum import Enum



BLUE= '#1f77b4'
RED= '#d62728'

class Singleton(object):
    def __new__(cls, *arg, **kwarg):
        if not hasattr(cls, '_instance'):            
            cls._instance= super().__new__(cls, *arg, **kwarg)
        return cls._instance

class NPlotter(Singleton): 
    
    def setBackend(self, backend):
        if backend:
            plt.switch_backend(backend)
        
    def waveProperty(self, waveData, plots= [2, 2]):        
        # Wave property analysis
        fig1, ax = plt.subplots(plots[0], plots[0])
        ax= ax.flatten()
        # Plot waves    
        for i in np.arange(len(ax)):
            ax[i].plot(waveData['Mean wave'][:, i], color= 'black', linewidth= 2.0)
            ax[i].plot(waveData['Mean wave'][:, i]+waveData['Std wave'][:, i], color= 'green', linestyle= 'dashed')
            ax[i].plot(waveData['Mean wave'][:, i]-waveData['Std wave'][:, i], color= 'green', linestyle= 'dashed')
        plt.show()    
        return fig1
                
    def isi(self, isiData):
        # Plot ISI
        # histogram
        fig1= plt.figure()
        ax= plt.gca()
        ax.bar(isiData['isiBins'], isiData['isiHist'], color= 'darkblue', edgecolor= 'darkblue', rasterized= True)
        ax.plot([5, 5,], [0, isiData['maxCount']], linestyle= 'dashed', linewidth= 2, color= 'red')
        ax.set_title('Distribution of inter-spike interval')
        ax.set_xlabel('ISI (ms)')
        ax.set_ylabel('Spike count')
        max_axis= isiData['isiBins'].max()
        max_axisLog=np.ceil(np.log10(max_axis))
                
        ## ISI scatterplot
        fig2= plt.figure()
        fig2.suptitle('Distribution of ISI \n (before and after spike)')
        # Scatter
        ax= fig2.add_subplot(211)
        ax.loglog(isiData['isiBefore'], isiData['isiAfter'], axes= ax, \
                linestyle= ' ', marker= 'o', markersize= 1, \
                markeredgecolor= 'black', markerfacecolor= None, rasterized= True)
        ax.autoscale(enable= True, axis= 'both', tight= True)
        ax.plot(ax.get_xlim(), [5, 5], linestyle= 'dashed', linewidth= 2, color= 'red')
        ax.set_aspect(1)
        #    ax.set_xlabel('Interval before (ms)')
        ax.set_ylabel('Interval after (ms)')
        
        #    
        logBins= np.logspace(0, max_axisLog, max_axisLog*70)
        joint_count, xedges, yedges= np.histogram2d(isiData['isiBefore'], isiData['isiAfter'], bins= logBins)
        
        # Scatter colored    
        _extent= [xedges[0], xedges[-2], yedges[0], yedges[-2]]
        
        ax = fig2.add_subplot(212, aspect= 'equal')
        
        c_map= plt.cm.jet
        c_map.set_under('white')
        ax.pcolormesh(xedges[0:-1], yedges[0:-1], joint_count, cmap= c_map, vmin= 1, rasterized= True)
        ax.plot(ax.get_xlim(), [5, 5], linestyle= 'dashed', linewidth= 2, color= 'red')
        plt.axis(_extent)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Interval before (ms)')
        ax.set_ylabel('Interval after (ms)')

        return fig1, fig2
                
    def isiCorr(self, isiCorrData):            
        fig1= plt.figure()
        ax= plt.gca()
        ax.bar(isiCorrData['isiCorrBins'], isiCorrData['isiCorr'], color= 'darkblue', edgecolor= 'darkblue', rasterized= True)
        ax.set_title('Autocorrelation Histogram \n' + '('+ str(abs(isiCorrData['isiCorrBins'].min()))+ 'ms)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Counts')
        
        return fig1
        
    def thetaCell(self, plotData):
        fig1= plt.figure()
        ax= plt.gca()
        ax.bar(plotData['isiCorrBins'], plotData['isiCorr'], color= 'darkblue', edgecolor= 'darkblue', rasterized= True)
        ax.plot(plotData['isiCorrBins'], plotData['corrFit'], linewidth= 2, color= 'red')
        ax.set_title('Autocorrelation Histogram \n' + '('+ str(abs(plotData['isiCorrBins'].min()))+ 'ms)')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Counts')

        return fig1        
        
    def lfpSpectrum(self, plotData):
        fig1= plt.figure()
        ax=plt.gca()
        ax.plot(plotData['f'], plotData['Pxx'], linewidth= 2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD ()')
        _extent= [0, plotData['f'].max(), 0, plotData['Pxx'].max()]
        plt.axis(_extent)

        return fig1
        
    def lfpSpectrum_tr(self, plotData):
        fig1= plt.figure()
        ax=plt.gca()
        c_map= plt.cm.jet
        pcm= ax.pcolormesh(plotData['t'], plotData['f'], plotData['Sxx'], cmap= c_map)
        _extent= [0, plotData['t'].max(), 0, plotData['f'].max()]
        plt.axis(_extent)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Frequency (Hz)')
        fig1.colorbar(pcm)
        
        return fig1
        
    def PLV(self, plvData):
        f= plvData['f']
        t= plvData['t']
        STA= plvData['STA']
        fSTA= plvData['fSTA']
        STP= plvData['STP']
        SFC= plvData['SFC']
        PLV= plvData['PLV']
        
        fig1= plt.figure()
        ax= plt.gca()
        ax.plot(t, STA, linewidth= 2, color= 'darkblue')
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('Spike-triggered average (STA)')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('STA (uV)')        
        
        fig2= plt.figure()
        ax= fig2.add_subplot(221)
        ax.plot(f,fSTA, linewidth= 2, color= 'darkblue')
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('fft of STA')        
        
        ax= fig2.add_subplot(222)
        ax.plot(f, STP, linewidth= 2, color= 'darkblue')
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('STP')
                
        ax= fig2.add_subplot(223)
        ax.plot(f, SFC, linewidth= 2, color= 'darkblue')
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('SFC')        
        
        ax= fig2.add_subplot(224)
        ax.plot(f, PLV, linewidth= 2, color= 'darkblue')
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('PLV')
        
        for ax in fig2.axes:
            ax.set_xlabel('Frequency (Hz)')
        
        fig2.suptitle('Frequency analysis of spike-triggered lfp metrics')

        return fig1, fig2
        
    def PLV_tr(self, plvData):
        
        offset= plvData['offset']
        f= plvData['f']
        fSTA= plvData['fSTA']
#        STP= plvData['STP']
        SFC= plvData['SFC']
        PLV= plvData['PLV']
        
        fig1= plt.figure()
        ax=plt.gca()
        c_map= plt.cm.jet
        pcm= ax.pcolormesh(offset, f, fSTA, cmap= c_map, rasterized= True)
        plt.title('Time-resolved fSTA')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Frequency (Hz)')
        fig1.colorbar(pcm)
        
        fig2= plt.figure()
        ax=plt.gca()
        c_map= plt.cm.jet
        pcm= ax.pcolormesh(offset, f, SFC, cmap= c_map, rasterized= True)
        plt.title('Time-resolved SFC')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Frequency (Hz)')
        fig2.colorbar(pcm)
                
        fig3= plt.figure()
        ax=plt.gca()
        c_map= plt.cm.jet
        pcm= ax.pcolormesh(offset, f, PLV, cmap= c_map, rasterized= True)
        plt.title('Time-resolved PLV')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Frequency (Hz)')
        fig3.colorbar(pcm)

        return fig1, fig2, fig3
                
    def PLV_bs(self, plvData):
        f= plvData['f']
        t= plvData['t']
        STAm= plvData['STAm']
        fSTAm= plvData['fSTAm']
        STPm= plvData['STPm']
        SFCm= plvData['SFCm']
        PLVm= plvData['PLVm']
        
        STAe= plvData['STAe']
        fSTAe= plvData['fSTAe']
        STPe= plvData['STPe']
        SFCe= plvData['SFCe']
        PLVe= plvData['PLVe']
        
        fig1= plt.figure()
        ax= plt.gca()
        ax.plot(t, STAm, linewidth= 2, color= 'darkblue', marker= 'o', \
                     markerfacecolor= 'darkblue', markeredgecolor= 'none')
        ax.fill_between(t, STAm- STAe, STAm+ STAe, \
                     facecolor= 'cornflowerblue', alpha= 0.5, edgecolor= 'none', rasterized= True)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('Spike-triggered average (STA)')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('STA (uV)')        
        
        fig2= plt.figure()
        ax=fig2.add_subplot(221)
        ax.plot(f, fSTAm, linewidth= 2, color= 'darkblue', marker= '.', rasterized= True)
        ax.fill_between(f, fSTAm- fSTAe, fSTAm+ fSTAe, \
                     facecolor= 'cornflowerblue', alpha= 0.5, edgecolor= 'none', rasterized= True)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('fft of STA')
        
        ax=fig2.add_subplot(222)     
        ax.plot(f, STPm, linewidth= 2, color= 'darkblue', marker= '.', rasterized= True)
        ax.fill_between(f, STPm- STPe, STPm+ STPe, \
                     facecolor= 'cornflowerblue', alpha= 0.5, edgecolor= 'none', rasterized= True)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('STP')
        
        ax=fig2.add_subplot(223)
        ax.plot(f, SFCm, linewidth= 2, color= 'darkblue', marker= '.', rasterized= True)
        ax.fill_between(f, SFCm- SFCe, SFCm+ SFCe, \
                     facecolor= 'cornflowerblue', alpha= 0.5, edgecolor= 'none', rasterized= True)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('SFC')
        
        ax=fig2.add_subplot(224)
        ax.plot(f, PLVm, linewidth= 2, color= 'darkblue', marker= '.', rasterized= True)
        ax.fill_between(f, PLVm- PLVe, PLVm+ PLVe, \
                     facecolor= 'cornflowerblue', alpha= 0.5, edgecolor= 'none', rasterized= True)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('PLV')
        
        for ax in fig2.axes:
            ax.set_xlabel('Frequency (Hz)')
        
        fig2.suptitle('Frequency analysis of spike-triggered lfp metrics (bootstrap)')

        return fig1, fig2
        
    def spikePhase(self, phaseData):
        phBins= phaseData['phBins']
        phCount= phaseData['phCount']
        
        fig1= plt.figure()
        ax= plt.gca()
        ax.bar(np.append(phBins, phBins+ 360), np.append(phCount, phCount), \
               color= 'slateblue', width= np.diff(phBins).mean(), alpha= 0.6, align= 'center', rasterized= True)
        ax.plot(np.append(phBins, phBins+ 360), 0.5*np.max(phCount)*(np.cos(np.append(phBins, phBins+ 360)*np.pi/180)+ 1), \
                color= 'red', linewidth= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('LFP phase distribution (red= reference cosine line)')
        ax.set_xlabel('Degrees')
        ax.set_ylabel('Spike count')
                
        fig2= plt.figure()
        ax= plt.gca(polar= True)
        ax.bar(phBins*np.pi/180, phCount, width= 3*np.pi/180, \
               color= 'blue', alpha= 0.6, bottom= np.max(phaseData['phCount'])/2, rasterized= True)
        ax.plot([0, phaseData['meanTheta']], [0, 1.5*np.max(phCount)], linewidth= 3, color= 'red', marker= '.')
        plt.title('LFP phase distribution (red= mean direction)')
                
        fig3= plt.figure()
        ax= fig3.add_subplot(211)
        #cdict= {'blue': (0, 0, 1),
        #       'white': (0, 0, 0)}
        #c_map = mcol.LinearSegmentedColormap('my_colormap', cdict, 256)
        ax.pcolormesh(phaseData['rasterbins'], np.arange(0, phaseData['raster'].shape[0]), \
                      phaseData['raster'], cmap= plt.cm.binary, rasterized= True)
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.title('Phase raster')
        ax.set_ylabel('Time')
        
        ax= fig3.add_subplot(212)
        ax.bar(phBins, phCount, color= 'slateblue', \
               width= np.diff(phBins).mean(), alpha= 0.6, align= 'center', rasterized= True)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Phase(deg)')
        ax.set_ylabel('Spike count')
        
        return fig1, fig2, fig3
        
    def speed(self, speedData):
        ## Speed analysis        
        fig1= plt.figure()
        ax= plt.gca()
        ax.scatter(speedData['bins'], speedData['rate'], c= BLUE, zorder= 1)
        ax.plot(speedData['bins'], speedData['fitRate'], color= RED, linewidth= 1.5, zorder= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('Speed vs Spiking Rate')
        ax.set_xlabel('Speed (cm/sec)')
        ax.set_ylabel('Spikes/sec')
        
        return fig1
        
    def angVel(self, angVelData):
        ## Angular velocity analysis        
        fig1= plt.figure()
        ax= plt.gca()
        ax.scatter(angVelData['leftBins'], angVelData['leftRate'], c= BLUE, zorder= 1)
        ax.plot(angVelData['leftBins'], angVelData['leftFitRate'], color= RED, linewidth= 1.5, zorder= 2)
        ax.scatter(angVelData['rightBins'], angVelData['rightRate'], c= BLUE, zorder= 1)
        ax.plot(angVelData['rightBins'], angVelData['rightFitRate'], color= RED, linewidth= 1.5, zorder= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('Angular Velocity vs Spiking Rate')
        ax.set_xlabel('Angular velocity (deg/sec)')
        ax.set_ylabel('Spikes/sec')
        
        return fig1
        
    def MRA(self, mraData):
        fig1= plt.figure()
        ax= plt.gca()
        ax.bar(np.arange(6), mraData['meanRsq'], color= 'royalblue', align= 'center')
        ax.errorbar(np.arange(6), mraData['meanRsq'], fmt= 'ro', yerr= mraData['stdRsq'], ecolor= 'k', elinewidth= 3)
        ax.set_title('Multiple regression scores')
        ax.set_ylabel('$R^2$')
        plt.xticks(np.arange(6),mraData['order'])
        plt.autoscale(enable=True, axis='both', tight=True)
        
        return fig1 
    
    def hdRate(self, hdData, ax= None):
        if not ax:
            fig1= plt.figure()
            ax= plt.gca(polar= True)
        bins= np.append(hdData['bins'], hdData['bins'][0])
        rate= np.append(hdData['smoothRate'], hdData['smoothRate'][0])
        ax.plot(np.radians(bins), rate, color= BLUE)

        ax.set_title('Head directional firing rate')
        ax.set_rticks([hdData['hdRate'].max()])
        
        return ax
        
    def hdSpike(self, hdData, ax= None):
        if not ax:
            fig1= plt.figure()
            ax= plt.gca(polar= True)            
        ax.scatter(np.radians(hdData['scatter_bins']), hdData['scatter_radius'], \
                   s= 1, c= RED, alpha= 0.75, edgecolors= 'none', rasterized= True)
        ax.set_rticks([])
        ax.spines['polar'].set_visible(False)
        return ax
        
    def hdFiring(self, hdData):
        fig1= plt.figure()
        self.hdSpike(hdData, ax= plt.gca(polar= True))
        
        fig2= plt.figure()
        ax= self.hdRate(hdData, ax= plt.gca(polar= True))
        bins= np.appendd(hdData['bins'], hdData['bins'][0])
        predRate= np.appendd(hdData['hdPred'], hdData['hdPred'][0])
        ax.plot(np.radians(bins), predRate, color= 'green')
        ax.set_rticks([hdData['hdRate'].max(), hdData['hdPred'].max()])
        
        return fig1, fig2
                
    def hdRateCCW(self, hdData):
        fig1= plt.figure()
        ax= plt.gca(polar= True)
        ax.plot(np.radians(hdData['bins']), hdData['hdRateCW'], color=  BLUE)
        ax.plot(np.radians(hdData['bins']), hdData['hdRateCCW'], color= RED)
        ax.set_title('Counter/clockwise firing rate')
        ax.set_rticks([hdData['hdRateCW'].max(), hdData['hdRateCCW'].max()])                
        return fig1
        
    def hdShuffle(self, hdShuffleData):
        fig1= plt.figure()
        ax= plt.gca()
        ax.bar(hdShuffleData['raylZEdges'], hdShuffleData['raylZCount'], color= 'slateblue', alpha= 0.6, \
                        width= np.diff(hdShuffleData['raylZEdges']).mean(), rasterized= True)
        ax.plot([hdShuffleData['per95'], hdShuffleData['per95']], \
                [0, hdShuffleData['raylZCount'].max()], color= 'red', linewidth= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('Rayleigh Z distribution for shuffled spikes (red= no shuffle)')
        ax.set_xlabel('Rayleigh Z score')
        ax.set_ylabel('Count')
        
        return fig1
        
    def hdSpikeTimeLapse(self, hdData):
        keys= [key[1] for key in list(enumerate(hdData.keys()))]
                
        fig= []
        axes= []
        keys= list(hdData.keys())
        nkey= len(keys)
        nfig= int(np.ceil(nkey/4))
        for n in range(nfig):        
            f, ax= plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
            fig.append(f)
            axes.extend(list(ax.flatten()))

        kcount= 0
        for key in keys:            
            self.hdSpike(hdData[key], ax= axes[kcount])       
            axes[kcount].set_title(key)            
            kcount+=1            
        return fig 
        
    def hdRateTimeLapse(self, hdData):
        keys= [key[1] for key in list(enumerate(hdData.keys()))]
                
        fig= []
        axes= []
        keys= list(hdData.keys())
        nkey= len(keys)
        nfig= int(np.ceil(nkey/4))
        for n in range(nfig):        
            f, ax= plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
            fig.append(f)
            axes.extend(list(ax.flatten()))

        kcount= 0
        for key in keys:
            self.hdRate(hdData[key], ax= axes[kcount])            
            axes[kcount].set_title(key)            
            kcount+=1            
        return fig
        
    def hdTimeShift(self, hdShiftData):
        fig1= plt.figure()
        ax= plt.gca()
        ax.plot(hdShiftData['shiftTime'], hdShiftData['skaggs'], marker= 'o', markerfacecolor= RED, linewidth= 2)
        ax.set_xlabel('Shift time (ms)')
        ax.set_ylabel('Skaggs IC')
        ax.set_title('Skaggs IC of hd firing in shifted time of spiking events')
        
        fig2= plt.figure()
        ax= plt.gca()
        ax.plot(hdShiftData['shiftTime'], hdShiftData['peakRate'], marker= 'o', markerfacecolor= RED, linewidth= 2)
        ax.set_xlabel('Shift time (ms)')
        ax.set_ylabel('Peak firing rate (spikes/sec)')
        ax.set_title('Peak FR of hd firing in shifted time of spiking events')
                    
        fig3= plt.figure()
        ax= plt.gca()
        ax.scatter(hdShiftData['shiftTime'], hdShiftData['delta'], c=RED, zorder= 2)
        ax.plot(hdShiftData['shiftTime'], hdShiftData['deltaFit'], color= BLUE, linewidth= 1.5, zorder= 1)
        ax.set_xlabel('Shift time (ms)')
        ax.set_ylabel('Delta (degree)')
        ax.set_title('Delta of hd firing in shifted time of spiking events')
            
        return fig1, fig2, fig3
    
    def locSpike(self, placeData, ax= None):
        # spatial firing map
        if not ax:
            fig= plt.figure()
            ax= plt.gca()
            
        ax.plot(placeData['posX'], placeData['posY'], color= 'black', zorder= 1)
        ax.scatter(placeData['spikeLoc'][0], placeData['spikeLoc'][1], \
                   s= 2, marker= '.', color= RED, zorder= 2)
        ax.set_ylim([0, placeData['yedges'].max()])
        ax.set_xlim([0, placeData['xedges'].max()])
        ax.set_aspect('equal')
        ax.invert_yaxis()
#        plt.autoscale(enable=True, axis='both', tight=True)            
        
        return ax
    
    def locRate(self, placeData, ax= None):
        if not ax:
            fig= plt.figure()
            ax= plt.gca()
        clist= [(0.0, 0.0, 1.0),
                (0.0, 1.0, 0.5),
                (0.9, 1.0, 0.0),
                (1.0, 0.75, 0.0),
                (0.9, 0.0, 0.0)]
        c_map= mcol.ListedColormap(clist)
        ax.pcolormesh(placeData['xedges'], placeData['yedges'], np.ma.array(placeData['smoothMap'], \
                        mask= np.isnan(placeData['smoothMap'])), cmap= c_map, rasterized= True)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.autoscale(enable=True, axis='both', tight=True)
        
        return ax

    def locFiring(self, placeData):
        fig= plt.figure()
        self.locSpike(placeData, ax= fig.add_subplot(121))
        self.locRate(placeData, ax= fig.add_subplot(122))
        return fig
        
    def locSpikeTimeLapse(self, placeData):
        keys= [key[1] for key in list(enumerate(placeData.keys()))]
                
        fig= []
        axes= []
        keys= list(placeData.keys())
        nkey= len(keys)
        nfig= int(np.ceil(nkey/4))
        for n in range(nfig):        
            f, ax= plt.subplots(2, 2, sharex= 'col', sharey= 'row')
            fig.append(f)
            axes.extend(list(ax.flatten()))

        kcount= 0
        for key in keys:            
            self.locSpike(placeData[key], ax= axes[kcount])
            axes[kcount].set_title(key)            
            kcount+=1            
        return fig 
        
    def locRateTimeLapse(self, placeData):
        keys= [key[1] for key in list(enumerate(placeData.keys()))]
                
        fig= []
        axes= []
        keys= list(placeData.keys())
        nkey= len(keys)
        nfig= int(np.ceil(nkey/4))
        for n in range(nfig):        
            f, ax= plt.subplots(2, 2, sharex= 'col', sharey= 'row')
            fig.append(f)
            axes.extend(list(ax.flatten()))

        kcount= 0
        for key in keys:
            self.locRate(placeData[key], ax= axes[kcount])            
            axes[kcount].set_title(key)            
            kcount+=1            
        return fig

    def locShuffle(self, locShuffleData):    
        # Loactional shuffling analysis
        fig1= plt.figure()        
        ax= fig1.add_subplot(221)
        ax.bar(locShuffleData['skaggsEdges'][:-1], locShuffleData['skaggsCount'], color= 'slateblue', alpha= 0.6, \
                        width= np.diff(locShuffleData['skaggsEdges']).mean(), rasterized= True)
        ax.plot([locShuffleData['skaggs95'], locShuffleData['skaggs95']], \
                [0, locShuffleData['skaggsCount'].max()], color= 'red', linewidth= 2)
        ax.plot([locShuffleData['refSkaggs'], locShuffleData['refSkaggs']], \
                [0, locShuffleData['skaggsCount'].max()], color= 'green', linewidth= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Skaggs IC')
           
        ax= fig1.add_subplot(222)
        ax.bar(locShuffleData['sparsityEdges'][:-1], locShuffleData['sparsityCount'], color= 'slateblue', alpha= 0.6, \
                        width= np.diff(locShuffleData['sparsityEdges']).mean(), rasterized= True)
        ax.plot([locShuffleData['sparsity05'], locShuffleData['sparsity05']], \
                [0, locShuffleData['sparsityCount'].max()], color= 'red', linewidth= 2)
        ax.plot([locShuffleData['refSparsity'], locShuffleData['refSparsity']], \
                [0, locShuffleData['sparsityCount'].max()], color= 'green', linewidth= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Sparsity')
                
        ax= fig1.add_subplot(223)
        ax.bar(locShuffleData['coherenceEdges'][:-1], locShuffleData['coherenceCount'], color= 'slateblue', alpha= 0.6, \
                        width= np.diff(locShuffleData['coherenceEdges']).mean(), rasterized= True)
        ax.plot([locShuffleData['coherence95'], locShuffleData['coherence95']], \
                [0, locShuffleData['coherenceCount'].max()], color= 'red', linewidth= 2)
        ax.plot([locShuffleData['refCoherence'], locShuffleData['refCoherence']], \
                [0, locShuffleData['coherenceCount'].max()], color= 'green', linewidth= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_xlabel('Coherence')
        
        for ax in fig1.axes:
            ax.set_ylabel('Count')
        
        fig1.suptitle('Distribution of locational firing specificity indices')
        
        return fig1
        
    def locTimeShift(self, locShiftData):        
        ## Locational time shift analysis

        fig1= plt.figure()
        ax= plt.gca()
        ax.plot(locShiftData['shiftTime'], locShiftData['skaggs'], linewidth= 2, zorder= 1)
        ax.scatter(locShiftData['shiftTime'], locShiftData['skaggs'], marker= 'o', color= RED, zorder= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('Skaggs IC')
        ax.set_xlabel('Shift time (ms)')
        ax.set_title('Skaggs IC of place firing in shifted time of spiking events')
        
        fig2= plt.figure()
        ax= plt.gca()
        ax.plot(locShiftData['shiftTime'], locShiftData['sparsity'], linewidth= 2, zorder= 1)        
        ax.scatter(locShiftData['shiftTime'], locShiftData['sparsity'], marker= 'o', color= RED, zorder= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('Sparsity')
        ax.set_xlabel('Shift time (ms)')
        ax.set_title('Sparsity of place firing in shifted time of spiking events')
        
        fig3= plt.figure()
        ax = plt.gca()
        ax.plot(locShiftData['shiftTime'], locShiftData['coherence'], linewidth= 2, zorder= 1)    
        ax.scatter(locShiftData['shiftTime'], locShiftData['coherence'], marker= 'o', color= RED, zorder= 2)
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylabel('Coherence')
        ax.set_xlabel('Shift time (ms)')
        ax.set_title('Coherence of place firing in shifted time of spiking events')
        
#        for ax in fig1.axes:
#            ax.set_xlabel('shift time')
#        fig1.suptitle('Specifity indices in time shift')
        
        return fig1, fig2, fig3
    
    def locAutoCorr(self, locAutoData):
        # Locational firing map autocorrelation     
        clist= [(1.0, 1.0, 1.0), 
                (0.0, 0.0, 0.5),
                (0.0, 0.0, 1.0),
                (0.0, 0.5, 1.0),
                (0.0, 0.75, 1.0),
                (0.5, 1.0, 0.0),
                (0.9, 1.0, 0.0),
                (1.0, 0.75, 0.0),
                (1.0, 0.4, 0.0),
                (1.0, 0.0, 0.0),
                (0.5, 0.0, 0.0)]
                
        c_map= mcol.ListedColormap(clist)
        
        fig1= plt.figure()
        ax= fig1.gca()
        pc= ax.pcolormesh(locAutoData['xshift'], locAutoData['yshift'], np.ma.array(locAutoData['corrMap'], \
                        mask= np.isnan(locAutoData['corrMap'])), cmap= c_map, rasterized= True)
        ax.set_title('Spatial correlation of firing intesnity map)')
        ax.set_xlabel('X-lag')
        ax.set_ylabel('Y-lag')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.autoscale(enable=True, axis='both', tight=True)
        plt.colorbar(pc)
        
        return fig1
        
    def rotCorr(self, plotData):
    # Locationa firing map rotational analysis        
        fig1= plt.figure()
        ax= fig1.gca()
        ax.plot(plotData['rotAngle'], plotData['rotCorr'], linewidth= 2, zorder= 1)
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, 360])
        #plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('Rotational correlation of spatial firing map')
        ax.set_xlabel('Rotation angle')
        ax.set_ylabel('Pearson correlation')
        
        return fig1
        
    def distRate(self, distData):
        fig1= plt.figure()
        ax= plt.gca()
        ax.plot(distData['distBins'], distData['smoothRate'], marker= 'o', markerfacecolor= RED, linewidth= 2, label= 'Firing rate')
        if 'rateFit' in distData.keys():
            ax.plot(distData['distBins'], distData['rateFit'], 'go-', markerfacecolor= 'brown', linewidth= 2, label= 'Fitted rate')
        ax.set_title('Distance from border vs spike rate')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Rate (spikes/sec)')
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.legend(loc = 'lower right')
        
        return fig1
        
    def stairPlot(self, distData):
        perSteps= distData['perSteps']
        perDist= distData['perDist']
        stepsize= np.diff(perSteps).mean()
        fig1= plt.figure()
        ax= plt.gca()
        for i, step in enumerate(perSteps):
            ax.plot([step, step+ stepsize], [perDist[i], perDist[i]], color= 'b', linestyle= '--', marker= 'o', markerfacecolor= RED, linewidth= 2)
            if i>0: #perSteps.shape[0]:
                ax.plot([step, step], [perDist[i-1], perDist[i]], color= 'b', linestyle= '--', linewidth= 2)
        ax.set_xlabel('% firing rate (spikes/sec)')
        ax.set_ylabel('Mean distance (cm)')
        
        return fig1
        
    def border(self, borderData):    
        fig1= plt.figure()

        ax= fig1.add_subplot(211)
        ax.bar(borderData['distBins'], borderData['distCount'], color= 'slateblue', alpha= 0.6, \
                        width= 0.5*np.diff(borderData['distBins']).mean())
        ax.set_title('Histogram of taxicab distance of active pixels')
        ax.set_xlabel('Taxicab distance(cm)')
        ax.set_ylabel('Active pixel count')                
        
        ax= fig1.add_subplot(212)
        ax.bar(borderData['circBins'], borderData['angDistCount'], color= 'slateblue', alpha= 0.6, \
                        width= 0.5*np.diff(borderData['circBins']).mean())
        ax.set_title('Angular distance vs Active pixel count')
        ax.set_xlabel('Angular distance')
        ax.set_ylabel('Active pixel count')
        plt.autoscale(enable=True, axis='both', tight=True)
         
        fig2= plt.figure()
        ax= plt.gca()        
        pcm= ax.pcolormesh(borderData['cBinsInterp'], borderData['dBinsInterp'], \
                           borderData['circLinMap'], cmap='seismic', rasterized= True)
        ax.invert_yaxis()
        plt.autoscale(enable=True, axis='both', tight=True)
        ax.set_title('Histogram for angle vs distance from border of active pixels')
        ax.set_xlabel('Angular distance (Deg)')
        ax.set_ylabel('Taxicab distance (cm)')
        fig2.colorbar(pcm)
                
        fig3= self.distRate(borderData)
        fig4= self.stairPlot(borderData)
        
        return fig1, fig2, fig3, fig4
        
    def gradient(self, gradientData):
        fig1= self.distRate(gradientData)
        
        fig2= plt.figure()
        ax= plt.gca()
        ax.plot(gradientData['distBins'], gradientData['diffRate'], color= BLUE, marker= 'o', markerfacecolor= RED, linewidth= 2)
        ax.set_title('Differential firing rate (fitted)')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Differential rate (spikes/sec)')
        plt.autoscale(enable=True, axis='both', tight=True)
        
        fig3= self.stairPlot(gradientData)
        
        return fig1, fig2, fig3
        
    def grid(self, gridData):
        fig1= self.locAutoCorr(gridData)
        ax= fig1.axes[0]
        xmax= gridData['xmax']
        ymax= gridData['ymax']
        xshift= gridData['xshift']
        
        ax.scatter(xmax, ymax, c= 'black', marker= 's', zorder= 2)
        for i in range(xmax.size):
            if i< xmax.size-1:
                ax.plot([xmax[i], xmax[i+1]], [ymax[i], ymax[i+1]], 'k', linewidth= 2)
            else:
                ax.plot([xmax[i], xmax[0]], [ymax[i], ymax[0]], 'k', linewidth= 2)
        ax.plot(xshift[xshift>=0], np.zeros(find(xshift>=0).size), 'k--', linewidth = 2)
        ax.plot(xshift[xshift>=0], xshift[xshift>=0]*ymax[0]/ xmax[0], 'k--', linewidth = 2)
        ax.set_title('Grid cell analysis')
        ax.set_xlim([gridData['xshift'].min(), gridData['xshift'].max()])
        ax.set_ylim([gridData['yshift'].min(), gridData['yshift'].max()])
        
        fig2= None
        if 'rotAngle' in gridData.keys() and 'rotCorr' in gridData.keys():
            fig2= self.rotCorr(gridData)
            ax= fig2.axes[0]
            rmax= gridData['rotCorr'].max()
            rmin= gridData['rotCorr'].min()
            for i, th in enumerate(gridData['anglemax']):
                ax.plot([th, th], [rmin, rmax], 'r--', linewidth= 1)
            for i, th in enumerate(gridData['anglemin']):
                ax.plot([th, th], [rmin, rmax], 'g--', linewidth= 1)
                
            ax.autoscale(enable=True, axis='both', tight=True)
            ax.set_title('Rotational correlation of autocorrelation map')
        if fig2:
            return fig1, fig2
        else:
            return fig1