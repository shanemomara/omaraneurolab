# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 22:34:02 2017

@author: Raju
"""
from PyQt5 import QtCore

import nc_ext, nc_data, nc_plotter
from imp import reload
reload(nc_ext)
reload(nc_data)
reload(nc_plotter)
from nc_data import NData, Nhdf, NClust
from nc_ext import Configuration, NLog
from nc_plotter import NPlotter

import logging, inspect
from collections import OrderedDict as oDict

import matplotlib.pyplot as plt
import matplotlib.figure

import numpy as np
import pandas as pd

import os.path

from matplotlib.backends.backend_pdf import PdfPages

class NeuroChaT(QtCore.QThread):
    finished= QtCore.pyqtSignal()
    def __init__(self, config= Configuration(), data= NData(), parent= None):
        super().__init__(parent)
        self.ndata= data
        self.__config= config
        self.nplot= NPlotter()
        self.log= NLog()
        self.hdf= Nhdf()
    def reset(self):
        self.__count= 0
        self.nwbFiles= []
        self.graphicsFiles= []
        self.cellid= []
        self.results= []
        self.saveToFile= False
        self._pdfFile= None
        if not self.getGraphicFormat():
            self.setGraphicFormat('PDF')
        self.nplot.setBackend(self.getGraphicFormat())
        
    def getOutputFiles(self):
        opFiles= {'Graphics Files': self.graphicsFiles,  
                  'NWB Files': self.nwbFiles}
        opFiles= pd.DataFrame.from_dict(opFiles)
        opFiles.index= self.cellid
        opFiles= opFiles[['Graphics Files', 'NWB Files']]
        
        return opFiles

    def updateResults(self, _results):
        self.results.append(_results.copy()) # without copy, list contains a reference to the original dictionary, and old results are replaced by the new one
    def getResults(self):
        keys= []
        for d in self.results:
            [keys.append(k) for k in list(d.keys()) if k not in keys]        
        results= pd.DataFrame(self.results, columns= keys)
        results.index= self.cellid
        return results
        
    def openpdf(self, filename= None):
        if filename is not None:
            words= filename.split(os.sep)
            directory= os.sep.join(words[:-1])
            if os.path.exists(directory):
                self._pdfFile= filename # Current PDF file being handled
                self.pdf= PdfPages(self._pdfFile)
                self.saveToFile= True
            else:
                self.saveToFile= False
                self._pdfFile= None
                logging.error('Cannot create PDF, file path is invalid')
        else:
            logging.error('No valid PDf file is specified')

    def closepdf(self):
        if self._pdfFile is not None:
            self.pdf.close()
            logging.info('Output graphics saved to '+ self._pdfFile)
        else:
            logging.warning('No PDF file for the NPlotter instance')
    
    def closefig(self, fig):
        if isinstance(fig, tuple) or isinstance(fig, list):
            for f in fig:
                if isinstance(f, matplotlib.figure.Figure):
                    if self.saveToFile:
                        self.pdf.savefig(f, dpi= 400)
#                        self.pdf.savefig(f)
                    plt.close(f)
                else:
                    logging.error('Invalid matplotlib.figure instance')                
        elif isinstance(fig, matplotlib.figure.Figure):
            if self.saveToFile:   
                self.pdf.savefig(fig)
            plt.close(fig)
        else:
            logging.error('Invalid matplotlib.figure instance')
        
    def run(self):
        self.reset()
        verified= True
    # Deduce the configuration
    # Same filename for spike, lfp and spatial will go for NWB
        if not any(self.getAnalysis('all')):
            verified = False
            logging.error('No analysis method has been selected')
        else:
            # Could take this to mode, but replication would occur for each data format
            mode_id= self.getAnalysisMode()[1]
            if (mode_id== 0 or mode_id== 1) and \
                (self.getDataFormat()== 'Axona' or self.getDataFormat()== 'Neuralynx'):
                if not os.path.isfile(self.getSpikeFile()):
                    verified = False
                    logging.error('Spike file does not exist')

                if not os.path.isfile(self.getSpatialFile()):
                    logging.warning('Position file does not exist')
            elif mode_id== 2:
                if not os.path.isfile(self.getExcelFile()):
                    verified = False
                    logging.error('Excel file does not exist')
                    
        if verified:
            self.__count= 0
            self.ndata.setDataFormat(self.getDataFormat())
#            mode= getattr(self, 'mode_'+ self.getDataFormat())
#            mode()
            self.mode()      
        self.finished.emit()

    def mode(self):
        info= {'spat': [], 'spike': [], 'unit': [], 'lfp': [], 'nwb': [], 'graphics': [], 'cellid': []}
        mode_id= self.getAnalysisMode()[1]
        if mode_id== 0 or mode_id== 1: # All the cells in the same tetrode will use the same lfp channel
            spatialFile= self.getSpatialFile()            
            spikeFile =self.getSpikeFile()
            lfpFile= self.getLfpFile()
#            if self.getDataFormat()== 'Axona':
#                lfpID= '.eeg'+ str(self.getLfpChan()) if self.getLfpChan()>0 else '.eeg'    
#                lfpFile= ''.join(spikeFile.split('.')[:-1])+ lfpID
#            elif self.getDataFormat()== 'Neuralynx':
#                lfpFile= os.sep.join(spikeFile.split(os.sep)[:-1].append(self.getLfpChan()+ '.ncs'))
#            elif self.getDataFormat()== 'HDF5':
#                pass
#                # Will implement later
                            
            self.ndata.setSpikeFile(spikeFile)
            self.ndata.loadSpike()

            units= [self.getUnitNo()] if mode_id==0 else self.ndata.getUnitList()
            if not units:
                logging.error('No unit found in analysis')
            else:
                for unit_no in units:
                    info['spat'].append(spatialFile)
                    info['spike'].append(spikeFile)
                    info['unit'].append(unit_no)
                    info['lfp'].append(lfpFile)

        elif mode_id== 2:
            excel_file= self.getExcelFile()
            if os.path.exists(excel_file):
                excel_info = pd.read_excel(excel_file)
                for row in excel_info.itertuples():
                    spikeFile= row[1]+ os.sep+ row[3]
                    unit_no= int(row[4])
                    lfpID= row[5]
                    
                    if self.getDataFormat()== 'Axona':
                        spatialFile= row[1]+ os.sep+ row[2]+ '.txt'
                        lfpFile= ''.join(spikeFile.split('.')[:-1])+ '.'+ lfpID
                        
                    elif self.getDataFormat()== 'Neuralynx':
                        spatialFile= row[1] + os.sep+ row[2]+ '.nvt'
                        lfpFile= row[1]+ os.sep+ lfpID+ '.ncs'
                        
                    elif self.getDataFormat()== 'NWB':
                        # excel list: directory| hdf5 file name w/o extension| spike group| unit_no| lfp group
                        hdf_name= row[1] + os.sep+ row[2]+ '.hdf5'
                        spikeFile= hdf_name+ '/processing/Shank'+ '/'+ row[3] 
                        spatialFile= hdf_name+ '+/processing/Behavioural/Position'
                        lfpFile= hdf_name+ '+/processing/Neural Continuous/LFP'+ '/' + lfpID
                                     
                    info['spat'].append(spatialFile)
                    info['spike'].append(spikeFile)
                    info['unit'].append(unit_no)
                    info['lfp'].append(lfpFile)
        
        if info['unit']:
            for i, unit_no in enumerate(info['unit']):
                logging.info('Starting a new unit...')
#                try:
                self.ndata.setSpatialFile(info['spat'][i])
                self.ndata.setSpikeFile(info['spike'][i])
                self.ndata.setLfpFile(info['lfp'][i])
                self.ndata.load()
                self.ndata.setUnitNo(info['unit'][i])
                
                self.ndata.resetResults()
                
                cell_id= self.hdf.resolve_analysis_path(spike= self.ndata.spike,  lfp= self.ndata.lfp)
                nwb_name= self.hdf.resolve_hdfname(data= self.ndata.spike)
                pdf_name= ''.join(nwb_name.split('.')[:-1])+ '_'+ \
                            cell_id+ '.'+ self.getGraphicFormat()
                
                info['nwb'].append(nwb_name)        
                info['cellid'].append(cell_id)
                info['graphics'].append(pdf_name)
                
                self.openpdf(pdf_name)
                
                fig= plt.figure()
                ax= fig.add_subplot(111)
                ax.text(0.1, 0.6, 'Cell ID= '+ cell_id+ '\n'+ \
                        'HDF5 file= '+ nwb_name.split(os.sep)[-1]+ '\n'+ \
                        'Graphics file= '+ pdf_name.split(os.sep)[-1], \
                        horizontalalignment= 'left', \
                        verticalalignment= 'center',\
                        transform= ax.transAxes,
                        clip_on= True)
                ax.set_axis_off()
                self.closefig(fig)
                
                # Set and open hdf5 file for saving graph data within self.execute()
                self.hdf.setFilename(nwb_name)
                if '/analysis/'+ cell_id in self.hdf.f:
                    del self.hdf.f['/analysis/'+ cell_id]
#                    self.hdf.f.require_group('/analysis/'+ cell_id)
#                self.hdf.delete_group_data(path= )
                
                self.execute(name= cell_id)

                self.closepdf()
                
                _results= self.ndata.getResults()
                
                self.updateResults(_results)
                self.hdf.save_dict_recursive(path= '/analysis/'+ cell_id+ '/', name= 'results', data= _results)
                
                self.hdf.close()
#                df_results= pd.DataFrame.from_dict(_results, orient= 'index')
#                df_results.to_hdf(nwb_name, '/analysis/'+ cell_id, mode= 'a', format= 'table')
                
                self.ndata.save_to_hdf5() # Saving data to hdf file
                
                self.__count+= 1
                logging.info('Units already analyzed= ' + str(self.__count))
#                except:
#                    logging.info('Erro in analyzing '+ info['unit'][i])
                
        logging.info('Total cell analyzed: '+ str(self.__count))
        self.cellid= info['cellid']
        self.nwbFiles= info['nwb']
        self.graphicsFiles= info['graphics']
        
    def execute(self, name= None):
        try:
            logging.info('Calculating environmental border...')
            params= {'locPixelSize' : 3,
                     'locChopBound' : 5}
            params.update(self.getParams(list(params.keys())))
            
            self.setBorder(self.calcBorder())
            
        except:
            logging.warning('Border calculation was not properly completed!')

        if self.getAnalysis('waveProperty'):
            logging.info('Assessing waveform properties...')
            try:
                graphData= self.waveProperty() # gd= graphData
                fig= self.nplot.waveProperty(graphData, [int (self.getTotalChannels()/2), 2])
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/waveProperty/', graphData= graphData)
            except:
                logging.error('Error in assessing waveform property')
            
        if self.getAnalysis('isi'):
            # ISI analysis
            logging.info('Calculating inter-spike interval distribution...')
            try:
                params= {'isiBin': 2,
                         'isiLength': 350}                
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.isi(bins= int(params['isiLength']/params['isiBin']), \
                                    bound= [0, params['isiLength']])
                fig= self.nplot.isi(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/isi/', graphData= graphData)
            except:
                logging.error('Error in assessing interspike interval distribution')            

        if self.getAnalysis('isiCorr'):
            ##Autocorr 1000ms
            logging.info('Calculating inter-spike interval autocorrelation histogram...')
            try:
                params= {'isiCorrBinLong': 2,
                         'isiCorrLenLong': 350,
                         'isiCorrBinShort': 1,
                         'isiCorrLenShort': 10}
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.isiCorr(bins= params['isiCorrBinLong'], \
                                        bound= [-params['isiCorrLenLong'], params['isiCorrLenLong']])
                fig= self.nplot.isiCorr(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/isiCorrLong/', graphData= graphData)
                # Autocorr 10ms    
                graphData= self.isiCorr(bins= params['isiCorrBinShort'], \
                                        bound= [-params['isiCorrLenShort'], params['isiCorrLenShort']])
                fig= self.nplot.isiCorr(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/isiCorrShort/', graphData= graphData)
            except:
                logging.error('Error in assessing ISI autocorrelation')
                
        if self.getAnalysis('thetaCell'):
            ## Theta-Index analysis
            logging.info('Estimating theta-modulation index...')
            try:
                params= {'isiCorrBinLong' : 2,
                         'isiCorrLenLong' : 350,
                         'thetaCellFreqMax' : 14.0,
                         'thetaCellFreqMin' : 4.0,
                         'thetaCellFreqStart' : 6.0,
                         'thetaCellTau1Max' : 5.0,
                         'thetaCellTau1Start' : 0.1,
                         'thetaCellTau2Max' : 0.05,
                         'thetaCellTau2Start' : 0.05}
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.thetaIndex( start= [params['thetaCellFreqStart'], params['thetaCellTau1Start'], params['thetaCellTau2Start']], \
                               lower= [params['thetaCellFreqMin'], 0, 0], \
                               upper= [params['thetaCellFreqMax'], params['thetaCellTau1Max'], params['thetaCellTau2Max']], \
                               bins= params['isiCorrBinLong'], \
                               bound= [-params['isiCorrLenLong'], params['isiCorrLenLong']])
                fig= self.nplot.thetaCell(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/thetaCell/', graphData= graphData)
            except:
                logging.error('Error in theta-index analysis')
                                         
        if self.getAnalysis('thetaSkipCell'):
            logging.info('Estimating theta-skipping index...')
            try:
                params= {'isiCorrBinLong' : 2,
                         'isiCorrLenLong' : 350,
                         'thetaCellFreqMax' : 14.0,
                         'thetaCellFreqMin' : 4.0,
                         'thetaCellFreqStart' : 6.0,
                         'thetaCellTau1Max' : 5.0,
                         'thetaCellTau1Start' : 0.1,
                         'thetaCellTau2Max' : 0.05,
                         'thetaCellTau2Start' : 0.05}
                params.update(self.getParams(list(params.keys())))                           
                
                graphData= self.thetaSkipIndex( start= [params['thetaCellFreqStart'], params['thetaCellTau1Start'], params['thetaCellTau2Start']], \
                               lower= [params['thetaCellFreqMin'], 0, 0], \
                               upper= [params['thetaCellFreqMax'], params['thetaCellTau1Max'], params['thetaCellTau2Max']], \
                               bins= params['isiCorrBinLong'], \
                               bound= [-params['isiCorrLenLong'], params['isiCorrLenLong']])
                fig= self.nplot.thetaCell(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/thetaSkipCell/', graphData= graphData)
            except:
                logging.error('Error in theta-skipping cell index analysis')                    
            
        if self.getAnalysis('burst'):
            ### Burst analysis
            logging.info('Analyzing bursting property...')
            try:
                params= {'burstThresh' : 5, 
                         'ibiThresh' : 50}
                params.update(self.getParams(list(params.keys())))
                
                self.burst(**params)
            except:
                logging.error('Error in analysing bursting property')
            
        if self.getAnalysis('speed'):
            ## Speed analysis
            logging.info('Calculating spike-rate vs running speed...')
            try:
                params= {'speedBin' : 1,
                         'speedMax' : 40,
                         'speedMin' : 0,
                         'speedRateKernLen' : 3}
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.speed(range= [params['speedMin'], params['speedMax']], \
                                      binsize= params['speedBin'], update= True)
                fig= self.nplot.speed(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/speed/', graphData= graphData)
            except:
                logging.error('Error in analysis of spike rate vs speed')            
        
        if self.getAnalysis('angVel'):
            ## Angular velocity analysis
            logging.info('Calculating spike-rate vs angular head velocity...')
            try:
                params={'angVelBin' : 10, 
                        'angVelCutoff' : 10,
                        'angVelKernLen' : 3,
                        'angVelMax' : 200,
                        'angVelMin' : -200}
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.angularVelocity(range= [params['angVelMin'], params['angVelMax']], \
                                    binsize= params['angVelBin'], cutoff= params['angVelCutoff'], update= True)
                fig= self.nplot.angVel(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/angVel/', graphData= graphData)
            except:
                logging.error('Error in analysis of spike rate vs angular velocity')
            
        if self.getAnalysis('hdRate'):
            logging.info('Assessing head-directional tuning...')
            try:
                params= {'hdBin' : 5,
                         'locPixelSize' : 3,
                         'hdAngVelCutoff' : 30,
                         'hdRateKernLen' : 5}
                params.update(self.getParams(list(params.keys())))

                hdData= self.hdRate(binsize= params['hdBin'], \
                        filter= ['b', params['hdRateKernLen']],\
                        pixel= params['locPixelSize'], update= True)
                fig= self.nplot.hdFiring(hdData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/hdRate/', graphData= hdData)
                
                hdData= self.hdRateCCW(binsize= params['hdBin'], \
                        filter= ['b', params['hdRateKernLen']],\
                        thresh= params['hdAngVelCutoff'],\
                        pixel= params['locPixelSize'], update= True)
                fig= self.nplot.hdRateCCW(hdData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/hdRateCCW/', graphData= hdData)
                
            except:
                logging.error('Error in analysis of spike rate vs head direction')            
            
        if self.getAnalysis('hdShuffle'):
            logging.info('Shuffling analysis of head-directional tuning...')
            try:
                params= {'hdShuffleLimit' : 0,
                         'hdShuffleNoBins' : 100,
                         'hdShuffleTotal' : 500}
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.hdShuffle(bins= params['hdShuffleNoBins'], \
                                          nshuff= params['hdShuffleTotal'], limit= params['hdShuffleLimit'])
                fig= self.nplot.hdShuffle(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/hdShuffle/', graphData= graphData)
            except:
                logging.error('Error in head directional shuffling analysis')
            
        if self.getAnalysis('hdTimeLapse'):
            logging.info('Time-lapsed head-directional tuning...')
            try:                
                graphData= self.hdTimeLapse()
                
                fig= self.nplot.hdSpikeTimeLapse(graphData)
                self.closefig(fig)
                
                fig= self.nplot.hdRateTimeLapse(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/hdTimeLapse/', graphData= graphData)
                
            except:
                logging.error('Error in locational time-lapse analysis')
            
        if self.getAnalysis('hdTimeShift'):
            logging.info('Time-shift analysis of head-directional tuning...')
            try:
                params= {'hdShiftMax' : 10,
                         'hdShiftMin' : -10,
                         'hdShiftStep' : 1}
                params.update(self.getParams(list(params.keys())))
                
                hdData= self.hdShift(shiftInd= np.arange(params['hdShiftMin'], \
                                                        params['hdShiftMax']+ params['hdShiftStep'], \
                                                        params['hdShiftStep']))
                fig= self.nplot.hdTimeShift(hdData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/hdTimeShift/', graphData= hdData)
            except:
                logging.error('Error in head directional time-shift analysis')

        if self.getAnalysis('locRate'):
            logging.info('Assessing of locational tuning...')
            try:
                params= {'locPixelSize' : 3,
                         'locChopBound' : 5,
                         'locRateFilter' : 'Box',
                         'locRateKernLen' : 5}
                params.update(self.getParams(list(params.keys())))
                
                if params['locRateFilter']== 'Gaussian':
                    filttype= 'g'
                else:
                    filttype= 'b'
                
                placeData= self.place(pixel= params['locPixelSize'], \
                              chopBound= params['locChopBound'],\
                              filter= [filttype, params['locRateKernLen']],\
                              brAdjust= True, update= True)
                fig= self.nplot.locFiring(placeData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/locRate/', graphData= placeData)
                
            except:
                logging.error('Error in analysis of location firing rate')
            
        if self.getAnalysis('locShuffle'):
            logging.info('Shuffling analysis of locational tuning...')
            try:
                params= {'locShuffleLimit' : 0,
                         'locShuffleNoBins' : 100,
                         'locShuffleTotal' : 500,
                         'locPixelSize' : 3,
                         'locChopBound' : 5,
                         'locRateFilter' : 'Box',
                         'locRateKernLen' : 5}
                params.update(self.getParams(list(params.keys())))
                
                if params['locRateFilter']== 'Gaussian':
                    filttype= 'g'
                else:
                    filttype= 'b'
                
                placeData= self.locShuffle(bins= params['locShuffleNoBins'], \
                                          nshuff= params['locShuffleTotal'], \
                                          limit= params['locShuffleLimit'], \
                                          pixel= params['locPixelSize'], \
                                          chopBound= params['locChopBound'], \
                                          filter= [filttype, params['locRateKernLen']],\
                                          brAdjust= True, update= False)
                fig= self.nplot.locShuffle(placeData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/locShuffle/', graphData= placeData)
            except:
                logging.error('Error in locational shiffling analysis')
            
        if self.getAnalysis('locTimeLapse'):
            logging.info('Time-lapse analysis of locational tuning...')
            try:
                params= {'locPixelSize' : 3,
                         'locChopBound' : 5,
                         'locRateFilter' : 'Box',
                         'locRateKernLen' : 5}
                params.update(self.getParams(list(params.keys())))
                
                if params['locRateFilter']== 'Gaussian':
                    filttype= 'g'
                else:
                    filttype= 'b'
                
                graphData= self.locTimeLapse(pixel= params['locPixelSize'], \
                              chopBound= params['locChopBound'],\
                              filter= [filttype, params['locRateKernLen']],\
                              brAdjust= True)
                
                fig= self.nplot.locSpikeTimeLapse(graphData)
                self.closefig(fig)
                
                fig= self.nplot.locRateTimeLapse(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/locTimeLapse/', graphData= graphData)
            except:
                logging.error('Error in locational time-lapse analysis')
            
        if self.getAnalysis('locTimeShift'):
            logging.info('Time-shift analysis of locational tuning...')
            try:
                params= {'locShiftMax' : 10,
                         'locShiftMin' : -10,
                         'locShiftStep' : 1,
                         'locPixelSize' : 3,
                         'locChopBound' : 5,
                         'locRateFilter' : 'Box',
                         'locRateKernLen' : 5}
                params.update(self.getParams(list(params.keys())))
                
                if params['locRateFilter']== 'Gaussian':
                    filttype= 'g'
                else:
                    filttype= 'b'
                
                plotData= self.locShift(shiftInd= np.arange(params['locShiftMin'], \
                                        params['locShiftMax']+ params['locShiftStep'], \
                                        params['locShiftStep']), \
                                        pixel= params['locPixelSize'], \
                                        chopBound= params['locChopBound'], \
                                        filter= [filttype, params['locRateKernLen']],\
                                        brAdjust= True, update= False)
                fig= self.nplot.locTimeShift(plotData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/locTimeShift/', graphData= plotData)
            except:
                logging.error('Error in locational time-shift analysis')            
            
        if self.getAnalysis('spatialCorr'):
            logging.info('Spatial and rotational correlation of locational tuning...')
            try:
                params=  {'locPixelSize' : 3,
                          'locChopBound' : 5,
                          'rotCorrBin' : 3,
                          'spatialCorrFilter' : 'Box',
                          'spatialCorrKernLen' : 5,
                          'spatialCorrMinObs' : 20}
                params.update(self.getParams(list(params.keys())))

                if params['spatialCorrFilter']== 'Gaussian':
                    filttype= 'g'
                else:
                    filttype= 'b'
                                    
                plotData= self.locAutoCorr(pixel= params['locPixelSize'], \
                              chopBound= params['locChopBound'],\
                              filter= [filttype, params['spatialCorrKernLen']],\
                              minPixel= params['spatialCorrMinObs'], brAdjust= True)
                fig= self.nplot.locAutoCorr(plotData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/spatialCorr/', graphData= plotData)
            
                plotData= self.locRotCorr(binsize= params['rotCorrBin'], \
                                          pixel= params['locPixelSize'], \
                                          chopBound= params['locChopBound'],\
                                          filter= [filttype, params['spatialCorrKernLen']],\
                                          minPixel= params['spatialCorrMinObs'], brAdjust= True)
                fig= self.nplot.rotCorr(plotData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/spatialCorr/', graphData= plotData)
                
            except:
                logging.error('Error in assessing spatial autocorrelation')
            
        if self.getAnalysis('grid'):
            logging.info('Assessing gridness...')
            try:
                params=  {'gridAngBin' : 3,
                          'gridAngTol' : 2,
                          'locPixelSize' : 3,
                          'locChopBound' : 5,
                          'rotCorrBin' : 3,
                          'spatialCorrFilter' : 'Box',
                          'spatialCorrKernLen' : 5,
                          'spatialCorrMinObs' : 20}
                params.update(self.getParams(list(params.keys())))
                
                if params['spatialCorrFilter']== 'Gaussian':
                    filttype= 'g'
                else:
                    filttype= 'b'
                    
                graphData= self.grid(angtol= params['gridAngTol'],\
                                     binsize= params['gridAngBin'], \
                                     pixel= params['locPixelSize'], \
                                     chopBound= params['locChopBound'],\
                                     filter= [filttype, params['spatialCorrKernLen']],\
                                     minPixel= params['spatialCorrMinObs'], \
                                     brAdjust= True) # Add other paramaters
                fig= self.nplot.grid(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/grid/', graphData= graphData)
                
            except:
                logging.error('Error in grid cell analysis')
            
        if self.getAnalysis('border'):
            logging.info('Estimating tuning to border...')
            try:
                params= {'borderAngBin' : 3,
                         'borderFiringThresh' : 0.1,
                         'borderStairSteps' : 5,
                         'burstThresh' : 5,
                         'locPixelSize' : 3,
                         'locChopBound' : 5,
                         'locRateFilter' : 'Box',
                         'locRateKernLen' : 5}
                params.update(self.getParams(list(params.keys())))
                
                if params['locRateFilter']== 'Gaussian':
                    filttype= 'g'
                else:
                    filttype= 'b'
                
                graphData= self.border(update= True, thresh= params['borderFiringThresh'], \
                                       cbinsize= params['borderAngBin'], \
                                       nstep= params['borderStairSteps'], \
                                       pixel= params['locPixelSize'], \
                                       chopBound= params['locChopBound'],\
                                       filter= [filttype, params['locRateKernLen']],\
                                       brAdjust= True)
                
                fig= self.nplot.border(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/border/', graphData= graphData)
            except:
                logging.error('Error in border cell analysis')
            
        if self.getAnalysis('gradient'):
            logging.info('Calculating gradient-cell properties...')
            try:
                params= {'gradAsympLim' : 0.25,
                         'gradDisplaceLim' : 0.25,
                         'gradGrowthRateLim' : 0.5,
                         'borderAngBin' : 3,
                         'borderFiringThresh' : 0.1,
                         'borderStairSteps' : 5,
                         'burstThresh' : 5,
                         'locPixelSize' : 3,
                         'locChopBound' : 5,
                         'locRateFilter' : 'Box',
                         'locRateKernLen' : 5}
                params.update(self.getParams(list(params.keys())))
                
                if params['locRateFilter']== 'Gaussian':
                    filttype= 'g'
                else:
                    filttype= 'b'
                    
                graphData= self.gradient(alim= params['gradAsympLim'], \
                                         blim= params['gradDisplaceLim'], \
                                         clim= params['gradGrowthRateLim'], \
                                         pixel= params['locPixelSize'], \
                                         chopBound= params['locChopBound'],\
                                         filter= [filttype, params['locRateKernLen']],\
                                         brAdjust= True)
                fig= self.nplot.gradient(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/gradient/', graphData= graphData)
            except:
                logging.error('Error in gradient cell analysis')
            
        if self.getAnalysis('mra'):
            logging.info('Multiple-regression analysis...')
            try:
                params= {'mraEpisode' : 120,
                         'mraInterval' : 0.1,
                         'mraNoRep' : 1000}
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.MRA(nrep= params['mraNoRep'], \
                                    episode= params['mraEpisode'], \
                                    subsampInterv= params['mraInterval'])
                
                fig= self.nplot.MRA(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/mra/', graphData= graphData)
            except:
                logging.error('Error in multiple-regression analysis')
            
        if self.getAnalysis('interDepend'):
            # No plot
            logging.info('Assessing dependence of variables to...')
            try:
                self.interdependence(pixel = 3, hdbinsize= 5, spbinsize= 1, sprange= [0, 40], \
                                    abinsize= 10, angvelrange= [-500, 500])
            except:
                logging.error('Error in interdependence analysis')
                                                
        if self.getAnalysis('lfpSpectrum'):
            try:
                params= {'lfpPreFiltHighCut' : 40,
                         'lfpPreFiltLowCut' : 1.5,
                         'lfpPreFiltOrder' : 10,
                         'lfpPwelchFreqMax' : 40,
                         'lfpPwelchNfft' : 1024,
                         'lfpPwelchOverlap' : 1.0,
                         'lfpPwelchSegSize' : 2.0,
                         'lfpStftFreqMax' : 40,
                         'lfpStftNfft' : 1024,
                         'lfpStftOverlap' : 1.0,
                         'lfpStftSegSize' : 2.0}
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.spectrum(window= params['lfpPwelchSegSize'],\
                                         noverlap= params['lfpPwelchOverlap'], \
                                         nfft= params['lfpPwelchNfft'],\
                                         ptype= 'psd', prefilt= True, \
                                         filtset= [params['lfpPreFiltOrder'], \
                                                   params['lfpPreFiltLowCut'], \
                                                   params['lfpPreFiltHighCut'], 'bandpass'], \
                                         fmax= params['lfpPwelchFreqMax'],\
                                         db= False, tr= False)
                fig= self.nplot.lfpSpectrum(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/lfpSpectrum/', graphData= graphData)
                
                graphData= self.spectrum(window= params['lfpStftSegSize'],\
                                         noverlap= params['lfpStftOverlap'],\
                                         nfft= params['lfpStftNfft'],\
                                         ptype= 'psd', prefilt= True,\
                                         filtset= [params['lfpPreFiltOrder'], \
                                                   params['lfpPreFiltLowCut'], \
                                                   params['lfpPreFiltHighCut'], 'bandpass'], \
                                         fmax= params['lfpStftFreqMax'],\
                                         db= True, tr= True)
                fig= self.nplot.lfpSpectrum_tr(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/lfpSpectrumTR/', graphData= graphData)
                
            except:
                logging.error('Error in analyzing lfp spectrum')
            
        if self.getAnalysis('spikePhase'):
            ### Analysis of Phase distribution
            logging.info('Analysing distribution of spike-phase in lfp...')
            try:
                params= {'phaseAmpThresh' : 0.15,
                         'phaseBin' : 5,
                         'phaseFreqMax' : 12.0,
                         'phaseFreqMin' : 6.0,
                         'phaseLockFreqMax' : 40,
                         'phaseLockNfft' : 1024,
                         'phasePowerThresh' : 0.1,
                         'phaseRasterBin' : 2,
                         'lfpPreFiltHighCut' : 40,
                         'lfpPreFiltLowCut' : 1.5,
                         'lfpPreFiltOrder' : 10}
                
                params.update(self.getParams(list(params.keys())))
                
                graphData= self.phaseDist(binsize= params['phaseBin'], \
                                          rbinsize= params['phaseRasterBin'],\
                                          fwin= [params['phaseFreqMin'], params['phaseFreqMax']],\
                                          pratio= params['phasePowerThresh'],\
                                          aratio= params['phaseAmpThresh'], 
                                          filtset= [params['lfpPreFiltOrder'], \
                                                   params['lfpPreFiltLowCut'], \
                                                   params['lfpPreFiltHighCut'], 'bandpass'])
                fig= self.nplot.spikePhase(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/spikePhase/', graphData= graphData)
            
            except:
                logging.error('Error in assessing spike-phase distribution')
            
        if self.getAnalysis('phaseLock'):
            # PLV with mode= None (all events or spikes)
            logging.info('Analysis of Phase-locking value and spike-filed coherence...')
            try:
                params= {'phaseLockFreqMax' : 40,
                         'phaseLockNfft' : 1024,
                         'phaseLockWinLow' : -0.4,
                         'phaseLockWinUp' : 0.4,
                         'phasePowerThresh' : 0.1,
                         'phaseRasterBin' : 2}
                params.update(self.getParams(list(params.keys())))

                reparam= {'window' : [params['phaseLockWinLow'], params['phaseLockWinUp']],
                          'nfft': params['phaseLockNfft'],
                          'fwin': [2, params['phaseLockFreqMax']],
                          'nsample': 2000,
                          'slide': 25,
                          'nrep': 500,
                          'mode': 'tr'}
                                          
                graphData= self.PLV(**reparam)
                fig= self.nplot.PLV_tr(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/phaseLockTR/', graphData= graphData)
                
                reparam.update({'mode': 'bs', 'nsample': 100})
                graphData= self.PLV(**reparam)
                fig= self.nplot.PLV_bs(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/phaseLockBS/', graphData= graphData)
                
                reparam.update({'mode': None})
                graphData= self.PLV(**reparam)
                fig= self.nplot.PLV(graphData)
                self.closefig(fig)
                self.plotData_to_hdf(name= name+ '/phaseLock/', graphData= graphData)
            except:
                logging.error('Error in spike-phase locking analysis')
                
            if self.getAnalysis('lfpSpikeCausality'):
#                logging.info('Assessing gridness...')
                logging.warning('Unit-LFP analysis has not been implemented yet!')
#        if self.getAnalysis('lfpSpikeCausality'):
#            print('do the lfpSpikeCausality analysis')       
        
# Manage Data
    def plotData_to_hdf(self, name= None, graphData= None):
        self.hdf.save_dict_recursive(path= '/analysis/', \
             name= name, data= graphData)
        
    def setNeuroData(self, ndata):
        if inspect.isclass(ndata):
            ndata= ndata()
        if isinstance(ndata, NData):
            self.ndata= ndata
        else:
            logging.warning('Inappropriate NeuroData object or class')
    def getNeuroData(self):
        return self.ndata
        
    def setConfiguration(self, config):
        if inspect.isclass(config):
            config= config()
        if isinstance(config, Configuration):
            self.__config= config
        else:
            logging.warning('Inappropriate Configuration object or class')
    def getConfiguration(self):
        return self.__config
# Forwarding to configuration class
    def __getattr__(self, arg):
        if hasattr(self.__config, arg):
            return getattr(self.__config, arg)
        elif hasattr(self.ndata, arg):
            return getattr(self.ndata, arg)
        else:
            logging.warning('No '+ arg+ ' method or attribute in NeuroChaT class')
    
    def convert_to_nwb(self, excel_file= None):
        if self.getDataFormat()== 'NWB':
            logging.error('NWB files do not need to be converted! Check file format option again!')
        info= {'spat': [], 'spike': [], 'lfp': []}
        export_info= oDict({'dir': [], 'nwb': [], 'spike': [], 'lfp': []})
        if os.path.exists(excel_file):
            excel_info = pd.read_excel(excel_file)
            for row in excel_info.itertuples():
                spikeFile= row[1]+ os.sep+ row[3]
                lfpID= row[4]
                
                if self.getDataFormat()== 'Axona':
                    spatialFile= row[1]+ os.sep+ row[2]+ '.txt'
                    lfpFile= ''.join(spikeFile.split('.')[:-1])+ '.'+ lfpID
                    
                elif self.getDataFormat()== 'Neuralynx':
                    spatialFile= row[1] + os.sep+ row[2]+ '.nvt'
                    lfpFile= row[1]+ os.sep+ lfpID+ '.ncs'
                                 
                info['spat'].append(spatialFile)
                info['spike'].append(spikeFile)
                info['lfp'].append(lfpFile)

        if info['spike']:
            for i, spike_file in enumerate(info['spike']):
    
                logging.info('Converting file groups: '+ str(i+ 1))
    #                try:
                self.ndata.setSpatialFile(info['spat'][i])
                self.ndata.setSpikeFile(info['spike'][i])
                self.ndata.setLfpFile(info['lfp'][i])
                self.ndata.load()
                self.ndata.save_to_hdf5()
                
                f_name= self.hdf.resolve_hdfname(data= self.ndata.spike)
                export_info['dir'].append(os.sep.join(f_name.split(os.sep)[:-1]))
                export_info['nwb'].append(f_name.split(os.sep)[-1].split('.')[0])
                
                export_info['spike'].append(self.hdf.get_file_tag(self.ndata.spike))
                export_info['lfp'].append(self.hdf.get_file_tag(self.ndata.lfp))   

        export_info= pd.DataFrame(export_info, columns= ['dir', 'nwb', 'spike', 'lfp'])
        words= excel_file.split(os.sep)
        name= 'NWB_list_' + words[-1]
        export_info.to_excel(os.path.join(os.sep.join(words[:-1]), name))
        logging.info('Conversion process completed!')
        
    def verify_units(self, excel_file= None):
        info= {'spike': [], 'unit': []}
        if os.path.exists(excel_file):
            excel_info = pd.read_excel(excel_file)
            for row in excel_info.itertuples():
                spikeFile= row[1]+ os.sep+ row[2]
                unit_no= int(row[3])
                if self.getDataFormat()== 'NWB':
                        # excel list: directory| spike group| unit_no
                        hdf_name= row[1] + os.sep+ row[2]+ '.hdf5'
                        spikeFile= hdf_name+ '/processing/Shank'+ '/'+ row[3]                 
                info['spike'].append(spikeFile)
                info['unit'].append(unit_no)
            n_units= excel_info.shape[0]
            
            excel_info= excel_info.assign(fileExists= pd.Series(np.zeros(n_units, dtype= bool)))
            excel_info= excel_info.assign(unitExists= pd.Series(np.zeros(n_units, dtype= bool)))
            
            if info['spike']:
                for i, spikeFile in enumerate(info['spike']):
        
                    logging.info('Verifying unit: '+ str(i+ 1))
                    if os.path.exists(spikeFile):
                        excel_info.loc[i, 'fileExists']= True
                        self.ndata.setSpikeFile(spikeFile)
                        self.ndata.loadSpike()
                        units= self.ndata.getUnitList()
                        
                        if info['unit'][i] in units:
                            excel_info.loc[i, 'unitExists']= True 
                            
            excel_info.to_excel(excel_file)
            logging.info('Verification process completed!')
        else:
            logging.error('Excel  file does not exist!')
            
    def cluster_evaluate(self, excel_file= None):
        info= {'spike': [], 'unit': []}
        if os.path.exists(excel_file):
            excel_info = pd.read_excel(excel_file)
            for row in excel_info.itertuples():
                spikeFile= row[1]+ os.sep+ row[2]
                unit_no= int(row[3])
                if self.getDataFormat()== 'NWB':
                        # excel list: directory| spike group| unit_no
                        hdf_name= row[1] + os.sep+ row[2]+ '.hdf5'
                        spikeFile= hdf_name+ '/processing/Shank'+ '/'+ row[3]                 
                info['spike'].append(spikeFile)
                info['unit'].append(unit_no)
            n_units= excel_info.shape[0]
        
            excel_info= excel_info.assign(BC= pd.Series(np.zeros(n_units)))
            excel_info= excel_info.assign(Dh= pd.Series(np.zeros(n_units)))
        
            if info['spike']:
                for i, spikeFile in enumerate(info['spike']):
        
                    logging.info('Evaluating unit: '+ str(i+ 1))
                    if os.path.exists(spikeFile):
                        self.ndata.setSpikeFile(spikeFile)
                        self.ndata.loadSpike()
                        units= self.ndata.getUnitList()
                        
                        if info['unit'][i] in units:
                            nclust= NClust(spike= self.ndata.spike)
                            bc, dh= nclust.cluster_separation(unit_no= info['unit'][i])
                            excel_info.loc[i, 'BC']= np.max(bc)
                            excel_info.loc[i, 'Dh']= np.min(dh)
            excel_info.to_excel(excel_file)
            logging.info('Cluster evaluation completed!')
        else:
            logging.error('Excel  file does not exist!')
            
    def cluster_similarity(self, excel_file= None):
        # test on Pawel's data
        nclust_1= NClust()
        nclust_2= NClust()
        info= {'spike_1': [], 'unit_1': [], 'spike_2': [], 'unit_2': []}
        if os.path.exists(excel_file):
            excel_info = pd.read_excel(excel_file)
            for row in excel_info.itertuples():
                spikeFile= row[1]+ os.sep+ row[2]
                unit_1= int(row[3])
                if self.getDataFormat()== 'NWB':
                        # excel list: directory| spike group| unit_no
                        hdf_name= row[1] + os.sep+ row[2]+ '.hdf5'
                        spikeFile= hdf_name+ '/processing/Shank'+ '/'+ row[3]                 
                info['spike_1'].append(spikeFile)
                info['unit_1'].append(unit_1)
                
                spikeFile= row[4]+ os.sep+ row[5]
                unit_2= int(row[6])
                if self.getDataFormat()== 'NWB':
                        # excel list: directory| spike group| unit_no
                        hdf_name= row[4] + os.sep+ row[5]+ '.hdf5'
                        spikeFile= hdf_name+ '/processing/Shank'+ '/'+ row[6]                 
                info['spike_2'].append(spikeFile)
                info['unit_2'].append(unit_2)
                
            n_comparison= excel_info.shape[0]
        
            excel_info= excel_info.assign(BC= pd.Series(np.zeros(n_comparison)))
            excel_info= excel_info.assign(Dh= pd.Series(np.zeros(n_comparison)))
        
            if info['spike_1']:
                for i in np.arange(n_comparison):
                    logging.info('Evaluating unit similarity row: '+ str(i+ 1))
                    if os.path.exists(info['spike_1']) and os.path.exists(info['spike_2']):
                        nclust_1.load(filename= info['spike_1'], system= self.getDataFormat())
                        nclust_2.load(filename= info['spike_2'], system= self.getDataFormat())
                        bc, dh= nclust_1.cluster_similarity(nclust= nclust_2, \
                                unit_1= info['unit_1'][i], unit_2= info['unit_2'][i])
                        excel_info.loc[i, 'BC']= bc
                        excel_info.loc[i, 'Dh']= dh
            excel_info.to_excel(excel_file)
            logging.info('Cluster similarity analysis completed!')
        else:
            logging.error('Excel  file does not exist!')