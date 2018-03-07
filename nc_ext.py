# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 19:43:32 2017

@author: Raju
"""
from PyQt5 import QtWidgets, QtCore
import logging
import pandas as pd
import numpy as np
import numpy.linalg as nalg
from collections import OrderedDict as oDict
import scipy
import scipy.stats as stats
import scipy.signal as sg
from scipy.fftpack import fft
import os.path
import time

import yaml
#from yaml import CLoader as Loader, CDumper as Dumper
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
def __dict_representer(dumper, data):   
    return dumper.represent_mapping(_mapping_tag, data.items())

def __dict_constructor(loader, node):
    return oDict(loader.construct_pairs(node))
yaml.add_representer(oDict, __dict_representer)
yaml.add_constructor(_mapping_tag, __dict_constructor)   

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s


def find(X, n=None, direction= 'all' ):
    if isinstance(X, list):
        X= np.array(X)
    X= X.flatten()
    if n is None:
        n= len(X)
    ind= np.where(X)[0]
    if ind.size:
        if direction== 'all' or direction== 'first':
            ind= ind[:n]
        elif direction== 'last':
            ind= ind[np.flipud(np.arange(-1, -(n+1), - 1))]
    return np.array(ind)

def find2d(X, n= None):
    if len(X.shape)== 2:
        J= []
        I= []
        for r in np.arange(X.shape[0]):
            I.extend(find(X[r, ]))
            J.extend(r*np.ones((len(find(X[r, ])), ), dtype= int))            
        if len(I):
            if n is not None and n< len(I):
                I= I[:n]
                J= J[:n]
        return np.array(J), np.array(I)
            
    else:
        logging.error('ndrray is not 2D. Check shape attributes of the input!')
            
    
def residualStat(y, y_fit, p):
   # p= total explanatory variables excluding constants
    _results= oDict()
    res= y- y_fit
    ss_res= np.sum(res**2)
    ss_tot= np.sum((y- np.mean(y))**2)
    r_sq= 1- ss_res/ss_tot
    adj_r_sq= 1- (ss_res/ ss_tot)* ((len(y)-1)/(len(y)- p-1))
    _results['Pearson R'], _results['Pearson P']= stats.pearsonr(y, y_fit)
    
    _results['Rsq']= r_sq 
    _results['adj Rsq']= adj_r_sq
    
    return _results
    
#def findpeaks(data, **kwargs):
#    data= np.array(data)
#    slope= np.diff(data)
#    start_at= kwargs.get('start', 0)
#    end_at= kwargs.get('end', slope.size)
#    thresh= kwargs.get('thresh', 0)
#
#    peak_loc= [j for j in np.arange(start_at, end_at-1) \
#                if slope[j] > 0 and slope[j+1] <= 0]
#    peak_val= [data[peak_loc[i]] for i in range(0, len(peak_loc))]
#    
#    valid_loc= [i for i in range(0, len(peak_loc)) if peak_val[i] >= thresh]
#    peak_val, peak_loc= zip(*((peak_val[i], peak_loc[i]) for i in range(0, len(valid_loc))))
#    
#    return np.array(peak_val), np.array(peak_loc)
    
def extrema(x, mincap= None, maxcap= None):
    x= np.array(x)
    # Flat peaks at the end of the series are not considered yet
    dx= np.diff(x)
    if not np.any(dx):
        return [], [], [], []
        
    a= find(dx!=0) # indices where x changes
    lm= find(np.diff(a)!=1)+1 # indices where a is not sequential
    d= a[lm]- a[lm-1]
    a[lm]= a[lm]- np.floor(d//2)
    
    xa= x[a] # series without flat peaks
    d= np.sign(xa[1:-1]- xa[:-2])- np.sign(xa[2:]- xa[1:-1])
    imax= a[find(d > 0)+1]
    xmax= x[imax]
    imin= a[find(d < 0)+1]
    xmin= x[imin]

    if mincap:
        imin= imin[xmin<= mincap]
        xmin= xmin[xmin<= mincap]
    if maxcap:
        imax= imax[xmax<= maxcap]
        xmax= xmax[xmax<= maxcap]
        
    return xmax, imax, xmin, imin
    
def bhatt(X1, X2):
    r1, c1= X1.shape
    r2, c2= X2.shape
    if c1== c2:
        mu1= X1.mean(axis= 0)
        mu2= X2.mean(axis= 0)
        C1= np.cov(X1.T)
        C2= np.cov(X2.T)
        C= (C1+ C2)/2
        chol= nalg.cholesky(C).T
        dmu= (mu1- mu2)@nalg.inv(chol)
        try:
            d= 0.125*dmu@(dmu.T)+ 0.5*np.log(nalg.det(C)/np.sqrt(nalg.det(C1)*nalg.det(C2)))
        except:
            d= 0.125*dmu@(dmu.T)+ 0.5*np.log(np.abs(nalg.det(C@nalg.inv(scipy.linalg.sqrtm(C1@C2)))))
        bc= np.exp(-1*d)
        
        return bc, d
    else:
        logging.error('Cannot measure Bhattacharyya distance, column sizes do not match!')
        
def hellinger(X1, X2):
    if X1.shape[1]!= X2.shape[1]:
        logging.error('Hellinger distance cannot be computed, column sizes do not match!')
    else:
        return np.sqrt(1- bhatt(X1, X2)[0])
    
def butter_filter(x, Fs, *args):
    gstop= 20 # minimum dB attenuation at stopabnd
    gpass= 3 # maximum dB loss during ripple
    order= args[0]
    for arg in args:
        if isinstance(arg, str):
            filttype= arg
    if filttype== 'lowpass' or filttype== 'highpass':
        wp= args[1]/(Fs/2)
        if wp>1:
            wp= 1
            if filttype== 'lowpass':
                logging.warning('Butterworth filter critical freqeuncy Wp is capped at 1')
            else:
                logging.error('Cannot highpass filter over Nyquist frequency!')
    
    elif filttype== 'bandpass':
        if len(args)<4:
            logging.error('Insufficient Butterworth filter arguments')
        else:
            wp= np.array(args[1:3])/(Fs/2)
            if wp[0]>= wp[1]:
                logging.error('Butterworth filter lower cutoff frequency must be smaller than upper cutoff freqeuncy!')
            
            if wp[0]== 0 and wp[1]>= 1:
                logging.error('Invalid filter specifications, check cutt off frequencies and sampling frequency!')
            elif wp[0]==0:
                wp= wp[1]
                filttype= 'lowpass'
                logging.warning('Butterworth filter type selected: lowpass')
            elif wp[1]>=1:
                wp= wp[0]
                filttype= 'highpass'
                logging.warning('Butterworth filter type selected: highpass')
    
    if filttype== 'lowpass':
        ws= min([wp+ 0.1, 1])
    elif filttype== 'highpass':
        ws= max([wp- 0.1, 0.01/(Fs/2)])
    elif filttype== 'bandpass':
        ws= np.zeros_like(wp)
        ws[0]= max([wp[0]- 0.1, 0.01/(Fs/2)])
        ws[1]= min([wp[1]+ 0.1, 1])

    min_order, min_wp= sg.buttord(wp, ws, gpass, gstop)
#    if order<= min_order:
#        order= min_order
#        wp= min_wp
        
    b, a= sg.butter(min_order, min_wp, btype= filttype, output= 'ba') 
    
    return sg.filtfilt(b, a, x)

def smooth1D(x, filttype= 'b', filtsize= 5, **kwargs):
    x= np.array(x)    
    direction= kwargs.get('dir', 0) # default along column
    if filttype== 'g':
        halfwid= np.round(3*filtsize)
        xx = np.arange(-halfwid, halfwid+1, 1)
        filt= np.exp(-(xx**2)/(2*filtsize**2))/(np.sqrt(2*np.pi)*filtsize)
    elif filttype== 'b':
        filt= np.ones(filtsize, )/filtsize
    
    if len(x.shape)==1:
        result= np.convolve(x, filt, mode= 'same')
    elif len[x.shape]==2:
        result= np.zeros(x.shape)
        if direction:
            for i in np.arange(0, x.shape[0]):
                result[i, :]= result= np.convolve(x[i, :], filt, mode= 'same')
        else:
            for i in np.arange(0, x.shape[0]):
                result[:, i]= result= np.convolve(x[:, i], filt, mode= 'same')
    return result
        
def smooth2d(x, filttype= 'b', filtsize= 5):
    nanInd= np.isnan(x)
    x[nanInd]= 0
    if filttype== 'g':
        halfwid= np.round(3*filtsize)
        xx, yy = np.meshgrid(np.arange(-halfwid, halfwid+1, 1), np.arange(-halfwid, halfwid+1, 1), copy= False)
        filt= np.exp(-(xx**2+ yy**2)/(2*filtsize**2)) # /(2*np.pi*filtsize**2) # This is the scaling used before; 
                                                        #But tested with ones(50, 50); gives a hogher value
        filt= filt/ np.sum(filt)
    elif filttype== 'b':
        filt= np.ones((filtsize,filtsize))/filtsize**2
        
    smoothX= sg.convolve2d(x, filt, mode= 'same')
    smoothX[nanInd]= np.nan
    
    return smoothX

def corrCoeff(x1, x2):
    try:
        return np.sum(np.multiply(x1- x1.mean(), x2- x2.mean()))/ \
            np.sqrt(np.sum((x1- x1.mean())**2)*np.sum((x2- x2.mean())**2))
    except:
        return 0
    
def rot2d(x, theta):
    return scipy.ndimage.interpolation.rotate(x, theta, reshape= False, mode= 'constant', cval= np.min(x))
    
def histogram(x, bins):
    # This function is not a replacement of np.histogram; it is created for convenience
    # of binned-based rate calculations and mimicking matlab histc that includes digitized indices
    if isinstance(bins, int):
        bins= np.arange(np.min(x), np.max(x), (np.max(x)- np.min(x))/bins)
    bins= np.append(bins, bins[-1]+ np.mean(np.diff(bins)))        
    return np.histogram(x, bins)[0], np.digitize(x, bins)-1, bins[:- 1]

def histogram2d(y, x, ybins, xbins):
    # This function is not a repalcement of np.histogram
    if isinstance(xbins, int):
        xbins= np.arange(np.min(x), np.max(x), (np.max(x)- np.min(x))/xbins)
    xbins= np.append(xbins, xbins[-1]+ np.mean(np.diff(xbins)))
    if isinstance(ybins, int):
        ybins= np.arange(np.min(y), np.max(y), (np.max(y)- np.min(y))/ybins)
    ybins= np.append(ybins, ybins[-1]+ np.mean(np.diff(ybins)))

    return np.histogram2d(y, x, [ybins, xbins])[0], ybins[:-1], xbins[:-1]
    
    
def chopEdges(x, xlen, ylen):
    y= np.copy(x)
    low_ind= [0, 0]
    high_ind= [x.shape[0], x.shape[1]]
    
    MOVEON= True
    while y.shape[1]> xlen and MOVEON:
        no_filled_bins1= np.sum(y[:, 0]> 0)
        no_filled_bins2= np.sum(y[:, -1]> 0)
        
        if no_filled_bins1== 0:
            low_ind[1] +=1
            MOVEON= True
        else:
            MOVEON= False
        if no_filled_bins2== 0:
            high_ind[1]-=1
            MOVEON= True
        else:
            MOVEON= False      
            
# Following is the old MATLAB logic, we have changed it to remove the edges with zero count
#        if no_filled_bins1< no_filled_bins2:
#            low_ind[1] +=1
#        else:
#            high_ind[1]-=1
        y= x[low_ind[0]: high_ind[0], low_ind[1]:high_ind[1]]

    MOVEON= True        
    while y.shape[0]> ylen and MOVEON:
        no_filled_bins1= np.sum(y[0, :]> 0)
        no_filled_bins2= np.sum(y[-1, :]> 0)

        if no_filled_bins1== 0:
            low_ind[0] +=1
            MOVEON= True
        else:
            MOVEON= False
        if no_filled_bins2== 0:
            high_ind[0]-=1
            MOVEON= True
        else:
            MOVEON= False
            
# Following is the old MATLAB logic, we have changed it to remove the edges with zero count            
#        if no_filled_bins1< no_filled_bins2:
#            low_ind[0] +=1
#        else:
#            high_ind[0]-=1
        y= x[low_ind[0]: high_ind[0], low_ind[1]:high_ind[1]]
        
    return low_ind, high_ind, y
     
def fft_psd(x, Fs, nfft= None, side= 'one', ptype= 'psd'):
    
    if nfft is None:
        nfft= 2**(np.floor(np.log2(len(x)))+1)
        
    if nfft< Fs:
        nfft= 2**(np.floor(np.log2(Fs))+1)
    nfft= int(nfft)
    dummy= np.zeros(nfft)
    if nfft> len(x):
        dummy[:len(x)]= x
        x= dummy    

    winfun= np.hanning(nfft)    
    xf= np.arange(0, Fs, Fs/nfft)
    f= xf[0: int(nfft/2)+ 1]


    if side== 'one':
        x_fft= fft(np.multiply(x, winfun), nfft)        
        if ptype== 'psd':
            x_fft= np.absolute(x_fft[0: int(nfft/2)+ 1])**2/nfft**2
            x_fft[1:-1]= 2*x_fft[1:-1]
            
    return x_fft, f

def linfit(X, Y, getPartial= False):
    _results= oDict()
    if len(X.shape)==2:
        Nd, Nobs= X.shape
    else:
        Nobs= X.shape[0]
        Nd= 1
    if Nobs== len(Y):        
        A= np.vstack([X, np.ones(X.shape[0])]).T
        B= np.linalg.lstsq(A, Y)[0]
        Y_fit= np.matmul(A, B)       
        _results['coeff']= B[:-1]
        _results['intercept']= B[-1]        
        _results['yfit']= Y_fit
        _results.update(residualStat(Y, Y_fit, 1))
    else:
            logging.error('linfit: Number of rows in X and Y does not match!')
            
    if Nd>1 and getPartial:
        semiCorr= np.zeros(Nd) # Semi partial correlation
        for d in np.arange(Nd):
            part_results= linfit(np.delete(X, 1, axis= 0), Y, getPartial= False)
            semiCorr[d]= _results['Rsq']- part_results['Rsq']
        _results['semiCorr']= semiCorr
    
    return _results

def findChunk(x):
    # x is a binary array input i.e. x= data> 0.5 will find all the chunks in data where data is greater than 0.5
    i= 0
    segsize= []
    segind= np.zeros(x.shape)
    while i< len(x):
        if x[i]:
            c=0
            j= i
            while i< len(x):
                if x[i]:                            
                    c+=1
                    i+=1
                else:
                    break
            segsize.append(c)
            segind[j:i]= c # indexing by size of the chunk
        i+=1    
    return segsize, segind
 
class CircStat(object):
    def __init__(self, **kwargs): #Currently supports 'deg'. Will be extended for 'rad'
        self._rho= kwargs.get('rho', None)
        self._theta= kwargs.get('theta', None)
        self._result= oDict()
    
    def setRho(self, rho= None):
        if rho is not None:
            self._rho= rho
    def getRho(self):
        return self._rho
        
    def setTheta(self, theta= None):
        if theta is not None:
            self._theta= theta
    def getTheta(self):
        return self._theta
        
    def getMeanStd(self):
        if self._rho is None or not len(self._rho):
            self._rho= np.ones(self._theta.shape)
        return self._calcMeanStd()

    def _calcMeanStd(self):
        result= {}
        if self._rho.shape[0]== self._theta.shape[0]:
            xm= np.sum(np.multiply(self._rho, np.cos(self._theta*np.pi/180)))
            ym= np.sum(np.multiply(self._rho, np.sin(self._theta*np.pi/180)))
            meanTheta= np.arctan2(ym, xm)* 180/np.pi            
            if meanTheta< 0:
                meanTheta= meanTheta+ 360
            meanRho= np.sqrt(xm**2+ ym**2)
            result['meanTheta']= meanTheta
            result['meanRho']= meanRho
            result['totalObs']= np.sum(self._rho)
            result['resultant']= meanRho/result['totalObs']
            try:
                x= -2*np.log(result['resultant'])
                if x< 0:
                    result['stdRho']= 0
                else:
                    result['stdRho']= np.sqrt()
            except:
                result['stdRho']= 0 # This except to proetct -ve inside sqrt
            
        else:
            logging.warning('Size of rho and theta must be equal')
            
        return result
        
    def getRaylStat(self):
        return self._raylStat()
        
    def _raylStat(self):
        result= {}
        N= self._result['totalObs']
        Rn= self._result['resultant']*N
        result['RaylZ']= Rn**2/ N
        result['RaylP']= np.exp(np.sqrt(1+ 4*N+ 4* (N**2- Rn**2))- (1+2*N))
        
        return result

    def getVonMissesStat(self):
        return self._vonMissesStat()
        
    def _vonMissesStat(self):
        result= {}
        R= self._result['resultant']
        N= self._result['totalObs']
        
        if R< 0.53:
            kappa= 2*R+ R**3+ 5*(R**5)/6
        elif R<= 0.53 and R< 0.85:
            kappa= -0.4+ 1.39*R+ 0.43/(1-R)
        else:
            kappa= 1/ (R**3- 4*R**2+ 3*R)
        
        if N< 15 and  N> 1:
            kappa= max(kappa- 2*(N*kappa)**-1, 0) if kappa< 2 else kappa*(N-1)**3/ (N**3+ N)
        
        result['vonMissesK']= kappa
        return result
        
    def calcStat(self):
        result= self._calcMeanStd()
        self._updateResult(result)
        result=  self._raylStat()
        self._updateResult(result)
        result= self._vonMissesStat()
        self._updateResult(result)
        
        return self.getResult()
    
    @staticmethod
    # Example, x = [270, 340, 350, 20, 40], y=  [270, 340, 350, 380, 400] etc.
    def circRegroup(x):
        y= np.copy(x)
        if any(np.logical_and(x>=0, x<= 90)) and any(np.logical_and(x>= 180, x<= 360)):
            y[np.logical_and(x>=0, x<= 90)]= x[np.logical_and(x>=0, x<= 90)]+ 360 
        return y
    
    def circHistogram(self, bins= 5):
        if isinstance(bins, int):
            bins= np.arange(0, 360, bins)
            
        nbins= bins.shape[0]
        count= np.zeros(bins.shape)
        ind= np.zeros(self._theta.shape, dtype= int)
        for i in np.arange(nbins):
            if i< nbins-1:
                ind[np.logical_and(self._theta>= bins[i], self._theta< bins[i+1])]= i
                count[i]= np.sum(np.logical_and(self._theta>= bins[i], self._theta< bins[i+1]))
                
            elif i== nbins-1:
                ind[np.logical_or(self._theta>= bins[i], self._theta< bins[0])]= i
                count[i]= np.sum(np.logical_or(self._theta>= bins[i], self._theta< bins[0]))
            
        return count, ind, bins

    def circSmooth(self, filttype= 'b', filtsize= 5):
        
        if filttype== 'g':
            halfwid= np.round(3*filtsize)
            xx = np.arange(-halfwid, halfwid+1, 1)
            filt= np.exp(-(xx**2)/(2*filtsize**2))/(np.sqrt(2*np.pi)*filtsize)
        elif filttype== 'b':
            filt= np.ones(filtsize, )/filtsize
     
        cs= CircStat();
        
        smoothTheta= np.zeros(self._theta.shape)
        N= self._theta.shape[0]
        L= filt.shape[0]
        l= int(np.floor(L/2))
        for i in np.arange(l):
            cs.setRho(filt[l- i:])
            cs.setTheta(self.circRegroup(self._theta[:L-l+ i]))
            csResult= cs.getMeanStd()
            smoothTheta[i]= csResult['meanTheta']
        for i in np.arange(l, N- l, 1):
            cs.setRho(filt)
            cs.setTheta(self.circRegroup(self._theta[i-l:i+l+ 1]))
            csResult= cs.getMeanStd()
            smoothTheta[i]= csResult['meanTheta']
        for i in np.arange(N- l, N):        
            cs.setTheta(self.circRegroup(self._theta[i- l:]))
            cs.setRho(filt[:len(self._theta[i- l:])])
            csResult= cs.getMeanStd()
            smoothTheta[i]= csResult['meanTheta']
            
        return smoothTheta
        
    def circScatter(self, bins= 2, step= 0.05, rmax= None):
        # Prepares the data for scatter plot.        
        count, ind, bins = self.circHistogram(bins= 2)
        radius= np.ones(ind.shape)
        theta= np.zeros(ind.shape)
        for i, b in enumerate(bins):
            rad= np.ones(find(ind== i).shape)+ np.array(list(step*j for j, loc in enumerate(find(ind==i))))
            if rmax:
                rad[rad> rmax]= rmax
            radius[ind== i]= rad
            theta[ind== i]= b
            
        return radius, theta
        
    def _updateResult(self, newResult= {}):
        self._result.update(newResult)
    
    def getResult(self):
        return self._result
                
class NLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setup()
    def setup(self):
        log = logging.getLogger()
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        fmt=logging.Formatter('%(asctime)s (%(filename)s)  %(levelname)s--  %(message)s', '%H:%M:%S')
        self.setFormatter(fmt)
        log.addHandler(self)
        # You can control the logging level
        log.setLevel(logging.DEBUG)
        logging.addLevelName(20, '')
    def emit(self, record):
        msg= self.format(record)
        level= record.levelname
        msg= level+ ':'+ msg
        print(msg)
        time.sleep(0.25)
#        self.emit(QtCore.SIGNAL('update_log(QString)'), msg)

class NLogBox(QtWidgets.QTextEdit):
    def __init__(self, parent= None):
        super().__init__(parent)
    def insertLog(self, msg):
        level= msg.split(':')[0].upper()
#        level= "WARNING"
        if level== "WARNING":
            color= "darkorange"
        elif level== "ERROR":
            color= "red"
        elif level== "INFO":
            color= "black"
        else:
            color= "blue"
        msg = '<font color=' + color + '>' + msg[msg.find(":")+1:] + '</font><br>'
        self.insertHtml(msg)
    def getText(self):
        return self.toPlainText()

class Configuration(object):
    
    'Advantages of YAML: It combines the simplicity of .properties or .ini files \
    (simplest YAML file is just a list of colon separated key-value pairs and \
    nothing else) with the ability to represent more complex data structures. \
    It is almost as flexible as JSON when it comes to arranging data in hierarchical\
    relationships, but its syntax is much simpler and less restrictive. \
    As a result files are more robust and less prone to breaking when edited manually by hand--- Reword it'
    def __init__(self, filename= []):
        self.filename= filename
        self.format= 'Axona'
        self.analysis_mode= 'Single Unit'
        self.mode_id= 0
        self.graphic_format= 'pdf'
        self.valid_graphics= {'PDF': 'pdf', 'Postscript': 'ps'}
        self.unit_no= 0
        self.cell_type= ''
        self.spike_file= ''
        self.spatial_file= ''
        self.lfp_file= ''
        self.nwb_file= ''
        self.excel_file= ''
        self.data_directory= ''
        self.config_directory= ''
        self.analyses= oDict()
        self.parameters= {}
        self.options= oDict()
        self.mode_dict= oDict([('Single Unit', 0),
            ('Single Session', 1), 
            ('Listed Units', 2)])
#        self.mode_dict= oDict([('Single Unit', 0),
#            ('Single Session', 1), 
#            ('Listed Units', 2),
#            ('Multiple Sessions', 3)])

    def setParam(self, name= None, value= None):
        if isinstance(name, str):
            self.parameters[name]= value
        else:
            logging.error('Parameter name and/or value is inappropriate')
    
    def getParams(self, name= None):
        if isinstance(name, list):
            params=  {}
            not_found= []
            for pname in name:
                if pname in self.parameters.keys():
                    params[pname]= self.parameters[pname]
                else:
                    not_found.append(pname)
            if not_found:
                logging.warning('Following parameters not found- '+ ','.join(not_found))
            return params
        elif isinstance(name, str):
            if name in self.parameters.keys():
                return self.parameters[name]
            else:
                logging.error(name+ ' is not found in parameter list')

    def setAnalysis(self, name= None, value= None):
            
        if isinstance(name, str) and isinstance(value, bool):
            self.analyses[name]= value
        else:
            logging.error('Parameter name and/or value is inappropriate')
            
    def getAnalysis(self, name= None):
        if name== 'all':
            return self.analyses.values()   
        elif name in self.analyses.keys():
            return self.analyses[name]
        else:
            logging.error(name+ ' is not found in parameter list')

    def getParamList(self):
        return self.parameters.keys()
    def getAnalysisList(self):
        return self.analyses.keys()
        
    def setDataFormat(self, file_format= None):
        if file_format:
            self.format= file_format
    def getDataFormat(self):
        return self.format

    def setAnalysisMode(self, analysis_mode= None):
        if analysis_mode in self.mode_dict.keys():
            self.analysis_mode= analysis_mode
            self.mode_id= self.mode_dict[analysis_mode]
        elif analysis_mode in self.mode_dict.values():
            self.mode_id= analysis_mode
            for key, val in self.mode_dict.items():
                if val==  analysis_mode:
                    self.analysis_mode= key
        else:
            logging.error('No/Invalid analysis mode!')
    def getAnalysisMode(self):
        return self.analysis_mode, self.mode_id
        
    def getAllModes(self):
        return self.mode_dict

    def setGraphicFormat(self, graphic_format= None):
        if graphic_format in self.valid_graphics.keys():
            self.graphic_format= self.valid_graphics[graphic_format]
        else:
            logging.error('No/Invalid graphic format!')
    def getGraphicFormat(self):
        return self.graphic_format
    
    def setUnitNo(self, unit_no= None):
        if  isinstance(unit_no, int):
            self.unit_no= unit_no
    def getUnitNo(self):
        return self.unit_no
        
    def setCellType(self, cell_type= None):
        self.cell_type= cell_type

    def getCellType(self):
        return self.cell_type
        
    def setSpikeFile(self, spike_file= None):
        if isinstance(spike_file, str):
            self.spike_file =spike_file
    def getSpikeFile(self):
        return self.spike_file
        
    def setSpatialFile(self, spatial_file= None):
        if isinstance(spatial_file, str):
            self.spatial_file =spatial_file
    def getSpatialFile(self):
        return self.spatial_file
    
    def setLfpFile(self, lfp_file= None):
        if isinstance(lfp_file, str):
            self.lfp_file =lfp_file
    def getLfpFile(self):
        return self.lfp_file
        
    def setNwbFile(self, nwb_file= None):
        if isinstance(nwb_file, str):
            self.nwb_file =nwb_file
    def getNwbFile(self):
        #if self.directory+ self.nwb_file exist, return, else create the name
        # self.creatNwbFile()
        return self.nwb_file
    def _createNwbFile(self):
        print('Create nwb filename from other filenames if not exists')
        
    def getExcelFile(self):
        return self.excel_file
    def setExcelFile(self, excel_file= None):
        # Check if this is a valid filename
        if excel_file:
            self.excel_file= excel_file
        else:
            logging.error('Invalid/No excel filename specified')
    
    def setDataDir(self, directory= None):
        # if if this is a valid directory
        if os.path.exists(directory):
            self.data_directory= directory
        else:
            logging.error('Invalid/No directory specified')            
    def getDataDir(self):
        return self.data_directory
        
    def setConfigDir(self, directory= None):
         # if if this is a valid directory
        if os.path.exists(directory):
            self.config_directory= directory
        else:
            logging.error('Invalid/No directory specified')                        
    def getConfigDir(self):
        return self.config_directory
        
    def setConfigFile(self, filename):
        self.filename= filename
    def getConfigFile(self):
        return self.filename                
    def saveConfig(self, filename= None):
        if filename:
            self.setConfigFile(filename)
        if not self.filename:
            logging.warning('No/invalid filename')
        else:
            # elseif verify valid filename try to save (through error) else error
            self._save()
    def loadConfig(self, filename= None):
        if filename:
            self.setConfigFile (filename)
        if not self.filename:
            logging.warning('No/Invalid filename')
        else:
            # elseif verify valid filename try to save (through error) else error
            self._load()             
            
    def _save(self):
        try:           
            with open(self.filename, 'w') as f:
                settings= oDict([('format', self.format),
                                 ('analysis_mode', self.analysis_mode),
                                 ('mode_id', self.mode_id),
                                 ('graphic_format', self.graphic_format),
                                 ('unit_no', self.unit_no),
                                 ('lfp_file', self.lfp_file),
                                 ('cell_type', self.cell_type),
                                 ('spike_file', self.spike_file),
                                 ('spatial_file', self.spatial_file),
                                 ('nwb_file', self.nwb_file),
                                 ('excel_file', self.excel_file),
                                 ('data_directory', self.data_directory)])

                cfgData= oDict([('settings', settings),
                                ('analyses', self.analyses),
                                ('parameters', self.parameters)])
                                
                yaml.dump(cfgData, f, default_flow_style= False)
        except:
            logging.error('Configuration cannot be saved in the specified file!');
            
    def _load(self):
         with open(self.filename, 'r') as f:
             cfgData= yaml.load(f)
             settings= cfgData.get('settings')
             for key, val in settings.items():
                 self.__setattr__(key, val)
             self.analyses= cfgData.get('analyses')
             self.parameters= cfgData.get('parameters')
                             
def Nxlwrite(file_name, data_frame, sheet_name= 'Sheet1', startRow= 0, startColumn= 0):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    data_frame.to_excel(writer, sheet_name)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
        
def addRadioButton(parent, position, objectName, text):
    button= QtWidgets.QRadioButton(parent)
    button.setGeometry(QtCore.QRect(*position))
    button.setObjectName(_fromUtf8(objectName))
    button.setText(text)
    return button

def addPushButton(parent, position, objectName, text):
    button= QtWidgets.QPushButton (parent)
    button.setGeometry(QtCore.QRect(*position))
    button.setObjectName(_fromUtf8(objectName))
    button.setText(text)
    return button
    
def addCheckBox(parent, position, objectName, text):
    box= QtWidgets.QCheckBox(parent)
    box.setGeometry(QtCore.QRect(*position))
    box.setObjectName(_fromUtf8(objectName))
    box.setText(text)
    return box

def addComboBox(parent, position, objectName):
    box= QtWidgets.QComboBox(parent)
    box.setGeometry(QtCore.QRect(*position))
    box.setObjectName(_fromUtf8(objectName))
    return box
    
def addLabel(parent, position, objectName, text):
    label= QtWidgets.QLabel(parent)
    label.setGeometry(QtCore.QRect(*position))
    label.setObjectName(_fromUtf8(objectName))
    label.setText(text)
    return label    

def addLineEdit(parent, position, objectName, text):
    line= QtWidgets.QLineEdit(parent)
    line.setGeometry(QtCore.QRect(*position))
    line.setObjectName(_fromUtf8(objectName))
    line.setText(text)
    return line

def addTextEdit(parent, position, objectName):
    textEdit= NTextEdit(parent)
    textEdit.setGeometry(QtCore.QRect(*position))
    textEdit.setObjectName(_fromUtf8(objectName))
    return textEdit
    
def addGroupBox(parent, position, objectName, title):
    groupBox= QtWidgets.QGroupBox(parent)
    groupBox.setGeometry(QtCore.QRect(*position))
    groupBox.setObjectName(_fromUtf8(objectName))
    groupBox.setTitle(title)
    return groupBox

def addWidget(parent, position, objectName):
    widget= QtWidgets.QGroupBox(parent)
    widget.setGeometry(QtCore.QRect(*position))
    widget.setObjectName(_fromUtf8(objectName))
    return widget
    
def addSpinBox(parent, position, objectName, min_val, max_val):
    box= QtWidgets.QSpinBox(parent)
    box.setGeometry(QtCore.QRect(*position))
    box.setObjectName(_fromUtf8(objectName))
    box.setMinimum(min_val)
    box.setMaximum(max_val)
    return box
    
    
def addRadioButton_2(text, objectName= ""):
    button= QtWidgets.QRadioButton()    
    button.setObjectName(_fromUtf8(objectName))
    button.setText(text)
    return button

def addPushButton_2(text, objectName= ""):
    button= QtWidgets.QPushButton ()
    button.setObjectName(_fromUtf8(objectName))
    button.setText(text)
    return button
    
def addCheckBox_2(text, objectName= ""):
    box= QtWidgets.QCheckBox()
    box.setObjectName(_fromUtf8(objectName))
    box.setText(text)
    return box

def addComboBox_2(objectName= ""):
    box= QtWidgets.QComboBox()
    box.setObjectName(_fromUtf8(objectName))
    return box
    
def addLabel_2(text, objectName= ""):
    label= QtWidgets.QLabel()
    label.setObjectName(_fromUtf8(objectName))
    label.setText(text)
    return label

def addLineEdit_2(text, objectName= ""):
    line= QtWidgets.QLineEdit()
    line.setObjectName(_fromUtf8(objectName))
    line.setText(text)
#    line.resize(line.minimumSizeHint())
    return line

def addTextEdit_2(objectName):
    textEdit= NTextEdit()
    textEdit.setObjectName(_fromUtf8(objectName))
    return textEdit
    
def addGroupBox_2(title, objectName= ""):
    groupBox= QtWidgets.QGroupBox()
    groupBox.setObjectName(_fromUtf8(objectName))
    groupBox.setTitle(title)
    return groupBox

def addWidget_2(objectName= ""):
    widget= QtWidgets.QGroupBox()
    widget.setObjectName(_fromUtf8(objectName))
    return widget
    
def addSpinBox_2(min_val, max_val, objectName= ""):
    box= QtWidgets.QSpinBox()
    box.setObjectName(_fromUtf8(objectName))
    box.setMinimum(min_val)
    box.setMaximum(max_val)
    return box
    
def addDoubleSpinBox_2(min_val, max_val, objectName= ""):
    box= QtWidgets.QDoubleSpinBox()
    box.setObjectName(_fromUtf8(objectName))
    box.setMinimum(min_val)
    box.setMaximum(max_val)
    return box

def addLogBox(objectName):
    logTextBox= NLogBox()
#    logTextBox.seObjectName(_fromUtf8(objectName))
    return logTextBox
#    return logTextBox.widget
    
import matplotlib.pyplot as plt
import itertools
def scatterplot_matrix(data, names= [], **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[y,x].scatter(data[x], data[y], **kwargs)

    # Label the diagonal subplots...
    if len(names)== numvars:
        for i, label in enumerate(names):
            axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)
